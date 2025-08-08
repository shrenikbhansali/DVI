# train_dvi.py
#
# Draft-Verify-Improve trainer
# ------------------------------------------------------------
# • loads an Early-Exit Llama
# • injects dual-LoRA (fast LoRA-S, frozen LoRA-D)
# • rolls out speculative steps, stores accepted tokens in a buffer
# • REINFORCE update on the shallow draft head + LoRA-S
#
#   L(θ) = – (reward – baseline) · log π_θ
#
# The script is CPU-testable (tiny configs) yet GPU-aware:
# if CUDA is available we honour multi-GPU via HF `device_map="auto"`.

import argparse, random
from typing import List, Optional

import torch, torch.nn as nn
from transformers import AutoTokenizer

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import inject_dual_lora, enable_lora_grads
from training.buffer      import ReplayBuffer


# ------------------------------------------------------------------
# Model utilities
# ------------------------------------------------------------------

def prepare_model_for_training(
    model: EarlyExitLlamaForCausalLM, early_layer: int
) -> EarlyExitLlamaForCausalLM:
    """Freeze base, add dual-LoRA, make LoRA-S + exit head trainable."""
    model = inject_dual_lora(model, exit_layer=early_layer, rank=8)

    for p in model.parameters():
        p.requires_grad = False

    enable_lora_grads(model, "lora_S", True)   # fast path
    enable_lora_grads(model, "lora_D", False)  # slow path frozen

    # — detach exit head so only it (not lm_head) is tuned -------------
    w = model.lm_head.weight.detach().clone()
    hidden, vocab = w.shape[1], w.shape[0]
    model.exit_proj = nn.Linear(hidden, vocab, bias=False, device=w.device)
    model.exit_proj.weight.data.copy_(w)
    model.exit_proj.weight.requires_grad = True
    model.lm_head.weight.requires_grad   = False
    return model


def load_pretrained(model_id: str, early_layer: int, dev: torch.device):
    """Load HF checkpoint → Early-Exit model → make trainable subset."""
    kwargs = {"EARLY_STOP_LAYER": early_layer}
    multi_gpu = torch.cuda.device_count() > 1

    model = EarlyExitLlamaForCausalLM.from_pretrained(
        model_id,
        # device_map="auto" if multi_gpu else None,
        **kwargs,
    )

    # if not multi_gpu:                       # single-GPU / CPU case
    model.to(dev)

    return prepare_model_for_training(model, early_layer)


# ------------------------------------------------------------------
# Sampling & update helpers
# ------------------------------------------------------------------

def sample_prompt(
    tokenizer: Optional[AutoTokenizer],
    prompts:   Optional[List[str]],
    device:    torch.device,
    vocab:     int,
) -> torch.Tensor:
    if tokenizer and prompts:
        text = random.choice(prompts)
        return tokenizer(text, return_tensors="pt").input_ids.to(device)
    # dummy random prompt (len 4) for unit tests / smoke runs
    return torch.randint(0, vocab, (1, 4), device=device)


def reinforce_update(model, opt, batch, baseline: float, clip: float):
    """One REINFORCE step on LoRA-S + exit head."""
    dev       = next(model.parameters()).device
    hidden2d  = batch["hidden"].to(dev).reshape(batch["hidden"].shape[0], -1)
    tokens    = batch["token"].to(dev)
    rewards   = batch["reward"].to(dev)

    if tokens.dim() == 1:                  # (B,) → (B,1)
        tokens = tokens.unsqueeze(1)

    logits = model.exit_proj(hidden2d)     # (B, |V|)
    log_pi = torch.log_softmax(logits, -1).gather(1, tokens)[:, 0]
    loss   = -((rewards - baseline) * log_pi).mean()

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step(); opt.zero_grad()
    return float(loss.item()), float(grad_norm)


def update_baseline(baseline: float, rewards: torch.Tensor) -> float:
    """Exponential-moving-average baseline."""
    return 0.9 * baseline + 0.1 * float(rewards.mean().item())


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_loop(args):
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained(args.model_id, args.early_layer, dev)

    tokenizer, prompts = None, None
    if args.prompt_file:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    opt     = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    buffer  = ReplayBuffer(args.buffer_cap, torch.device("cpu"))   # buffer on CPU
    baseline = 0.0

    for gstep in range(args.max_steps):
        prompt = sample_prompt(tokenizer, prompts, dev, model.config.vocab_size)
        ids    = prompt[:, :1]                                     # start with <BOS?>

        # ---------- rollout ----------------------------------------
        for _ in range(args.rollout_len):
            step = model.spec_decode_step(ids[:, -1:])

            token_cuda = step.token.to(dev)        # 🟢 keep on same device
            buffer.append(
                step.hidden.squeeze(0),            # (H)  kept on CPU
                int(step.token.squeeze().item()),  # scalar
                float(step.accept.squeeze().item()),
                conf=0.0,
            )
            ids = torch.cat([ids, token_cuda], dim=-1)  # devices now match

        # ---------- policy update ----------------------------------
        if buffer.accepted_count() >= args.batch_size:
            batch = buffer.sample(args.batch_size, accepted_only=True)
            loss, gnorm = reinforce_update(model, opt, batch, baseline, args.clip)
            baseline = update_baseline(baseline, batch["reward"])
            print(
                f"step {gstep:5d} | loss {loss:7.4f} | "
                f"grad {gnorm:7.4f} | baseline {baseline:6.3f}"
            )
        else:                                           
            print(
                f"step {gstep:5d} | accepted "
                f"{buffer.accepted_count():3d} / {args.batch_size}"
            )



# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",     type=str,   required=True)
    p.add_argument("--early_layer",  type=int,   required=True)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--rollout_len",  type=int,   default=4)
    p.add_argument("--buffer_cap",   type=int,   default=512)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--clip",         type=float, default=1.0)
    p.add_argument("--max_steps",    type=int,   default=1000)
    p.add_argument("--prompt_file",  type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
