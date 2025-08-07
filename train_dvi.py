import argparse
import random
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import inject_dual_lora, enable_lora_grads
from training.buffer import ReplayBuffer

# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------

def prepare_model_for_training(model: EarlyExitLlamaForCausalLM, early_layer: int) -> EarlyExitLlamaForCausalLM:
    """Freeze base model and enable grads for LoRA-S & exit head."""
    model = inject_dual_lora(model, exit_layer=early_layer, rank=8)
    for p in model.parameters():
        p.requires_grad = False
    enable_lora_grads(model, "lora_S", True)
    enable_lora_grads(model, "lora_D", False)
    # Detach exit head from lm_head to keep only exit_proj trainable
    weight = model.lm_head.weight.detach().clone()
    hidden, vocab = weight.shape[1], weight.shape[0]
    model.exit_proj = nn.Linear(hidden, vocab, bias=False, device=weight.device)
    model.exit_proj.weight.data.copy_(weight)
    model.exit_proj.weight.requires_grad = True
    model.lm_head.weight.requires_grad = False
    return model


def load_pretrained(model_id: str, early_layer: int, device: torch.device) -> EarlyExitLlamaForCausalLM:
    """Load HF weights and prepare for training."""
    kwargs = {"EARLY_STOP_LAYER": early_layer}
    use_auto = torch.cuda.device_count() > 1
    try:
        model = EarlyExitLlamaForCausalLM.from_pretrained(
            model_id, device_map="auto" if use_auto else None, **kwargs
        )
    except TypeError:
        use_auto = False
        model = EarlyExitLlamaForCausalLM.from_pretrained(model_id, **kwargs)
    if not use_auto:
        model.to(device)
    model = prepare_model_for_training(model, early_layer)
    return model

# -----------------------------------------------------------------------------
# Sampling & updates
# -----------------------------------------------------------------------------

def sample_prompt(tokenizer: Optional[AutoTokenizer], prompts: Optional[List[str]], device: torch.device, vocab: int) -> torch.Tensor:
    if tokenizer and prompts:
        text = random.choice(prompts)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    else:
        ids = torch.randint(0, vocab, (1, 4), device=device)
    return ids


def reinforce_update(model, opt, batch, baseline: float, clip: float):
    hidden2d = batch["hidden"].view(batch["hidden"].size(0), -1)
    logits = model.exit_proj(hidden2d)
    tokens = batch["token"]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(1)
    log_pi = torch.log_softmax(logits, -1).gather(1, tokens)[:, 0]
    adv = batch["reward"] - baseline
    loss = -(adv * log_pi).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    opt.zero_grad()
    return loss.item(), float(grad_norm)


def update_baseline(baseline: float, rewards: torch.Tensor) -> float:
    return 0.9 * baseline + 0.1 * rewards.mean().item()

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained(args.model_id, args.early_layer, device)
    tokenizer = None
    prompts = None
    if args.prompt_file is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    buffer = ReplayBuffer(args.buffer_cap, device)
    baseline = 0.0

    for global_step in range(args.max_steps):
        prompt = sample_prompt(tokenizer, prompts, device, model.config.vocab_size)
        ids = prompt[:, :1]
        for _ in range(args.rollout_len):
            step = model.spec_decode_step(ids[:, -1:])
            buffer.append(step.hidden.clone(), int(step.token), float(step.accept), conf=0.0)
            ids = torch.cat([ids, step.token], dim=-1)
        if buffer.accepted_count() >= args.batch_size:
            batch = buffer.sample(args.batch_size, accepted_only=True)
            loss, grad_norm = reinforce_update(model, opt, batch, baseline, args.clip)
            baseline = update_baseline(baseline, batch["reward"])
            print(f"step {global_step}: loss={loss:.3f} grad={grad_norm:.3f} baseline={baseline:.3f}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--early_layer", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rollout_len", type=int, default=4)
    p.add_argument("--buffer_cap", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--prompt_file", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    train_loop(args)

if __name__ == "__main__":
    main()
