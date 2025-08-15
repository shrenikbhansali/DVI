#!/usr/bin/env python
"""DVI smoketest script.

Orchestrates a tiny run that demonstrates the draft path can be trained
with a mixture of KL and reinforcement learning.  The goal is not state
of the art quality but exercising the training pipeline end‑to‑end.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.buffer import ReplayBuffer
from training.kl_mix import cosine_decay_lambda, exp_decay_lambda
from train_dvi import prepare_model_for_training, mixed_update, update_baseline

# ``args`` is populated in ``main`` and accessed by helper functions (e.g.
# ``rollout``) via the global namespace.
args: Optional[argparse.Namespace] = None
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_outdir(path: str) -> Path:
    out = Path(path)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(outdir: Path, level: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "logs" / "smoketest.log")
    fh.setFormatter(fmt)
    logger.handlers = []
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_alpaca_sample(dest: Path, max_prompts: int) -> List[Dict[str, str]]:
    """Download a small Alpaca JSON file.

    The function is resilient: if download fails we return an empty list and
    the caller can fall back to random prompts.
    """
    import json as _json
    import requests

    if dest.exists():
        with dest.open("r", encoding="utf-8") as f:
            return _json.load(f)[:max_prompts]

    url = "https://raw.githubusercontent.com/tatsu-lab/alpaca/main/alpaca_data.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()[:max_prompts]
        dest.write_text(_json.dumps(data), encoding="utf-8")
        return data
    except Exception as exc:  # pragma: no cover - network failure branch
        logging.warning("Failed to download Alpaca sample: %s", exc)
        return []


def load_prompts(args, tokenizer: Optional[AutoTokenizer]) -> List[str]:
    if args.data_mode == "alpaca":
        path = Path(args.data_path) if args.data_path else Path(args.outdir) / "alpaca_sample.json"
        data = download_alpaca_sample(path, args.max_prompts)
        prompts = []
        for rec in data:
            instr = rec.get("instruction", "")
            inp = rec.get("input", "")
            prompts.append(f"Instruction: {instr}\nInput: {inp}\nAnswer:")
        return prompts
    elif args.data_mode == "lines" and args.data_path:
        with open(args.data_path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()][: args.max_prompts]
    return []  # random mode or missing path


@torch.no_grad()
def sample_prompt(tokenizer: Optional[AutoTokenizer], prompts: List[str], device: torch.device, vocab: int, trunc: int) -> torch.Tensor:
    if tokenizer and prompts:
        text = random.choice(prompts)[:trunc]
        return tokenizer(text, return_tensors="pt").input_ids.to(device)
    # fallback random prompt of length 4
    return torch.randint(0, vocab, (1, 4), device=device)


# ---------------------------------------------------------------------------
# Rollout utilities
# ---------------------------------------------------------------------------


def entropy_from_logits(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits, dim=-1)
    return float(-(logp.exp() * logp).sum(dim=-1).mean().item())


@dataclass
class RolloutStats:
    accept_flags: List[int]
    duration: float


def rollout(
    model: EarlyExitLlamaForCausalLM,
    ids: torch.Tensor,
    K: int,
    buffer: ReplayBuffer,
    debug: List[Dict],
    tokenizer: Optional[AutoTokenizer],
    phase: str,
    step: int,
    token_pos: List[int],
    mismatch_hist: Dict[int, int],
    check_vlogits_shape: List[bool],
) -> RolloutStats:
    """Roll out ``K`` speculative steps and append transitions to buffer."""
    device = ids.device
    accept_flags: List[int] = []
    start = time.time()
    for k in range(K):
        vlogits = model.verifier_logits_for_next(ids[:, -1:])
        step_out = model.spec_decode_step(ids[:, -1:])

        if not check_vlogits_shape:
            vocab = model.lm_head.weight.shape[0]
            assert vlogits.shape == (ids.size(0), vocab), "vlogits has unexpected shape"
            check_vlogits_shape.append(True)

        buffer.append(
            step_out.hidden.squeeze(0).cpu(),
            int(step_out.token.squeeze().item()),
            float(step_out.accept.squeeze().item()),
            float(step_out.conf.squeeze().item()),
            vlogits.squeeze(0).cpu(),
        )

        ids = torch.cat([ids, step_out.token.to(device)], dim=-1)
        accept = int(step_out.accept.item())
        accept_flags.append(accept)

        # first-mismatch histogram
        v_top = int(vlogits.argmax(dim=-1).item())
        if v_top != int(step_out.token.item()):
            pos = token_pos[0]
            mismatch_hist[pos] = mismatch_hist.get(pos, 0) + 1
        token_pos[0] += 1

        # debug samples --------------------------------------------------
        if len(debug) < args.debug_samples:
            draft_logits = model.exit_proj(step_out.hidden.to(device)).squeeze(1).float()
            draft_probs = torch.softmax(draft_logits, dim=-1)[0]
            v_probs = torch.softmax(vlogits.float(), dim=-1)[0]
            d_prob, d_id = torch.topk(draft_probs, 5)
            v_prob, v_id = torch.topk(v_probs, 5)
            tok_id = int(step_out.token.item())
            entry = {
                "phase": phase,
                "step": step,
                "prompt_text": tokenizer.decode(ids[0].tolist()) if tokenizer else "",
                "draft_token_id": tok_id,
                "draft_token_str": tokenizer.decode([tok_id]) if tokenizer else str(tok_id),
                "verifier_argmax_id": int(v_top),
                "verifier_argmax_str": tokenizer.decode([v_top]) if tokenizer else str(v_top),
                "accept": accept,
                "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                "conf_verifier_selected": float(v_probs[tok_id]),
                "position": token_pos[0] - 1,
            }
            debug.append(entry)
    dur = time.time() - start
    return RolloutStats(accept_flags=accept_flags, duration=dur)


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------


def run_phase(
    phase: str,
    steps: int,
    args,
    model: EarlyExitLlamaForCausalLM,
    tokenizer: Optional[AutoTokenizer],
    prompts: List[str],
    buffer: ReplayBuffer,
    baseline: float,
    global_step: int,
    stats: Dict[str, List[float]],
    history: Dict[str, List[float]],
    writer: Optional[SummaryWriter],
    csv_writer: csv.DictWriter,
    mismatch_hist: Dict[int, int],
    token_pos: List[int],
    check_vlogits_shape: List[bool],
    tracker: 'MetricTracker',
    opt: torch.optim.Optimizer,
    baseline_ref: float,
) -> (float, int):
    dev = next(model.parameters()).device
    for s in range(1, steps + 1):
        global_step += 1
        prompt = sample_prompt(tokenizer, prompts, dev, model.config.vocab_size, args.prompt_trunc)
        rollout_stats = rollout(
            model,
            prompt[:, :1],
            args.rollout_len,
            buffer,
            history.setdefault("debug", []),
            tokenizer,
            phase,
            global_step,
            token_pos,
            mismatch_hist,
            check_vlogits_shape,
        )
        stats.setdefault(phase, []).append(sum(rollout_stats.accept_flags) / len(rollout_stats.accept_flags))

        # determine if we have enough samples for update
        if phase == "baseline":
            loss = rl_loss = kl_loss = gnorm = 0.0
            kl_lambda = 0.0
        else:
            accepted_only = False
            use_rl = phase != "warmup" or not args.no_rl_during_warmup
            if use_rl and not args.rl_all_tokens:
                accepted_only = True
            count = buffer.accepted_count() if accepted_only else len(buffer)
            if count >= args.batch_size:
                batch = buffer.sample(args.batch_size, accepted_only=accepted_only)
                if phase == "warmup":
                    kl_lambda = 1.0
                    use_rl = False
                elif phase == "mixed":
                    if args.kl_schedule == "exp":
                        kl_lambda = exp_decay_lambda(global_step, args.kl_lambda0, args.kl_tau, args.kl_min)
                    else:
                        total = args.baseline_steps + args.warmup_steps + args.mixed_steps + args.pure_rl_steps
                        kl_lambda = cosine_decay_lambda(global_step, total, args.kl_lambda0, args.kl_min)
                    use_rl = True
                else:  # pure_rl
                    kl_lambda = 0.0
                    use_rl = True
                loss, gnorm, rl_loss, kl_loss = mixed_update(
                    model,
                    opt,
                    batch,
                    baseline=baseline,
                    clip=args.clip,
                    kl_lambda=kl_lambda,
                    kl_dir=args.kl_dir,
                    kl_temperature=args.kl_temperature,
                    use_rl=use_rl,
                    rl_all_tokens=args.rl_all_tokens,
                )
                if use_rl and args.adv_baseline_mode == "ema":
                    baseline = update_baseline(baseline, batch["reward"])
                draft_ent = entropy_from_logits(model.exit_proj(batch["hidden"].to(dev).squeeze(1)).float())
                verifier_ent = entropy_from_logits(batch["vlogits"].to(dev).float())
            else:
                loss = rl_loss = kl_loss = gnorm = 0.0
                kl_lambda = 0.0
                draft_ent = verifier_ent = 0.0

        # metric aggregation ---------------------------------------------
        tracker.update_window(rollout_stats.accept_flags, rollout_stats.duration)
        history.setdefault("acc", []).append(tracker.acceptance_rate())
        history.setdefault("ctar1", []).append(tracker.ctar_w(1))
        history.setdefault("ctar2", []).append(tracker.ctar_w(2))
        history.setdefault("comp", []).append(tracker.compression())
        history.setdefault("tok_s", []).append(tracker.throughput())
        history.setdefault("loss", []).append(loss)
        history.setdefault("rl", []).append(rl_loss)
        history.setdefault("kl", []).append(kl_loss)
        history.setdefault("lambda", []).append(kl_lambda)
        history.setdefault("grad", []).append(gnorm)
        history.setdefault("baseline", []).append(baseline)
        history.setdefault("ent_d", []).append(draft_ent)
        history.setdefault("ent_v", []).append(verifier_ent)

        row = {
            "phase": phase,
            "step": global_step,
            "acc": tracker.acceptance_rate(),
            "ctar1": tracker.ctar_w(1),
            "ctar2": tracker.ctar_w(2),
            "comp": tracker.compression(),
            "tok_s": tracker.throughput(),
            "loss": loss,
            "rl": rl_loss,
            "kl": kl_loss,
            "lambda": kl_lambda,
            "grad": gnorm,
            "baseline": baseline,
            "ent_d": draft_ent,
            "ent_v": verifier_ent,
        }
        csv_writer.writerow(row)
        if writer:
            for k, v in row.items():
                if k in {"phase", "step"}:
                    continue
                writer.add_scalar(f"{phase}/{k}", v, global_step)

        if global_step % args.eval_interval == 0:
            delta = tracker.acceptance_rate() - baseline_ref
            logging.info(
                f"[PHASE {phase}][step {global_step}] acc={tracker.acceptance_rate():.2f} (Δ{delta:+.2f}) "
                f"CTAR1={tracker.ctar_w(1):.2f} CTAR2={tracker.ctar_w(2):.2f} comp={tracker.compression():.2f} "
                f"tok/s={tracker.throughput():.1f} RL={rl_loss:.2f} KL={kl_loss:.2f} total={loss:.2f} "
                f"λ={kl_lambda:.2f} grad={gnorm:.2f} base={baseline:.2f}"
            )
    return baseline, global_step


# ---------------------------------------------------------------------------
# Metric tracker class
# ---------------------------------------------------------------------------


class MetricTracker:
    def __init__(self, K: int):
        self.K = K
        self.reset()

    def reset(self):
        self.tokens = 0
        self.accepted = 0
        self.windows = 0
        self.ctar_counts = [0 for _ in range(self.K)]
        self.throughput_tokens = 0
        self.throughput_time = 0.0

    def update_window(self, accepts: Iterable[int], duration: float) -> None:
        accepts = list(accepts)
        self.windows += 1
        self.tokens += len(accepts)
        self.accepted += sum(accepts)
        for w in range(1, self.K + 1):
            if len(accepts) >= w and all(accepts[:w]):
                self.ctar_counts[w - 1] += 1
        self.throughput_tokens += len(accepts)
        self.throughput_time += duration

    def acceptance_rate(self) -> float:
        return self.accepted / self.tokens if self.tokens else 0.0

    def ctar_w(self, w: int) -> float:
        return self.ctar_counts[w - 1] / self.windows if self.windows else 0.0

    def compression(self) -> float:
        return self.accepted / self.windows if self.windows else 0.0

    def throughput(self) -> float:
        return self.throughput_tokens / self.throughput_time if self.throughput_time else 0.0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("DVI smoketest")
    # model / device ------------------------------------------------------
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--early_layer", type=int, default=2)
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="float16")
    p.add_argument("--device", type=str, default="auto")

    # data ---------------------------------------------------------------
    p.add_argument("--data_mode", choices=["alpaca", "lines", "random"], default="alpaca")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--max_prompts", type=int, default=2000)
    p.add_argument("--prompt_trunc", type=int, default=128)

    # rollout / training lengths ---------------------------------------
    p.add_argument("--K", "--rollout_len", type=int, default=4)
    p.add_argument("--baseline_steps", type=int, default=150)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--mixed_steps", type=int, default=400)
    p.add_argument("--pure_rl_steps", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=25)

    # optim / buffer ----------------------------------------------------
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--buffer_cap", type=int, default=1024)
    p.add_argument("--clip", type=float, default=1.0)

    # SAL / schedule ----------------------------------------------------
    p.add_argument("--kl_schedule", choices=["exp", "cosine"], default="exp")
    p.add_argument("--kl_lambda0", type=float, default=1.0)
    p.add_argument("--kl_min", type=float, default=0.0)
    p.add_argument("--kl_tau", type=int, default=200)
    p.add_argument("--kl_dir", choices=["v2d", "d2v"], default="v2d")
    p.add_argument("--kl_temperature", type=float, default=1.5)

    # RL specifics ------------------------------------------------------
    p.add_argument("--rl_all_tokens", action="store_true", default=True)
    p.add_argument("--no-rl_all_tokens", dest="rl_all_tokens", action="store_false")
    p.add_argument("--no_rl_during_warmup", action="store_true", default=True)
    p.add_argument("--adv_baseline_mode", choices=["ema", "none"], default="ema")

    # output / reproducibility -----------------------------------------
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--log_level", choices=["INFO", "DEBUG"], default="INFO")
    p.add_argument("--debug_samples", type=int, default=32)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global args
    args = parse_args()
    outdir = setup_outdir(args.outdir)
    logger = setup_logging(outdir, args.log_level)
    set_seed(args.seed)

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dev = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = DTYPE_MAP[args.dtype]

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    except Exception:  # pragma: no cover - tokenizer failure
        tokenizer = None
        logging.warning("Falling back to random prompts; tokenizer load failed")

    model = EarlyExitLlamaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, EARLY_STOP_LAYER=args.early_layer)
    model.to(dev)
    model = prepare_model_for_training(model, args.early_layer)

    # KV side-effect-free check ----------------------------------------
    ids = torch.randint(0, model.config.vocab_size, (1, 1), device=dev)
    v1 = model.verifier_logits_for_next(ids)
    v2 = model.verifier_logits_for_next(ids)
    assert torch.allclose(v1, v2), "verifier accessor has side effects on outputs"

    prompts = load_prompts(args, tokenizer)
    buffer = ReplayBuffer(args.buffer_cap, torch.device("cpu"))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    baseline = 0.0

    tracker = MetricTracker(args.K)
    writer = SummaryWriter(outdir) if args.tensorboard else None

    csv_file = open(outdir / "metrics.csv", "w", newline="")
    fieldnames = ["phase", "step", "acc", "ctar1", "ctar2", "comp", "tok_s", "loss", "rl", "kl", "lambda", "grad", "baseline", "ent_d", "ent_v"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    stats: Dict[str, List[float]] = {}
    history: Dict[str, List[float]] = {}
    mismatch_hist: Dict[int, int] = {}
    token_pos = [0]
    check_vlogits_shape: List[bool] = []

    # phases ------------------------------------------------------------
    global_step = 0
    baseline, global_step = run_phase("baseline", args.baseline_steps, args, model, tokenizer, prompts, buffer, baseline, global_step, stats, history, writer, csv_writer, mismatch_hist, token_pos, check_vlogits_shape, tracker, opt, 0.0)
    baseline_ref = sum(stats["baseline"]) / len(stats["baseline"]) if stats.get("baseline") else 0.0
    baseline_ctar1 = tracker.ctar_w(1)
    baseline_comp = tracker.compression()
    baseline, global_step = run_phase("warmup", args.warmup_steps, args, model, tokenizer, prompts, buffer, baseline, global_step, stats, history, writer, csv_writer, mismatch_hist, token_pos, check_vlogits_shape, tracker, opt, baseline_ref)

    warmup_mean = sum(stats.get("warmup", [0.0])) / max(len(stats.get("warmup", [])), 1)
    if warmup_mean < baseline_ref + 0.03:
        raise SystemExit("Acceptance failed to improve during warmup; aborting")

    baseline, global_step = run_phase("mixed", args.mixed_steps, args, model, tokenizer, prompts, buffer, baseline, global_step, stats, history, writer, csv_writer, mismatch_hist, token_pos, check_vlogits_shape, tracker, opt, baseline_ref)
    baseline, global_step = run_phase("pure_rl", args.pure_rl_steps, args, model, tokenizer, prompts, buffer, baseline, global_step, stats, history, writer, csv_writer, mismatch_hist, token_pos, check_vlogits_shape, tracker, opt, baseline_ref)

    csv_file.close()
    if writer:
        writer.close()

    # summary -----------------------------------------------------------
    end_acc = sum(history.get("acc", [])[-10:]) / max(len(history.get("acc", [])[-10:]), 1)
    ctar1_final = history.get("ctar1", [0.0])[-1]
    comp_final = history.get("comp", [0.0])[-1]
    peak_tok_s = max(history.get("tok_s", [0.0]))
    summary = {
        "baseline_acc_mean": baseline_ref,
        "baseline_acc_std": float(torch.tensor(stats.get("baseline", [0.0])).std().item()) if stats.get("baseline") else 0.0,
        "end_acc": end_acc,
        "delta_acc": end_acc - baseline_ref,
        "ctar1_final": ctar1_final,
        "ctar1_delta": ctar1_final - baseline_ctar1,
        "compression_final": comp_final,
        "compression_delta": comp_final - baseline_comp,
        "peak_tok_s": peak_tok_s,
        "final_lambda": history.get("lambda", [0.0])[-1] if history.get("lambda") else 0.0,
        "final_baseline": baseline,
        "verdict": "PASS" if end_acc - baseline_ref >= 0.03 else "WARN",
    }
    with open(outdir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(outdir / "debug_samples.jsonl", "w", encoding="utf-8") as f:
        for rec in history.get("debug", []):
            f.write(json.dumps(rec) + "\n")

    # plots -------------------------------------------------------------
    try:  # pragma: no cover - plotting is best effort
        import matplotlib.pyplot as plt

        xs = list(range(1, len(history.get("acc", [])) + 1))
        plt.figure(); plt.plot(xs, history.get("acc", [])); plt.xlabel("step"); plt.ylabel("acceptance"); plt.savefig(outdir / "plots" / "acceptance.png"); plt.close()
        plt.figure(); plt.plot(xs, history.get("ctar1", []), label="CTAR1"); plt.plot(xs, history.get("ctar2", []), label="CTAR2"); plt.legend(); plt.savefig(outdir / "plots" / "ctar.png"); plt.close()
        plt.figure(); plt.plot(xs, history.get("loss", []), label="loss"); plt.plot(xs, history.get("rl", []), label="rl"); plt.plot(xs, history.get("kl", []), label="kl"); plt.legend(); plt.savefig(outdir / "plots" / "losses.png"); plt.close()
        plt.figure(); plt.plot(xs, history.get("tok_s", [])); plt.xlabel("step"); plt.ylabel("tok/s"); plt.savefig(outdir / "plots" / "throughput.png"); plt.close()
    except Exception as exc:
        logging.warning("Plot generation failed: %s", exc)

    if mismatch_hist:
        logging.warning("first-mismatch positions: %s", mismatch_hist)


if __name__ == "__main__":  # pragma: no cover
    main()
