"""Utility helpers for DVI training pipeline.

This module houses functions that were previously defined inside the
standalone smoketest script. They are now reusable across training,
evaluation and logging modules.
"""
import os
import json
import gc
import random
import time
from typing import List, Optional, Dict
from statistics import median

import torch
from datasets import load_dataset
from transformers import __version__ as transformers_ver
import time
from statistics import median


__all__ = [
    "set_seed",
    "ensure_dirs",
    "env_dump",
    "build_prompts_from_alpaca",
    "ctar",
    "free_cuda",
    "_cuda_sync",
    # added helpers for compression + timing
    "count_transformer_layers",
    "theoretical_compression",
    "measure_generate_walltime",
]


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)


def env_dump(outdir: Optional[str] = None) -> None:
    info = dict(
        torch_version=torch.__version__,
        cuda=torch.version.cuda if torch.cuda.is_available() else None,
        transformers=transformers_ver,
        gpus=torch.cuda.device_count(),
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None),
    )
    msg = "[env] " + json.dumps(info, indent=2)
    print(msg, flush=True)
    if outdir:
        with open(os.path.join(outdir, "logs", "env.json"), "w") as f:
            json.dump(info, f, indent=2)


def build_prompts_from_alpaca(limit: int) -> List[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts: List[str] = []
    for ex in ds:
        inst = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        if inp:
            text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            text = f"### Instruction:\n{inst}\n\n### Response:\n"
        if len(text) > 20:
            prompts.append(text)
        if len(prompts) >= limit:
            break
    try:
        ds.cleanup_cache_files()
    except Exception:
        pass
    return prompts


def ctar(bits: List[int], w: int) -> float:
    if len(bits) < w:
        return 0.0
    tot = len(bits) - w + 1
    ok = 0
    for i in range(tot):
        if all(bits[i + j] == 1 for j in range(w)):
            ok += 1
    return ok / max(1, tot)


def free_cuda(note: str = "") -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    if note:
        print(f"[mem] cleared caches {note}", flush=True)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------
# Added: compression + timing
# -----------------------------

def count_transformer_layers(model) -> int:
    """
    Best-effort count of decoder block layers across common HF model structures.
    Returns >=1 to avoid zero-division downstream.
    """
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        cur = model
        ok = True
        for p in path.split("."):
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__len__"):
            try:
                return len(cur)
            except Exception:
                pass
    return 1


def theoretical_compression(accept_rate: float, early_layer: int, total_layers: int):
    """
    Expected compute ratio vs full baseline under early-exit:
      Let f = early_layer / total_layers in [0,1].
      cost_ratio = f + (1 - accept_rate) * (1 - f)
      compression_ratio = cost_ratio (baseline normalized to 1)
      speedup_est = 1 / compression_ratio
    """
    total_layers = max(int(total_layers), 1)
    f = max(0.0, min(1.0, float(early_layer) / float(total_layers)))
    comp_ratio = f + (1.0 - float(accept_rate)) * (1.0 - f)
    comp_ratio = max(1e-6, float(comp_ratio))
    return comp_ratio, 1.0 / comp_ratio

def _safe_max_ctx_len(tok, model, default: int = 2048) -> int:
    """
    Return a sane maximum context length for tokenization/generation.
    Prefers model.config.max_position_embeddings; falls back to tokenizer
    if it looks reasonable, otherwise to `default`.
    """
    cfg = getattr(model, "config", None)

    # Primary: model config (most reliable)
    for key in ("max_position_embeddings", "n_positions", "max_seq_len", "max_sequence_length", "seq_length"):
        v = getattr(cfg, key, None) if cfg is not None else None
        if isinstance(v, int) and v > 0:
            # Clamp to avoid absurd values
            return int(min(v, 131072))

    # Secondary: tokenizer’s model_max_length, but only if it’s not a sentinel
    try:
        tmax = int(getattr(tok, "model_max_length", 0))
        if 0 < tmax < 1_000_000:
            return int(tmax)
    except Exception:
        pass

    # Fallback
    return int(default)


# put this near the other helpers in training/utils.py
def _get_input_vocab_size(model) -> Optional[int]:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return int(emb.weight.size(0))
    except Exception:
        pass
    # best-effort fallbacks for common HF layouts
    for path in ("model.embed_tokens", "model.model.embed_tokens", "transformer.wte"):
        cur = model
        ok = True
        for p in path.split("."):
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok and hasattr(cur, "weight"):
            try:
                return int(cur.weight.size(0))
            except Exception:
                pass
    return None

def measure_generate_walltime(
    model,
    tok,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    greedy: bool = False,
    repeats: int = 3,
    use_dvi_spec: bool = False,
    draft_k: int = 4,
    temperature: float = 1.0,
    quiet: bool = True,
    early_layer_override: Optional[int] = None,
):
    """
    Returns:
      - elapsed_seconds if use_dvi_spec=False
      - (elapsed_seconds, spec_metrics_dict) if use_dvi_spec=True
    """
    times: List[float] = []
    spec_last: Optional[Dict[str, float]] = None
    for _ in range(repeats):
        _cuda_sync()
        t0 = time.perf_counter()
        if use_dvi_spec:
            from training.spec_decode import generate_with_dvi_spec

            _, spec_metrics = generate_with_dvi_spec(
                model, tok, prompts,
                max_new_tokens=max_new_tokens,
                draft_k=draft_k,
                greedy=greedy,
                temperature=temperature,
                early_layer=early_layer_override or getattr(model, "early_layer", None),
                device=next(model.parameters()).device,
                quiet=quiet,
            )
            spec_last = spec_metrics.to_dict()
        else:
            with torch.no_grad():
                _ = model.generate(
                    **tok(prompts, return_tensors="pt", padding=True).to(next(model.parameters()).device),
                    max_new_tokens=max_new_tokens,
                    do_sample=not greedy,
                    temperature=max(1e-6, temperature),
                    use_cache=True,
                )
        _cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    elapsed = float(sorted(times)[len(times)//2])
    return (elapsed, spec_last) if use_dvi_spec else elapsed
