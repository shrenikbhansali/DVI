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
from typing import List, Optional
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

def measure_generate_walltime(model, tok, prompts: List[str],
                              max_new_tokens: int = 64,
                              greedy: bool = True,
                              repeats: int = 3) -> float:
    """
    Return median walltime (seconds) for a .generate() pass over `prompts`.
    Robust to vocab/tokenizer mismatches and CUDA asserts during sampling.
    """
    device = next(model.parameters()).device
    model.eval()

    # Ensure decode IDs exist on model.config
    try:
        if getattr(model.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
            model.config.eos_token_id = tok.eos_token_id
        if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id
        if getattr(model.config, "bos_token_id", None) is None and getattr(tok, "bos_token_id", None) is not None:
            model.config.bos_token_id = tok.bos_token_id
        if getattr(model.config, "use_cache", None) is False:
            model.config.use_cache = True
    except Exception:
        pass

    # Choose a safe context length and clamp tokenizer max_length to avoid OverflowError
    max_ctx = _safe_max_ctx_len(tok, model, default=2048)
    max_new_tokens = int(max(1, max_new_tokens))
    # Leave some headroom for generation
    max_input_len = max(8, max_ctx - max_new_tokens - 8)

    # Tokenize with safe truncation/padding
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_input_len),
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Validate tokenizer/model vocab alignment (avoid device-side asserts)
    in_vocab = _get_input_vocab_size(model)
    if in_vocab is not None:
        max_id = int(input_ids.max().item())
        if max_id >= in_vocab:
            raise RuntimeError(
                f"Tokenizer produced id {max_id} >= input embedding vocab {in_vocab}. "
                "Ensure tokenizer matches the model and pad_token is set (e.g., tok.pad_token = tok.eos_token)."
            )

    def _do_generate(force_greedy: bool) -> float:
        _cuda_sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=not force_greedy,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                bos_token_id=getattr(tok, "bos_token_id", None),
            )
        _cuda_sync()
        return time.perf_counter() - t0

    times: List[float] = []
    for _ in range(max(1, repeats)):
        try:
            times.append(_do_generate(force_greedy=greedy))
        except RuntimeError as e:
            msg = str(e)
            if "device-side assert" in msg or "CUDA error" in msg:
                print("[time][warn] CUDA assert during sampling; retrying with greedy decode for timing.", flush=True)
                times.append(_do_generate(force_greedy=True))
            else:
                raise
    return float(median(times))
