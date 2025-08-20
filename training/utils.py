"""Utility helpers for DVI training pipeline.

This module houses functions that were previously defined inside the
standalone smoketest script. They are now reusable across training,
evaluation and logging modules.
"""
import os
import json
import gc
import random
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import __version__ as transformers_ver


__all__ = [
    "set_seed",
    "ensure_dirs",
    "env_dump",
    "build_prompts_from_alpaca",
    "ctar",
    "free_cuda",
    "_cuda_sync",
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
