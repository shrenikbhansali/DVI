"""Acceptance evaluation and CTAR metrics."""
import json
from typing import Dict, List, Optional, Tuple

import torch

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.kv import (
    estimate_kv_cache,
    drafter_hidden_no_cache,
    prime_kv_full,
    advance_kv_with_committed,
    clear_all_kv,
)
from training.utils import ctar, free_cuda

__all__ = ["eval_acceptance"]

_KV_WARN_COUNT = 0
_KV_WARN_LIMIT = 8


def _top1(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1, keepdim=True)[0, 0].item())


@torch.inference_mode()
def eval_acceptance(spec: EarlyExitLlamaForCausalLM, tok, prompts: List[str],
                   rollout_len: int, steps_per_prompt: int = 1,
                   dump_debug: bool = False, dump_path: Optional[str] = None, topk: int = 5,
                   quiet: bool = False) -> Tuple[float, Dict[int, float]]:
    global _KV_WARN_COUNT
    spec.eval()
    dev = next(spec.parameters()).device
    accepts: List[int] = []
    dbg_out: List[Dict] = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(dev)
        prime_kv_full(spec, enc["input_ids"])
        ids_last = enc["input_ids"][:, -1:]
        for _ in range(steps_per_prompt * rollout_len):
            v_logits = spec.verifier_logits_for_next(ids_last)
            if not torch.isfinite(v_logits).all():
                accepts.append(0)
                break
            v_top1 = _top1(v_logits)
            kv_bytes_b, kv_seq_b = estimate_kv_cache(spec)
            d_hidden = drafter_hidden_no_cache(spec, ids_last)
            kv_bytes_a, kv_seq_a = estimate_kv_cache(spec)
            if (kv_bytes_b != kv_bytes_a or kv_seq_b != kv_seq_a) and _KV_WARN_COUNT < _KV_WARN_LIMIT:
                print("[kv] WARNING: probe seems to have mutated past_key_values during eval.", flush=True)
                _KV_WARN_COUNT += 1
            h = d_hidden.squeeze(1).float()
            if hasattr(spec, "exit_pre_norm") and spec.exit_pre_norm is not None:
                h = spec.exit_pre_norm(h)
            d_logits = spec.exit_proj(h)
            if hasattr(spec, "exit_logit_scale"):
                d_logits = spec.exit_logit_scale * d_logits
            if not torch.isfinite(d_logits).all():
                accepts.append(0)
                d_top1 = -1
            else:
                d_top1 = _top1(d_logits)
                accepts.append(int(d_top1 == v_top1))
            if dump_debug and dump_path:
                try:
                    v_prob, v_id = torch.topk(torch.softmax(v_logits.float(), dim=-1)[0], k=topk)
                    d_prob, d_id = torch.topk(torch.softmax(d_logits.float(), dim=-1)[0], k=topk)
                    dbg_out.append({
                        "draft_top1": int(d_top1),
                        "verifier_top1": int(v_top1),
                        "accept": int(d_top1 == v_top1),
                        "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                        "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                    })
                except Exception:
                    pass
            v_token = torch.tensor([[v_top1]], device=ids_last.device, dtype=ids_last.dtype)
            advance_kv_with_committed(spec, v_token)
            ids_last = v_token
        clear_all_kv(spec)
        if not quiet:
            free_cuda("(eval prompt done)")
        else:
            free_cuda("")
    acc = sum(accepts) / max(1, len(accepts))
    c = {w: ctar(accepts, w) for w in (1, 2, 3, 4)}
    if dump_debug and dump_path and dbg_out:
        with open(dump_path, "a") as f:
            for rec in dbg_out:
                f.write(json.dumps(rec) + "\n")
    return acc, c
