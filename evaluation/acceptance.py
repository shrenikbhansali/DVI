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
from training.modeling import exit_logits_from_hidden_k, adapter_guard

__all__ = ["eval_acceptance"]

_KV_WARN_COUNT = 0
_KV_WARN_LIMIT = 8


def _top1(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1, keepdim=True)[0, 0].item())


# ``torch.inference_mode`` creates tensors that carry an "inference" flag and
# cannot participate in autograd.  Evaluation only needs gradients disabled, so
# we use ``torch.no_grad`` here to avoid producing inference tensors that could
# leak into later training phases.
@torch.no_grad()
def eval_acceptance(spec: EarlyExitLlamaForCausalLM, tok, prompts: List[str],
                   rollout_len: int, steps_per_prompt: int = 1,
                   dump_debug: bool = False, dump_path: Optional[str] = None, topk: int = 5,
                   quiet: bool = False, k_max: int = 4) -> Tuple[float, Dict[int, float]]:
    """Compute teacher-forced cross-token acceptance rates (CTAR).

    Draft and verifier logits are compared greedily (argmax) under teacher
    forcing—the verifier's top-1 token is always fed as the next input.
    Because runtime speculative decoding requires the entire drafted prefix to
    survive, these CTARs can overestimate block-level acceptance for k>1.
    """
    global _KV_WARN_COUNT
    spec.eval()
    dev = next(spec.parameters()).device
    accepts_seq: List[List[int]] = []
    dbg_out: List[Dict] = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(dev)
        prime_kv_full(spec, enc["input_ids"])
        ids_last = enc["input_ids"][:, -1:]
        cur_bits: List[int] = []
        for _ in range(steps_per_prompt * rollout_len):
            with adapter_guard(spec, "verify"):
                v_logits = spec.verifier_logits_for_next(ids_last)
            if not torch.isfinite(v_logits).all():
                cur_bits.append(0)
                break
            v_top1 = _top1(v_logits)
            kv_bytes_b, kv_seq_b = estimate_kv_cache(spec)
            with adapter_guard(spec, "draft"):
                d_hidden = drafter_hidden_no_cache(spec, ids_last)
            kv_bytes_a, kv_seq_a = estimate_kv_cache(spec)
            if (kv_bytes_b != kv_bytes_a or kv_seq_b != kv_seq_a) and _KV_WARN_COUNT < _KV_WARN_LIMIT:
                print("[kv] WARNING: probe seems to have mutated past_key_values during eval.", flush=True)
                _KV_WARN_COUNT += 1
            with adapter_guard(spec, "draft"):
                d_logits = exit_logits_from_hidden_k(spec, d_hidden)
            if not torch.isfinite(d_logits).all():
                cur_bits.append(0)
                d_top1 = -1
            else:
                d_top1 = _top1(d_logits)
                cur_bits.append(int(d_top1 == v_top1))
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
        accepts_seq.append(cur_bits)
        clear_all_kv(spec)
        if not quiet:
            free_cuda("(eval prompt done)")
        else:
            free_cuda("")
    flat = sum((seq for seq in accepts_seq), [])
    acc = sum(flat) / max(1, len(flat))
    c = {w: ctar(accepts_seq, w) for w in range(1, k_max + 1)}
    if dump_debug and dump_path and dbg_out:
        with open(dump_path, "a") as f:
            for rec in dbg_out:
                f.write(json.dumps(rec) + "\n")
    return acc, c
