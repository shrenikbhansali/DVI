"""Rollout utilities and buffer debug helpers."""
from typing import List, Dict, Optional

import json
import torch

from training.kv import drafter_hidden_no_cache, advance_kv_with_committed, prime_kv_full
from training.buffer import ReplayBuffer

__all__ = ["rollout_collect", "buf_debug"]


def _top1(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1, keepdim=True)[0, 0].item())


@torch.inference_mode()
def rollout_collect(spec, tok, prompt: str,
                    buf: ReplayBuffer, steps: int,
                    debug_out: Optional[List[Dict]] = None, topk: int = 5) -> int:
    spec.eval()
    dev = next(spec.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(dev)
    prime_kv_full(spec, enc["input_ids"])
    last = enc["input_ids"][:, -1:]
    n_collected = 0
    for _ in range(steps):
        v_logits = spec.verifier_logits_for_next(last)
        if not torch.isfinite(v_logits).all():
            break
        v_top1 = _top1(v_logits)
        d_hidden = drafter_hidden_no_cache(spec, last)
        h = d_hidden.squeeze(1).float()
        if hasattr(spec, "exit_pre_norm") and spec.exit_pre_norm is not None:
            h = spec.exit_pre_norm(h)
        d_logits = spec.exit_proj(h)
        if hasattr(spec, "exit_logit_scale"):
            d_logits = spec.exit_logit_scale * d_logits
        d_top1 = _top1(d_logits) if torch.isfinite(d_logits).all() else -1
        accept_bit = int(d_top1 == v_top1)
        buf.append(
            hidden=d_hidden.squeeze(0).squeeze(0).cpu(),
            token=int(v_top1),
            reward=float(accept_bit),
            conf=0.0,
            vlogits=v_logits.squeeze(0).cpu(),
        )
        if debug_out is not None:
            try:
                v_prob, v_id = torch.topk(torch.softmax(v_logits.float(), dim=-1)[0], k=topk)
                d_prob, d_id = torch.topk(torch.softmax(d_logits.float(), dim=-1)[0], k=topk)
                debug_out.append({
                    "draft_top1": int(d_top1),
                    "verifier_top1": int(v_top1),
                    "accept": accept_bit,
                    "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                    "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                })
            except Exception:
                pass
        v_token = torch.tensor([[v_top1]], device=last.device, dtype=last.dtype)
        advance_kv_with_committed(spec, v_token)
        last = v_token
        n_collected += 1
    return n_collected


def buf_debug(buf: ReplayBuffer, k: int = 16) -> None:
    size = len(buf)
    acc = buf.accepted_count()
    stats = {"total": size, "accepted": acc, "accept_rate_est": (acc / size) if size else 0.0}
    print("[buf]", json.dumps(stats, indent=2))
    if size == 0:
        return
    show = min(k, size)
    try:
        samp = buf.sample(show, accepted_only=False)
        for i in range(show):
            print(f"  sample[{i:02d}] tok={int(samp['token'][i])} r={float(samp['reward'][i])}")
    except ValueError:
        pass
