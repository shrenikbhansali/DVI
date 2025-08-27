"""Rollout utilities and buffer debug helpers."""
from typing import List, Dict, Optional

import json
import torch
import torch.nn.functional as F

from training.kv import drafter_hidden_no_cache, advance_kv_with_committed, prime_kv_full
from training.buffer import ReplayBuffer
from training.modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
)
from training.mem import timing_trace

__all__ = ["rollout_collect", "rollout_collect_k_spec", "buf_debug"]


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


def _sample_from_logits(logits: torch.Tensor, greedy: bool, temperature: float) -> torch.Tensor:
    if greedy or temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def rollout_collect_k_spec(
    spec,
    tok,
    prompt: str,
    buf: ReplayBuffer,
    steps: int,
    *,
    k: int = 1,
    greedy: bool = False,
    temperature: float = 1.0,
    debug_out: Optional[List[Dict]] = None,
    topk: int = 5,
) -> int:
    """Collect rollouts using k-token speculative drafting.

    This mirrors ``training/spec_decode.generate_with_dvi_spec`` but instead of
    generating text it pushes per-token records into ``buf`` for training.
    """

    spec.eval()
    device = next(spec.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # --- prime KV caches on the prompt ---
    h_k_prompt, shallow_past = run_shallow_until_k(
        spec, input_ids=input_ids, attention_mask=attn_mask, past_key_values=None, use_cache=True
    )
    with torch.no_grad():
        _, deep_past = run_deep_from_k(
            spec, hidden_k=h_k_prompt, past_key_values=None, use_cache=True
        )

    # choose true last token (avoid PAD)
    if attn_mask is not None:
        lengths = attn_mask.long().sum(dim=1)
        last_idx = torch.clamp(lengths - 1, min=0)
    else:
        last_idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=device, dtype=torch.long)
    last_tokens = input_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)

    B = input_ids.size(0)
    n_collected = 0

    while n_collected < steps:
        draft_tokens = []
        draft_hidden = []
        tmp_shallow = shallow_past
        prev = last_tokens
        for _ in range(k):
            h_k, tmp_shallow = run_shallow_until_k(
                spec,
                input_ids=prev.unsqueeze(1),
                past_key_values=tmp_shallow,
                attention_mask=None,
                use_cache=True,
            )
            logits = exit_logits_from_hidden_k(spec, h_k)  # [B,1,V]
            nxt = _sample_from_logits(logits[:, -1, :], greedy=greedy, temperature=temperature)
            draft_tokens.append(nxt)
            draft_hidden.append(h_k[:, -1, :])
            prev = nxt

        prop_seq = torch.stack(draft_tokens, dim=1)        # [B,k]
        hidden_seq = torch.stack(draft_hidden, dim=1)      # [B,k,H]

        with torch.no_grad():
            deep_logits, deep_past_full = run_deep_from_k(
                spec, hidden_k=hidden_seq, past_key_values=deep_past, use_cache=True
            )

        verify_argmax = deep_logits.argmax(dim=-1)         # [B,k]
        matches = verify_argmax.eq(prop_seq)
        rewards = matches.float()

        # per-sample accepted prefix length
        all_matched = matches.all(dim=1)
        first_mismatch = (~matches).float().argmax(dim=1)
        prefix_lens = torch.where(
            all_matched, torch.full_like(first_mismatch, k), first_mismatch
        )
        accept_len = int(prefix_lens.min().item())

        # append drafted positions to buffer
        va_list = verify_argmax.detach().cpu().tolist()
        rw_list = rewards.detach().cpu().tolist()
        for b in range(B):
            for d in range(k):
                buf.append(
                    hidden=hidden_seq[b, d].detach().cpu(),
                    token=int(va_list[b][d]),
                    reward=float(rw_list[b][d]),
                    conf=0.0,
                    vlogits=deep_logits[b, d].detach().cpu(),
                )

        n_collected += B * k

        if debug_out is not None:
            try:
                v_prob, v_id = torch.topk(torch.softmax(deep_logits[0, 0].float(), dim=-1), k=topk)
                d_logits = exit_logits_from_hidden_k(spec, hidden_seq[0:1, 0:1, :])
                d_prob, d_id = torch.topk(torch.softmax(d_logits[0, 0].float(), dim=-1), k=topk)
                debug_out.append({
                    "draft_top1": int(prop_seq[0, 0]),
                    "verifier_top1": int(verify_argmax[0, 0]),
                    "accept": int(matches[0, 0]),
                    "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                    "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                })
            except Exception:
                pass

        # accept common prefix
        if accept_len > 0:
            accepted_block = prop_seq[:, :accept_len]
            past_len_s = shallow_past[0][0].shape[2] if shallow_past and shallow_past[0] is not None else 0
            past_len_d = deep_past[0][0].shape[2] if deep_past and deep_past[0] is not None else 0
            new_shallow = []
            for (k_, v_) in tmp_shallow:
                ks = k_[:, :, : past_len_s + accept_len, :].contiguous().clone()
                vs = v_[:, :, : past_len_s + accept_len, :].contiguous().clone()
                new_shallow.append((ks, vs))
                storage_elems = ks.untyped_storage().nbytes() // ks.element_size()
                timing_trace(
                    f"rollout compact shallow KV to {ks.shape} storage={storage_elems}"
                )
            shallow_past = tuple(new_shallow)
            new_deep = []
            for (k_, v_) in deep_past_full:
                ks = k_[:, :, : past_len_d + accept_len, :].contiguous().clone()
                vs = v_[:, :, : past_len_d + accept_len, :].contiguous().clone()
                new_deep.append((ks, vs))
                storage_elems = ks.untyped_storage().nbytes() // ks.element_size()
                timing_trace(
                    f"rollout compact deep KV to {ks.shape} storage={storage_elems}"
                )
            deep_past = tuple(new_deep)
            last_tokens = accepted_block[:, -1]

        # fix single mismatch
        if accept_len < k:
            mismatch_tok = verify_argmax[:, accept_len]
            h_fix, shallow_past = run_shallow_until_k(
                spec,
                input_ids=mismatch_tok.unsqueeze(1),
                past_key_values=shallow_past,
                attention_mask=None,
                use_cache=True,
            )
            with torch.no_grad():
                _, deep_past = run_deep_from_k(
                    spec, hidden_k=h_fix, past_key_values=deep_past, use_cache=True
                )
            last_tokens = mismatch_tok

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
