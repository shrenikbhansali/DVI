"""Rollout utilities and buffer debug helpers."""
from typing import List, Dict, Optional

import json
import torch
from typing import List, Dict, Optional

from training.buffer import ReplayBuffer
from training.modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
    cast_exit_to_base_dtype,
    adapter_guard,
)
from training.sampling import sample_from_logits

__all__ = ["rollout_collect", "rollout_collect_k_spec", "buf_debug"]


@torch.inference_mode()
def rollout_collect(spec, tok, prompt: str,
                    buf: ReplayBuffer, steps: int,
                    debug_out: Optional[List[Dict]] = None, topk: int = 5) -> int:
    """Legacy single-token rollout forwarded to k-spec path."""
    return rollout_collect_k_spec(
        spec,
        tok,
        prompt,
        buf,
        steps,
        k=1,
        greedy=True,
        temperature=1.0,
        debug_out=debug_out,
        topk=topk,
    )


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
    cast_exit_to_base_dtype(spec)
    device = next(spec.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # --- prime KV caches on the prompt ---
    with adapter_guard(spec, "draft"):
        h_k_prompt, shallow_past = run_shallow_until_k(
            spec, input_ids=input_ids, attention_mask=attn_mask, past_key_values=None, use_cache=True
        )
    with adapter_guard(spec, "verify"), torch.no_grad():
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
        shallow_snaps = []
        for _ in range(k):
            with adapter_guard(spec, "draft"):
                h_k, tmp_shallow = run_shallow_until_k(
                    spec,
                    input_ids=prev.unsqueeze(1),
                    past_key_values=tmp_shallow,
                    attention_mask=None,
                    use_cache=True,
                )
                logits = exit_logits_from_hidden_k(spec, h_k)
            nxt = sample_from_logits(logits[:, -1, :], greedy=greedy, temperature=temperature)
            draft_tokens.append(nxt)
            draft_hidden.append(h_k[:, -1, :])
            shallow_snaps.append(tmp_shallow)
            prev = nxt

        prop_seq = torch.stack(draft_tokens, dim=1)
        hidden_seq = torch.stack(draft_hidden, dim=1)

        with adapter_guard(spec, "verify"), torch.no_grad():
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
            snap = shallow_snaps[accept_len - 1]
            new_shallow = []
            for (k_, v_) in snap:
                ks = k_.contiguous().clone()
                vs = v_.contiguous().clone()
                new_shallow.append((ks, vs))
            shallow_past = tuple(new_shallow)
            past_len_d = deep_past[0][0].shape[2] if deep_past and deep_past[0] is not None else 0
            new_deep = []
            for (k_, v_) in deep_past_full:
                ks = k_[:, :, : past_len_d + accept_len, :].contiguous().clone()
                vs = v_[:, :, : past_len_d + accept_len, :].contiguous().clone()
                new_deep.append((ks, vs))
            deep_past = tuple(new_deep)
            last_tokens = accepted_block[:, -1]

        # fix single mismatch
        if accept_len < k:
            mismatch_tok = verify_argmax[:, accept_len]
            with adapter_guard(spec, "draft"):
                h_fix, shallow_past = run_shallow_until_k(
                    spec,
                    input_ids=mismatch_tok.unsqueeze(1),
                    past_key_values=shallow_past,
                    attention_mask=None,
                    use_cache=True,
                )
            with adapter_guard(spec, "verify"), torch.no_grad():
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
