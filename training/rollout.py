"""Rollout utilities and buffer debug helpers."""
from typing import List, Dict, Optional

import json
import torch

from training.buffer import ReplayBuffer
from training.modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
    adapter_guard,
)
from training.sampling import sample_from_logits
from training.align_telemetry import AlignLogger, AlignTelemetryParams

__all__ = ["rollout_collect", "rollout_collect_k_spec", "buf_debug"]


@torch.no_grad()
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


@torch.no_grad()
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
    telemetry: Optional[AlignTelemetryParams] = None,
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
    with adapter_guard(spec, "draft"):
        h_k_prompt, shallow_past = run_shallow_until_k(
            spec, input_ids=input_ids, attention_mask=attn_mask, past_key_values=None, use_cache=True
        )
    with adapter_guard(spec, "verify"), torch.no_grad():
        _, deep_past = run_deep_from_k(
            spec, hidden_k=h_k_prompt, past_key_values=None, use_cache=True
        )
    logger = AlignLogger(telemetry)
    kv_len_shallow = logger.kv_len_from_past(shallow_past)
    kv_len_deep = logger.kv_len_from_past(deep_past)

    # choose true last token (avoid PAD)
    if attn_mask is not None:
        lengths = attn_mask.long().sum(dim=1)
        last_idx = torch.clamp(lengths - 1, min=0)
    else:
        last_idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=device, dtype=torch.long)
    last_tokens = input_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)

    B = input_ids.size(0)
    n_collected = 0
    steps_done = 0

    while steps_done < steps:
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

        kv_len_shallow_before = kv_len_shallow
        kv_len_deep_before = kv_len_deep

        with adapter_guard(spec, "verify"), torch.no_grad():
            deep_logits, deep_past_full = run_deep_from_k(
                spec, hidden_k=hidden_seq, past_key_values=deep_past, use_cache=True
            )

        if not torch.isfinite(deep_logits).all():
            deep_logits = torch.nan_to_num(deep_logits)
            deep_argmax = deep_logits.argmax(dim=-1)
            matches0 = torch.zeros_like(prop_seq, dtype=torch.bool)
        else:
            deep_argmax = deep_logits.argmax(dim=-1)
            matches0 = deep_argmax.eq(prop_seq)

        all_matched0 = matches0.all(dim=1)
        first_mismatch0 = (~matches0).float().argmax(dim=1)
        prefix_lens0 = torch.where(
            all_matched0, torch.full_like(first_mismatch0, k), first_mismatch0
        )
        accept_len_default = int(prefix_lens0.min().item())

        accept_len = accept_len_default
        prefix_lens = prefix_lens0

        if logger.cfg.auto_offset > 0 and deep_argmax.size(1) > 1:
            matches1 = deep_argmax[:, 1:].eq(prop_seq[:, :-1])
            all_matched1 = matches1.all(dim=1)
            first_mismatch1 = (~matches1).float().argmax(dim=1)
            prefix_lens1 = torch.where(
                all_matched1, torch.full_like(first_mismatch1, k - 1), first_mismatch1
            )
            accept_len_p1 = int(prefix_lens1.min().item())
            if accept_len_p1 > accept_len:
                accept_len = accept_len_p1
                prefix_lens = prefix_lens1

        if deep_past_full is None and accept_len > 0:
            accept_len = 0
            prefix_lens = torch.zeros_like(prefix_lens)

        va_list = deep_argmax.detach().cpu().tolist()
        for b in range(B):
            m = int(prefix_lens[b].item())
            if m == k:
                # verifier accepted all k tokens – record each one
                for d in range(k):
                    with torch.inference_mode(False):
                        hidden = hidden_seq[b, d].detach().clone().cpu()
                        vlogits = deep_logits[b, d].detach().clone().cpu()
                    buf.append(
                        hidden=hidden,
                        token=int(va_list[b][d]),
                        reward=1.0,
                        conf=0.0,
                        vlogits=vlogits,
                    )
                kept = k
            else:
                for d in range(m):
                    with torch.inference_mode(False):
                        hidden = hidden_seq[b, d].detach().clone().cpu()
                        vlogits = deep_logits[b, d].detach().clone().cpu()
                    buf.append(
                        hidden=hidden,
                        token=int(va_list[b][d]),
                        reward=1.0,
                        conf=0.0,
                        vlogits=vlogits,
                    )
                d = m
                with torch.inference_mode(False):
                    hidden = hidden_seq[b, d].detach().clone().cpu()
                    vlogits = deep_logits[b, d].detach().clone().cpu()
                buf.append(
                    hidden=hidden,
                    token=int(va_list[b][d]),
                    reward=0.0,
                    conf=0.0,
                    vlogits=vlogits,
                )
                kept = m + 1
            n_collected += kept

        if debug_out is not None:
            try:
                v_prob, v_id = torch.topk(torch.softmax(deep_logits[0, 0].float(), dim=-1), k=topk)
                d_logits = exit_logits_from_hidden_k(spec, hidden_seq[0:1, 0:1, :])
                d_prob, d_id = torch.topk(torch.softmax(d_logits[0, 0].float(), dim=-1), k=topk)
                debug_out.append({
                    "draft_top1": int(prop_seq[0, 0]),
                    "verifier_top1": int(deep_argmax[0, 0]),
                    "accept": int(matches0[0, 0]),
                    "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                    "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                })
            except Exception:
                pass

        # accept common prefix or commit one verifier token on miss
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
        else:
            mismatch_tok = deep_argmax[:, 0]
            # Advance caches with the committed verifier token keeping local
            # ``shallow_past``/``deep_past`` in sync.
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
                    spec,
                    hidden_k=h_fix,
                    past_key_values=deep_past,
                    use_cache=True,
                )
            last_tokens = mismatch_tok

        # persist cache state back onto the model for external observers
        combined_past = tuple(list(shallow_past) + list(deep_past))
        try:
            spec.past_key_values = combined_past
        except Exception:
            pass
        try:
            if hasattr(spec, "model"):
                spec.model.past_key_values = combined_past
        except Exception:
            pass

        kv_len_shallow = logger.kv_len_from_past(shallow_past)
        kv_len_deep = logger.kv_len_from_past(deep_past)

        sample0 = {}
        try:
            sample0["hidden0"] = hidden_seq[0, 0]
            sample0["deep_logits0"] = deep_logits[0, 0]
        except Exception:
            pass

        steps_done += 1
        logger.block_report(
            phase="rollout",
            step_idx=steps_done,
            k=k,
            B=B,
            greedy=greedy,
            temperature=temperature,
            prop_seq=prop_seq,
            deep_logits=deep_logits,
            deep_argmax=deep_argmax,
            accept_len_default=accept_len_default,
            kv_len_shallow_before=kv_len_shallow_before,
            kv_len_deep_before=kv_len_deep_before,
            kv_len_shallow_after=kv_len_shallow,
            kv_len_deep_after=kv_len_deep,
            gold=None,
            sample0_tensors=sample0,
        )

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
