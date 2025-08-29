# evaluation/acceptance.py
"""Acceptance evaluation and CTAR metrics."""
import json
from typing import Dict, List, Optional, Tuple

import torch

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import set_active_adapter  # for final restore
from training.utils import ctar, free_cuda
from training.modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
    cast_exit_to_base_dtype,
)

__all__ = ["eval_acceptance"]


@torch.inference_mode()
def eval_acceptance(
    spec: EarlyExitLlamaForCausalLM,
    tok,
    prompts: List[str],
    rollout_len: int,
    steps_per_prompt: int = 1,
    dump_debug: bool = False,
    dump_path: Optional[str] = None,
    topk: int = 5,
    quiet: bool = False,
    k_max: int = 4,
) -> Tuple[float, Dict[int, float]]:
    """
    Measure single-step acceptance and CTAR without per-token adapter flips.
    We mimic the block logic used elsewhere, but for k=1.
    """
    spec.eval()
    cast_exit_to_base_dtype(spec)
    dev = next(spec.parameters()).device

    # remember and restore adapter at the end (hygiene)
    _prev_adapter = getattr(spec, "_dvi_active_adapter", None)

    accepts_seq: List[List[int]] = []
    dbg_out: List[Dict] = []

    for p in prompts:
        enc = tok(p, return_tensors="pt").to(dev)
        input_ids = enc["input_ids"]
        attn_mask = enc.get("attention_mask")

        # true last token per sample
        if attn_mask is not None:
            lengths = attn_mask.long().sum(dim=1)
            last_idx = torch.clamp(lengths - 1, min=0)
        else:
            last_idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1,
                                  device=dev, dtype=torch.long)
        ids_last = input_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)

        # prime caches on the full prompt (fast path will select adapters as needed)
        h_k_prompt, shallow_past = run_shallow_until_k(
            spec, input_ids=input_ids, attention_mask=attn_mask,
            past_key_values=None, use_cache=True
        )
        _, deep_past = run_deep_from_k(
            spec, hidden_k=h_k_prompt, past_key_values=None, use_cache=True
        )

        cur_bits: List[int] = []
        total_steps = steps_per_prompt * rollout_len

        for _ in range(total_steps):
            # draft one token from shallow (k=1)
            h_k, tmp_shallow = run_shallow_until_k(
                spec,
                input_ids=ids_last.unsqueeze(1),
                past_key_values=shallow_past,
                attention_mask=None,
                use_cache=True,
            )
            d_logits = exit_logits_from_hidden_k(spec, h_k)      # [B,1,V]
            d_top1 = d_logits[:, -1, :].argmax(dim=-1)           # [B]

            # verify that position with deep from hidden
            deep_logits, deep_past_full = run_deep_from_k(
                spec,
                hidden_k=h_k,
                past_key_values=deep_past,
                use_cache=True,
            )
            v_top1 = deep_logits[:, -1, :].argmax(dim=-1)        # [B]

            # accept bit
            match = v_top1.eq(d_top1)                             # [B]
            cur_bits.append(int(match.all().item()))  # single-sample in practice

            if dump_debug and dump_path:
                try:
                    v_prob, v_id = torch.topk(torch.softmax(deep_logits[0, -1].float(), dim=-1), k=topk)
                    d_prob, d_id = torch.topk(torch.softmax(d_logits[0, -1].float(), dim=-1), k=topk)
                    dbg_out.append({
                        "draft_top1": int(d_top1[0]),
                        "verifier_top1": int(v_top1[0]),
                        "accept": int(match[0]),
                        "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                        "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                    })
                except Exception:
                    pass

            # commit the verifier token and advance both caches by 1
            ids_last = v_top1
            if bool(match.all()):
                # use the shallow snapshot we already computed (accepted)
                shallow_past = tuple((k_.contiguous().clone(), v_.contiguous().clone()) for (k_, v_) in tmp_shallow)
                past_len_d = deep_past[0][0].shape[2] if deep_past and deep_past[0] is not None else 0
                new_deep = []
                for (k_, v_) in deep_past_full:
                    ks = k_[:, :, : past_len_d + 1, :].contiguous().clone()
                    vs = v_[:, :, : past_len_d + 1, :].contiguous().clone()
                    new_deep.append((ks, vs))
                deep_past = tuple(new_deep)
            else:
                # mismatch → recompute shallow/deep with the verifier token
                h_fix, shallow_past = run_shallow_until_k(
                    spec,
                    input_ids=ids_last.unsqueeze(1),
                    past_key_values=shallow_past,
                    attention_mask=None,
                    use_cache=True,
                )
                _, deep_past = run_deep_from_k(
                    spec,
                    hidden_k=h_fix,
                    past_key_values=deep_past,
                    use_cache=True,
                )

        accepts_seq.append(cur_bits)

        if not quiet:
            free_cuda("(eval prompt done)")
        else:
            free_cuda("")

    # metrics
    flat = sum((seq for seq in accepts_seq), [])
    acc = sum(flat) / max(1, len(flat))
    c = {w: ctar(flat, w) for w in range(1, k_max + 1)}  # CTAR over the flat bitstream

    # persist debug (optional)
    if dump_debug and dump_path and dbg_out:
        with open(dump_path, "a") as f:
            for rec in dbg_out:
                f.write(json.dumps(rec) + "\n")

    # restore adapter hygiene
    try:
        if _prev_adapter is not None:
            set_active_adapter(spec, _prev_adapter)
        setattr(spec, "_dvi_active_adapter", _prev_adapter)
    except Exception:
        pass

    return acc, c
