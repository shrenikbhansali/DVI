from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import os
import torch

from .modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
    adapter_guard,
)
from .sampling import sample_from_logits
from .utils import theoretical_compression, count_transformer_layers
from .mem import timing_trace  # kept for compatibility
from .kv import clear_all_kv, prime_kv_full, advance_kv_with_committed


@dataclass
class SpecMetrics:
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    steps: int = 0
    deep_tokens: int = 0
    comp_ratio_runtime_est: float = 1.0
    prefix_hist: List[int] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def deep_to_commit(self) -> float:
        return self.deep_tokens / max(1, self.committed)

    def to_dict(self) -> Dict[str, float]:
        out = {
            "spec/proposed": float(self.proposed),
            "spec/accepted": float(self.accepted),
            "spec/committed": float(self.committed),
            "spec/accept_rate": float(self.accept_rate),
            "spec/deep_tokens": float(self.deep_tokens),
            "spec/deep_to_commit": float(self.deep_to_commit),
            "spec/comp_rt_est": float(self.comp_ratio_runtime_est),
            "spec/steps": float(self.steps),
        }
        for i, v in enumerate(self.prefix_hist):
            out[f"spec/prefix_hist_{i}"] = float(v)
        return out


# Generation only requires gradients disabled; using ``torch.no_grad`` avoids
# producing inference tensors that could accidentally leak into later
# optimisation steps.
@torch.no_grad()
def generate_with_dvi_spec(
    model,
    tok,
    prompts: Optional[List[str]] = None,
    *,
    enc: Optional[Dict[str, torch.Tensor]] = None,
    max_new_tokens: int = 64,
    draft_k: int = 4,
    greedy: bool = False,
    temperature: float = 1.0,
    early_layer: Optional[int] = None,
    device: Optional[torch.device] = None,
    quiet: bool = True,
) -> Tuple[List[torch.Tensor], SpecMetrics]:
    """Self-speculative decoding using vectorised block verification."""

    model.eval()
    device = device or next(model.parameters()).device
    k = early_layer if early_layer is not None else getattr(model, "early_layer", None)
    assert isinstance(k, int) and k > 0

    # Encode inputs once if needed
    if enc is not None:
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
    else:
        assert prompts is not None, "Either `enc` or `prompts` must be provided."
        enc = tok(prompts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

    B = int(input_ids.size(0))

    # Choose the true last token per sample (avoid PAD)
    if attn_mask is not None:
        lengths = attn_mask.long().sum(dim=1)
        last_idx = torch.clamp(lengths - 1, min=0)
    else:
        last_idx = torch.full((B,), input_ids.size(1) - 1, device=device, dtype=torch.long)
    last_tokens = input_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)

    generated: List[List[int]] = [[] for _ in range(B)]

    # Prime KV caches on the prompt (strict adapter separation)
    with adapter_guard(model, "draft"):
        h_k_prompt, shallow_past = run_shallow_until_k(
            model,
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=None,
            use_cache=True,
        )
    with adapter_guard(model, "verify"):
        _, deep_past = run_deep_from_k(
            model,
            hidden_k=h_k_prompt,
            past_key_values=None,
            use_cache=True,
        )

    metrics = SpecMetrics()
    metrics.prefix_hist = [0 for _ in range(draft_k + 1)]
    total_new = 0

    # align debug throttling
    if not hasattr(generate_with_dvi_spec, "_align_prints"):
        generate_with_dvi_spec._align_prints = 0
    max_align_prints = int(os.getenv("DVI_ALIGN_PRINTS", "3"))

    while total_new < max_new_tokens:
        draft_tokens: List[torch.Tensor] = []
        draft_hidden: List[torch.Tensor] = []
        shallow_snapshots: List[Tuple] = []
        tmp_shallow = shallow_past
        prev = last_tokens

        # ---- draft k tokens ----
        for _ in range(draft_k):
            with adapter_guard(model, "draft"):
                h_k, tmp_shallow = run_shallow_until_k(
                    model,
                    input_ids=prev.unsqueeze(1),
                    past_key_values=tmp_shallow,
                    attention_mask=None,
                    use_cache=True,
                )
                logits = exit_logits_from_hidden_k(model, h_k)
            nxt = sample_from_logits(logits[:, -1, :], greedy=greedy, temperature=temperature)
            draft_tokens.append(nxt)
            draft_hidden.append(h_k[:, -1, :])
            shallow_snapshots.append(tmp_shallow)
            prev = nxt

        prop_seq = torch.stack(draft_tokens, dim=1)   # [B,k]
        hidden_seq = torch.stack(draft_hidden, dim=1) # [B,k,H]
        metrics.proposed += int(B * draft_k)

        # ---- verify entire block ----
        with adapter_guard(model, "verify"):
            deep_logits, deep_past_full = run_deep_from_k(
                model,
                hidden_k=hidden_seq,
                past_key_values=deep_past,
                use_cache=True,
            )
        metrics.steps += 1
        metrics.deep_tokens += int(B * draft_k)
        deep_calls = 1

        # --- Misalignment debug (prints to stdout if requested) ---
        if os.getenv("DVI_ALIGN_DEBUG", ""):
            _va = deep_logits.argmax(dim=-1)  # [B, k]
            m00 = (_va[:, 0] == prop_seq[:, 0]).float().mean().item()
            m10 = float("nan")
            m11 = float("nan")
            if _va.size(1) > 1:
                m10 = (_va[:, 1] == prop_seq[:, 0]).float().mean().item()
                m11 = (_va[:, 1] == prop_seq[:, 1]).float().mean().item()

            if generate_with_dvi_spec._align_prints < max_align_prints:
                print(
                    f"[align/spec] k={draft_k} match(0↔0)={m00:.3f} "
                    f"match(1↔0)={m10:.3f} match(1↔1)={m11:.3f}",
                    flush=True,
                )
                generate_with_dvi_spec._align_prints += 1

            # Optional: deeper "gold" check for B==1
            if os.getenv("DVI_ALIGN_GOLD", "") and B == 1 and generate_with_dvi_spec._align_prints <= max_align_prints:
                try:
                    with adapter_guard(model, "verify"):
                        # Build current prefix for sample 0
                        prefix_ids = input_ids[0].detach().tolist() + generated[0]
                        prefix = torch.tensor([prefix_ids], device=device, dtype=input_ids.dtype)
                        clear_all_kv(model)
                        if prefix.size(1) >= 1:
                            prime_kv_full(model, prefix[:, :-1])
                            last = prefix[:, -1:]
                        else:
                            prime_kv_full(model, prefix)
                            last = prefix[:, -1:]

                        gold = []
                        steps_to_check = min(2, prop_seq.size(1))
                        for _ in range(steps_to_check):
                            g_logits = model.verifier_logits_for_next(last)
                            g_top1 = int(g_logits.argmax(dim=-1)[0, 0].item())
                            gold.append(g_top1)
                            nxt = torch.tensor([[g_top1]], device=last.device, dtype=last.dtype)
                            advance_kv_with_committed(model, nxt)
                            last = nxt

                    v_arg = deep_logits.argmax(dim=-1)  # [1,k]
                    d0 = int(prop_seq[0, 0].item())
                    d1 = int(prop_seq[0, 1].item()) if prop_seq.size(1) > 1 else -1
                    v0 = int(v_arg[0, 0].item())
                    v1 = int(v_arg[0, 1].item()) if v_arg.size(1) > 1 else -1
                    g0 = gold[0] if len(gold) > 0 else -1
                    g1 = gold[1] if len(gold) > 1 else -1
                    print(
                        f"[align/spec] gold k={draft_k} "
                        f"deep≈gold: {(v0==g0):d}/{(v1==g1):d} | "
                        f"draft={d0},{d1} deep={v0},{v1} gold={g0},{g1}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[align/spec] gold-check error: {e}", flush=True)
        # --- end misalignment debug ---

        verify_argmax = deep_logits.argmax(dim=-1)  # [B,k]
        matches = verify_argmax.eq(prop_seq)

        # compute accepted prefix length across batch (min)
        all_matched = matches.all(dim=1)
        first_mismatch = (~matches).float().argmax(dim=1)
        prefix_lens = torch.where(all_matched, torch.full_like(first_mismatch, draft_k), first_mismatch)
        hist = torch.bincount(prefix_lens, minlength=draft_k + 1)
        for i in range(draft_k + 1):
            metrics.prefix_hist[i] += int(hist[i].item())
        accept_len = int(prefix_lens.min().item())

        # commit accepted prefix
        if accept_len > 0:
            accepted_block = prop_seq[:, :accept_len]
            snap = shallow_snapshots[accept_len - 1]
            new_shallow = []
            for (k_, v_) in snap:
                new_shallow.append((k_.contiguous().clone(), v_.contiguous().clone()))
            shallow_past = tuple(new_shallow)

            past_len_d = deep_past[0][0].shape[2] if deep_past and deep_past[0] is not None else 0
            new_deep = []
            for (k_, v_) in deep_past_full:
                ks = k_[:, :, : past_len_d + accept_len, :].contiguous().clone()
                vs = v_[:, :, : past_len_d + accept_len, :].contiguous().clone()
                new_deep.append((ks, vs))
            deep_past = tuple(new_deep)

            acc_list = accepted_block.tolist()
            for b in range(B):
                generated[b].extend(acc_list[b])
            last_tokens = accepted_block[:, -1]
            total_new += accept_len
            metrics.accepted += int(B * accept_len)
            metrics.committed += int(B * accept_len)

        if total_new >= max_new_tokens:
            break

        # ---- optional single-token fix (commit one deep token when mismatch) ----
        if accept_len < draft_k:
            mismatch_tok = verify_argmax[:, accept_len]
            with adapter_guard(model, "draft"):
                h_fix, shallow_past = run_shallow_until_k(
                    model,
                    input_ids=mismatch_tok.unsqueeze(1),
                    past_key_values=shallow_past,
                    attention_mask=None,
                    use_cache=True,
                )
            with adapter_guard(model, "verify"):
                _, deep_past = run_deep_from_k(
                    model,
                    hidden_k=h_fix,
                    past_key_values=deep_past,
                    use_cache=True,
                )
            metrics.steps += 1
            metrics.deep_tokens += int(B)
            deep_calls += 1

            tok_list = mismatch_tok.tolist()
            for b in range(B):
                generated[b].append(tok_list[b])
            last_tokens = mismatch_tok
            total_new += 1
            metrics.committed += int(B)

        # safety: ensure ≤ 2 deep calls per block
        assert deep_calls <= 2, "Deep called more than twice in a block"

    total_layers = count_transformer_layers(model)
    comp_ratio, _ = theoretical_compression(metrics.accept_rate, k, total_layers)
    metrics.comp_ratio_runtime_est = comp_ratio

    outputs = [torch.tensor(seq, device=device, dtype=torch.long) for seq in generated]
    return outputs, metrics
