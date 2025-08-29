from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
    cast_exit_to_base_dtype,
    adapter_guard,
)
from .sampling import sample_from_logits
from .utils import theoretical_compression, count_transformer_layers
# timing_trace is kept for compatibility; may be used externally
from .mem import timing_trace


@dataclass
class SpecMetrics:
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    steps: int = 0
    deep_tokens: int = 0
    comp_ratio_runtime_est: float = 1.0

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def deep_to_commit(self) -> float:
        return self.deep_tokens / max(1, self.committed)

    def to_dict(self) -> Dict[str, float]:
        return {
            "spec/proposed": float(self.proposed),
            "spec/accepted": float(self.accepted),
            "spec/committed": float(self.committed),
            "spec/accept_rate": float(self.accept_rate),
            "spec/deep_tokens": float(self.deep_tokens),
            "spec/deep_to_commit": float(self.deep_to_commit),
            "spec/comp_rt_est": float(self.comp_ratio_runtime_est),
        }

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
    cast_exit_to_base_dtype(model)
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

    # Prime KV caches on the prompt
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
    total_new = 0

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

        prop_seq = torch.stack(draft_tokens, dim=1)  # [B,k]
        hidden_seq = torch.stack(draft_hidden, dim=1)  # [B,k,H]
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

        verify_argmax = deep_logits.argmax(dim=-1)  # [B,k]
        matches = verify_argmax.eq(prop_seq)

        all_matched = matches.all(dim=1)
        first_mismatch = (~matches).float().argmax(dim=1)
        prefix_lens = torch.where(all_matched, torch.full_like(first_mismatch, draft_k), first_mismatch)
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

        # ---- optional single-token fix ----
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
