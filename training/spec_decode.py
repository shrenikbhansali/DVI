from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import os
import torch
import torch.nn.functional as F

from .modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
)
from .utils import theoretical_compression, count_transformer_layers
from .mem import deep_kv_purge

_TRACE = bool(os.getenv("DVI_TIMING_TRACE"))


@dataclass
class SpecMetrics:
    proposed: int = 0
    accepted: int = 0
    steps: int = 0
    comp_ratio_runtime_est: float = 1.0

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    def to_dict(self) -> Dict[str, float]:
        return {
            "spec/proposed": float(self.proposed),
            "spec/accepted": float(self.accepted),
            "spec/accept_rate": float(self.accept_rate),
            "spec/comp_rt_est": float(self.comp_ratio_runtime_est),
        }


def _sample_from_logits(logits: torch.Tensor, greedy: bool, temperature: float) -> torch.Tensor:
    if greedy or temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

@torch.inference_mode()
def generate_with_dvi_spec(
    model,
    tok,
    prompts: Optional[List[str]] = None,
    *,
    enc: Optional[Dict[str, torch.Tensor]] = None,  # can pass already-encoded batch
    max_new_tokens: int = 64,
    draft_k: int = 4,
    greedy: bool = False,
    temperature: float = 1.0,
    early_layer: Optional[int] = None,
    device: Optional[torch.device] = None,
    quiet: bool = True,
) -> Tuple[List[torch.Tensor], SpecMetrics]:
    """Self-speculative decoding using shallow/deep split.

    IMPORTANT: we must seed the first proposed token from the *last real
    token* of each prompt (not the last column, which may be PAD).
    """
    model.eval()
    device = device or next(model.parameters()).device
    k = early_layer if early_layer is not None else getattr(model, "early_layer", None)
    assert isinstance(k, int) and k > 0

    # --- use pre-encoded inputs if provided; otherwise tokenize once ---
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

    B = input_ids.size(0)

    # ---- choose the TRUE last token per sample (avoid PAD) ----
    if attn_mask is not None:
        # lengths = number of non-pad tokens in each row
        lengths = attn_mask.long().sum(dim=1)                            # [B]
        last_idx = torch.clamp(lengths - 1, min=0)                       # [B]
    else:
        last_idx = torch.full((B,), input_ids.size(1) - 1,
                              device=device, dtype=torch.long)           # [B]
    last_tokens = input_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # [B]

    generated: List[List[int]] = [[] for _ in range(B)]

    # prime shallow & deep KV caches on the full prompt
    h_k_prompt, shallow_past = run_shallow_until_k(
        model,
        input_ids=input_ids,
        attention_mask=attn_mask,
        past_key_values=None,
        use_cache=True,
    )
    _, deep_past = run_deep_from_k(
        model,
        hidden_k=h_k_prompt,
        past_key_values=None,
        use_cache=True,
    )

    metrics = SpecMetrics()
    total_new = 0

    while total_new < max_new_tokens:
        proposed_tokens: List[torch.Tensor] = []
        proposed_hidden: List[torch.Tensor] = []
        draft_shallow_past = shallow_past

        # ----- draft k tokens with shallow stack -----
        for d in range(draft_k):
            prev = last_tokens if d == 0 else proposed_tokens[-1]
            h_k, draft_shallow_past = run_shallow_until_k(
                model,
                input_ids=prev.unsqueeze(1),
                past_key_values=draft_shallow_past,
                attention_mask=None,
                use_cache=True,
            )
            logits_draft = exit_logits_from_hidden_k(model, h_k)  # [B,1,V]
            next_tok = _sample_from_logits(
                logits_draft[:, -1, :], greedy=greedy, temperature=temperature
            )
            proposed_tokens.append(next_tok)
            proposed_hidden.append(h_k[:, -1, :])

        prop_seq = torch.stack(proposed_tokens, dim=1)   # [B, k]
        hidden_seq = torch.stack(proposed_hidden, dim=1) # [B, k, H]
        metrics.proposed += int(B * draft_k)

        # ----- verify with deep stack in one shot -----
        with torch.no_grad():
            deep_logits, deep_past_full = run_deep_from_k(
                model,
                hidden_k=hidden_seq,
                past_key_values=deep_past,
                use_cache=True,
            )
        verify_argmax = deep_logits.argmax(dim=-1)  # [B, k]

        # compute common accept prefix length (per sample)
        matches = verify_argmax.eq(prop_seq)  # [B, k]
        prefix_lens = []
        for b in range(B):
            row = matches[b]
            if torch.all(row):
                m = draft_k
            else:
                nz = (~row).nonzero(as_tuple=False)
                m = int(nz[0].item()) if nz.numel() else draft_k
            prefix_lens.append(m)
        prefix_lens = torch.tensor(prefix_lens, device=device)
        accept_len = int(prefix_lens.min().item())

        # ----- accept the common prefix -----
        if accept_len > 0:
            accepted_block = prop_seq[:, :accept_len]
            past_len_s = shallow_past[0][0].shape[2] if shallow_past and shallow_past[0] is not None else 0
            past_len_d = deep_past[0][0].shape[2] if deep_past and deep_past[0] is not None else 0

            shallow_past = tuple(
                (
                    k[:, :, : past_len_s + accept_len, :].clone(),
                    v[:, :, : past_len_s + accept_len, :].clone(),
                )
                for (k, v) in draft_shallow_past
            )
            deep_past = tuple(
                (
                    k[:, :, : past_len_d + accept_len, :].clone(),
                    v[:, :, : past_len_d + accept_len, :].clone(),
                )
                for (k, v) in deep_past_full
            )
            if _TRACE:
                try:
                    k0 = shallow_past[0][0]
                    print(
                        f"[trace] compact shallow layer0 shape={tuple(k0.shape)} storage={k0.storage().size()}",
                        flush=True,
                    )
                except Exception:
                    pass
            for b in range(B):
                generated[b].extend(accepted_block[b].tolist())
            last_tokens = accepted_block[:, -1]
            total_new += accept_len
            metrics.accepted += int(B * accept_len)
            metrics.steps += 1
            if total_new >= max_new_tokens:
                break

        # ----- single fix token when a mismatch occurs -----
        if accept_len < draft_k:
            mismatch_tok = verify_argmax[:, accept_len]
            h_fix, shallow_past = run_shallow_until_k(
                model,
                input_ids=mismatch_tok.unsqueeze(1),
                past_key_values=shallow_past,
                attention_mask=None,
                use_cache=True,
            )
            _, deep_past = run_deep_from_k(
                model,
                hidden_k=h_fix,
                past_key_values=deep_past,
                use_cache=True,
            )
            for b in range(B):
                generated[b].append(int(mismatch_tok[b].item()))
            last_tokens = mismatch_tok
            total_new += 1
            metrics.steps += 1

    total_layers = count_transformer_layers(model)
    comp_ratio, _ = theoretical_compression(metrics.accept_rate, k, total_layers)
    metrics.comp_ratio_runtime_est = comp_ratio

    outputs = [torch.tensor(seq, device=device, dtype=torch.long) for seq in generated]
    deep_kv_purge(model, tag="generate_with_dvi_spec-end")
    return outputs, metrics
