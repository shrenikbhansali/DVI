from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F

from .modeling import (
    run_shallow_until_k,
    run_deep_from_k,
    exit_logits_from_hidden_k,
    cast_exit_to_base_dtype,
)
from .utils import theoretical_compression, count_transformer_layers
from .mem import timing_trace


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
    cast_exit_to_base_dtype(model)
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
        draft_pasts: List[Tuple] = []

        # ----- draft k tokens with shallow stack -----
        prev = last_tokens
        for _ in range(draft_k):
            h_k, draft_shallow_past = run_shallow_until_k(
                model,
                input_ids=prev.unsqueeze(1),
                past_key_values=draft_shallow_past,
                attention_mask=None,
                use_cache=True,
            )
            logits_draft = exit_logits_from_hidden_k(model, h_k)
            nxt = _sample_from_logits(logits_draft[:, -1, :], greedy=greedy, temperature=temperature)
            proposed_tokens.append(nxt)
            proposed_hidden.append(h_k[:, -1, :])
            draft_pasts.append(draft_shallow_past)
            prev = nxt

        metrics.proposed += int(B * draft_k)

        # ----- sequentially verify each drafted token -----
        commit_all = True
        for d in range(draft_k):
            h_d = proposed_hidden[d].unsqueeze(1)
            deep_logits, deep_past_next = run_deep_from_k(
                model,
                hidden_k=h_d,
                past_key_values=deep_past,
                use_cache=True,
            )
            metrics.steps += 1
            v_tok = deep_logits[:, -1, :].argmax(dim=-1)
            match = v_tok.eq(proposed_tokens[d])
            if bool(match.all()):
                shallow_past = draft_pasts[d]
                deep_past = deep_past_next
                tok_list = proposed_tokens[d].tolist()
                for b in range(B):
                    generated[b].append(tok_list[b])
                last_tokens = proposed_tokens[d]
                total_new += 1
                metrics.accepted += int(B)
                if total_new >= max_new_tokens:
                    commit_all = False
                    break
            else:
                # mismatch: fix with verifier token
                commit_all = False
                tok_list = v_tok.tolist()
                h_fix, shallow_past = run_shallow_until_k(
                    model,
                    input_ids=v_tok.unsqueeze(1),
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
                metrics.steps += 1
                for b in range(B):
                    generated[b].append(tok_list[b])
                last_tokens = v_tok
                total_new += 1
                break

        if total_new >= max_new_tokens or not commit_all:
            if total_new >= max_new_tokens:
                break
            continue

    total_layers = count_transformer_layers(model)
    comp_ratio, _ = theoretical_compression(metrics.accept_rate, k, total_layers)
    metrics.comp_ratio_runtime_est = comp_ratio

    outputs = [torch.tensor(seq, device=device, dtype=torch.long) for seq in generated]
    return outputs, metrics
