from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
from .align_telemetry import AlignLogger, AlignTelemetryParams


@dataclass
class SpecMetrics:
    proposed: int = 0
    accepted: int = 0
    committed: int = 0
    steps: int = 0
    deep_tokens: int = 0
    comp_ratio_runtime_est: float = 1.0
    prefix_hist: List[int] = field(default_factory=list)
    draft_k: int = 0
    invariant_error: int = 0
    nan_events: int = 0
    oom_retries: int = 0

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def deep_to_commit(self) -> float:
        return self.deep_tokens / max(1, self.committed)

    def _derive_from_hist(self) -> Tuple[float, float, float, float, List[float]]:
        k = int(self.draft_k)
        hist = self.prefix_hist
        N = max(1, sum(hist))
        EL = sum(i * c for i, c in enumerate(hist)) / N
        p0 = hist[0] / N if len(hist) > 0 else 0.0
        pfull = hist[k] / N if k < len(hist) else 0.0
        pge2 = sum(hist[2:k + 1]) / N if k >= 2 else 0.0
        ctar_hat = [1.0 - p0]
        for j in range(2, k + 1):
            ctar_hat.append(sum(hist[j:k + 1]) / N)
        return EL, p0, pfull, pge2, ctar_hat

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
            "spec/invariant_error": float(self.invariant_error),
            "spec/nan_events": float(self.nan_events),
            "spec/oom_retries": float(self.oom_retries),
        }
        hist_N = int(sum(self.prefix_hist))
        out["spec/hist_N"] = float(hist_N)
        for i, v in enumerate(self.prefix_hist):
            out[f"spec/prefix_hist_{i}"] = float(v)
        if self.draft_k:
            EL, p0, pfull, pge2, ctar_hat = self._derive_from_hist()
            out["spec/E[L]"] = float(EL)
            out["spec/p0"] = float(p0)
            out["spec/p_full"] = float(pfull)
            out["spec/p_ge2"] = float(pge2)
            for j, val in enumerate(ctar_hat, start=1):
                out[f"spec/ctar{j}_hat"] = float(val)
            acc_from_hist = EL / max(1, self.draft_k)
            out["spec/accept_rate_from_hist"] = float(acc_from_hist)
            out["spec/accept_rate_delta"] = float(out["spec/accept_rate"] - acc_from_hist)
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
    telemetry: Optional[AlignTelemetryParams] = None,
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

    logger = AlignLogger(telemetry)
    kv_len_shallow = logger.kv_len_from_past(shallow_past)
    kv_len_deep = logger.kv_len_from_past(deep_past)

    metrics = SpecMetrics(draft_k=draft_k)
    metrics.prefix_hist = [0 for _ in range(draft_k + 1)]
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

        prop_seq = torch.stack(draft_tokens, dim=1)   # [B,k]
        hidden_seq = torch.stack(draft_hidden, dim=1) # [B,k,H]
        metrics.proposed += int(B * draft_k)

        # ---- verify entire block ----
        kv_len_shallow_before = kv_len_shallow
        kv_len_deep_before = kv_len_deep

        with adapter_guard(model, "verify"):
            deep_logits, deep_past_full = run_deep_from_k(
                model,
                hidden_k=hidden_seq,
                past_key_values=deep_past,
                use_cache=True,
            )
        metrics.steps += 1
        metrics.deep_tokens += int(B * draft_k)

        if not torch.isfinite(deep_logits).all():
            metrics.nan_events += 1
            deep_logits = torch.nan_to_num(deep_logits)

        deep_argmax = deep_logits.argmax(dim=-1)  # [B,k]
        matches0 = deep_argmax.eq(prop_seq)

        all_matched0 = matches0.all(dim=1)
        first_mismatch0 = (~matches0).float().argmax(dim=1)
        prefix_lens0 = torch.where(all_matched0, torch.full_like(first_mismatch0, draft_k), first_mismatch0)
        counts = torch.bincount(prefix_lens0.to(dtype=torch.long, device="cpu"), minlength=draft_k + 1)
        for i in range(draft_k + 1):
            metrics.prefix_hist[i] += int(counts[i])
        accept_len_default = int(prefix_lens0.min().item())

        accept_len = accept_len_default

        if logger.cfg.auto_offset > 0 and deep_argmax.size(1) > 1:
            matches1 = deep_argmax[:, 1:].eq(prop_seq[:, :-1])
            all_matched1 = matches1.all(dim=1)
            first_mismatch1 = (~matches1).float().argmax(dim=1)
            prefix_lens1 = torch.where(
                all_matched1, torch.full_like(first_mismatch1, draft_k - 1), first_mismatch1
            )
            accept_len_p1 = int(prefix_lens1.min().item())
            if accept_len_p1 > accept_len:
                accept_len = accept_len_p1

        metrics.accepted += int(B * accept_len)

        # commit accepted prefix or one verifier token on miss
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
            metrics.committed += int(B * accept_len)
        else:
            v1 = deep_argmax[:, 0]
            for b in range(B):
                generated[b].append(int(v1[b]))
            last_tokens = v1
            with adapter_guard(model, "draft"):
                h_fix, shallow_past = run_shallow_until_k(
                    model,
                    input_ids=v1.unsqueeze(1),
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
            total_new += 1
            metrics.committed += int(B)

        kv_len_shallow = logger.kv_len_from_past(shallow_past)
        kv_len_deep = logger.kv_len_from_past(deep_past)

        sample0 = {}
        try:
            sample0["prop_seq0"] = prop_seq[0]
            sample0["deep_argmax0"] = deep_argmax[0]
            sample0["deep_logits0"] = deep_logits[0]
        except Exception:
            pass

        logger.block_report(
            phase="spec",
            step_idx=metrics.steps,
            k=draft_k,
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

        # invariants: accept_rate vs histogram
        if metrics.proposed > 0:
            hist_EL, _, _, _, _ = metrics._derive_from_hist()
            acc_from_hist = hist_EL / max(1, metrics.draft_k)
            if abs(metrics.accept_rate - acc_from_hist) > 1e-6:
                metrics.invariant_error += 1

        if total_new >= max_new_tokens:
            break

    total_layers = count_transformer_layers(model)
    comp_ratio, _ = theoretical_compression(metrics.accept_rate, k, total_layers)
    metrics.comp_ratio_runtime_est = comp_ratio

    outputs = [torch.tensor(seq, device=device, dtype=torch.long) for seq in generated]
    return outputs, metrics
