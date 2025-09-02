"""Key-value cache helpers.

These utilities ensure that probing the drafter path does not mutate the
underlying KV cache.  They mirror the behaviour of the smoketest script
but are importable by both the trainer and the evaluator.
"""
from typing import List, Tuple, Optional

import torch

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.modeling import adapter_guard

__all__ = [
    "_kv_snapshot",
    "_kv_restore",
    "_first_nonempty_pkv",
    "estimate_kv_cache",
    "clear_all_kv",
    "prime_kv_full",
    "advance_kv_with_committed",
    "persist_kv_cache",
    "drafter_hidden_no_cache",
]


def _kv_snapshot(spec) -> List[Tuple[object, str, Optional[object]]]:
    slots = []
    for obj in (spec, getattr(spec, "model", None)):
        if obj is None:
            continue
        for attr in ("past_key_values", "_past_key_values"):
            if hasattr(obj, attr):
                slots.append((obj, attr, getattr(obj, attr, None)))
    return slots


def _kv_restore(slots: List[Tuple[object, str, Optional[object]]]) -> None:
    for obj, attr, val in slots:
        try:
            setattr(obj, attr, val)
        except Exception:
            pass


def _first_nonempty_pkv(spec: EarlyExitLlamaForCausalLM):
    for obj in (spec, getattr(spec, "model", None)):
        if obj is None:
            continue
        for name in ("past_key_values", "_past_key_values"):
            if hasattr(obj, name):
                pkv = getattr(obj, name)
                if pkv:
                    return pkv
    return None


def estimate_kv_cache(spec: EarlyExitLlamaForCausalLM) -> Tuple[int, int]:
    pkv = _first_nonempty_pkv(spec)
    if not pkv:
        return 0, 0
    total_bytes = 0
    est_seq_len = 0
    try:
        for layer in pkv:
            for t in layer:
                if isinstance(t, torch.Tensor):
                    total_bytes += t.element_size() * t.nelement()
                    if t.ndim >= 2:
                        est_seq_len = max(est_seq_len, int(t.shape[-2]))
    except Exception:
        pass
    return int(total_bytes), int(est_seq_len)



def clear_all_kv(spec, verbose: bool = False, tag: str = "") -> None:
    touched = []
    for obj in (spec, getattr(spec, "model", None)):
        if obj is None:
            continue
        for name in ("past_key_values", "_past_key_values"):
            if hasattr(obj, name):
                try:
                    setattr(obj, name, None)
                    touched.append(f"{obj.__class__.__name__}.{name}")
                except Exception:
                    pass
    if verbose:
        msg = ", ".join(touched) if touched else "none"
        print(f"[kv-clear]{(' '+tag) if tag else ''} -> {msg}", flush=True)


# These helpers are used during evaluation and rollout to manipulate KV caches.
# We only need gradients disabled, so ``torch.no_grad`` suffices and avoids
# generating inference tensors that might later leak into training.
@torch.no_grad()
def prime_kv_full(spec: EarlyExitLlamaForCausalLM, input_ids: torch.Tensor) -> None:
    clear_all_kv(spec)
    with adapter_guard(spec, "draft"):
        h = spec.forward_draft_or_large_model(in_tokens_small=input_ids)
    with adapter_guard(spec, "verify"):
        _ , _ = spec.forward_draft_or_large_model(in_features_large=h)


@torch.no_grad()
def advance_kv_with_committed(spec: EarlyExitLlamaForCausalLM, token_ids: torch.Tensor) -> None:
    with adapter_guard(spec, "draft"):
        h = spec.forward_draft_or_large_model(in_tokens_small=token_ids)
    with adapter_guard(spec, "verify"):
        _ , _ = spec.forward_draft_or_large_model(in_features_large=h)


def persist_kv_cache(
    spec: EarlyExitLlamaForCausalLM,
    shallow_past: Optional[Tuple],
    deep_past: Optional[Tuple],
) -> None:
    """Write concatenated ``shallow_past`` and ``deep_past`` back to ``spec``.

    This keeps ``spec.past_key_values`` and the underlying ``spec.model`` cache
    in sync with externally managed KV tuples.
    """
    if shallow_past is None and deep_past is None:
        return
    combined = tuple(list(shallow_past or []) + list(deep_past or []))
    for obj in (spec, getattr(spec, "model", None)):
        if obj is None:
            continue
        try:
            obj.past_key_values = combined
        except Exception:
            pass


@torch.no_grad()
def drafter_hidden_no_cache(spec: EarlyExitLlamaForCausalLM, ids_last: torch.Tensor) -> torch.Tensor:
    slots = _kv_snapshot(spec)
    try:
        with adapter_guard(spec, "draft"):
            out = spec.forward_draft_or_large_model(in_tokens_small=ids_last, use_cache=False)
        return out
    except TypeError:
        pass
    except Exception:
        pass
    finally:
        _kv_restore(slots)

    toggled = []
    def _toggle(obj):
        if obj is not None and hasattr(obj, "config") and hasattr(obj.config, "use_cache"):
            toggled.append((obj, obj.config.use_cache))
            obj.config.use_cache = False
    _toggle(spec)
    _toggle(getattr(spec, "model", None))
    try:
        with adapter_guard(spec, "draft"):
            out = spec.forward_draft_or_large_model(in_tokens_small=ids_last)
    finally:
        for obj, prev in toggled:
            try:
                obj.config.use_cache = prev
            except Exception:
                pass
        _kv_restore(slots)
    return out
