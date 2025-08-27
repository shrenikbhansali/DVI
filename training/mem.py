import os
import gc
from typing import Any, Set
import torch

__all__ = ["deep_kv_purge"]

_TRACE = bool(os.getenv("DVI_TIMING_TRACE"))

# Attributes that may hold KV-cache style tensors
_KV_ATTRS = {
    "past_key_values",
    "_past_key_values",
    "present",
    "presents",
    "present_key_value",
    "past_key_value",
    "cache_k",
    "cache_v",
    "k_cache",
    "v_cache",
    "key_cache",
    "value_cache",
}


def deep_kv_purge(obj: Any, tag: str = "") -> int:
    """Recursively clear common KV-cache attributes from ``obj``.

    Returns number of attributes cleared. Idempotent and safe on arbitrary
    objects; tensors and non-modules are ignored. Designed to be cheap enough to
    call frequently.
    """
    seen: Set[int] = set()
    cleared = []

    def _purge(x: Any):
        if x is None:
            return
        if isinstance(x, (int, float, str, bytes)):
            return
        xid = id(x)
        if xid in seen:
            return
        seen.add(xid)

        for name in list(getattr(x, "__dict__", {}).keys()):
            lname = name.lower()
            if lname in _KV_ATTRS or any(k in lname for k in ("past_key", "_past_key", "cache", "present")):
                try:
                    if getattr(x, name) is not None:
                        cleared.append(f"{type(x).__name__}.{name}")
                    setattr(x, name, None)
                except Exception:
                    pass

        if isinstance(x, torch.nn.Module):
            for child in x.children():
                _purge(child)

        # Explore common container attributes on wrappers
        for attr in ("model", "base_model", "module", "inner_model", "model_wrapped"):
            if hasattr(x, attr):
                _purge(getattr(x, attr))

        if isinstance(x, (list, tuple, set)):
            for it in x:
                _purge(it)
        elif isinstance(x, dict):
            for it in x.values():
                _purge(it)

    _purge(obj)

    if _TRACE:
        msg = ", ".join(cleared) if cleared else "none"
        print(f"[kv-purge]{(' '+tag) if tag else ''} -> {msg}", flush=True)

    gc.collect()
    return len(cleared)
