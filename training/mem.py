import os
import torch.nn as nn

_TRACE = bool(int(os.environ.get("DVI_TIMING_TRACE", "0")))


def timing_trace(msg: str) -> None:
    if _TRACE:
        print(f"[trace] {msg}", flush=True)


def deep_kv_purge(model) -> int:
    """Recursively clear all known KV caches from `model`.

    Returns the number of attributes cleared. Safe to call repeatedly.
    """
    attrs = [
        "past_key_values",
        "_past_key_values",
        "past_key_value",
        "_past_key_value",
        "present",
        "present_key_value",
        "cache_k",
        "cache_v",
    ]
    cleared = 0
    if model is None:
        return cleared

    objs = [model]
    for path in (
        "model",
        "base_model",
        "model.model",
        "base_model.model",
        "base_model.model.model",
        "model.base_model",
    ):
        cur = model
        ok = True
        for p in path.split('.'):
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok and cur not in objs:
            objs.append(cur)

    for m in list(objs):
        if isinstance(m, nn.Module):
            for sub in m.modules():
                if sub not in objs:
                    objs.append(sub)

    for obj in objs:
        for attr in attrs:
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                except Exception:
                    val = None
                if val is not None:
                    cleared += 1
                    try:
                        setattr(obj, attr, None)
                    except Exception:
                        pass
    if _TRACE:
        timing_trace(f"deep_kv_purge cleared {cleared} attrs")
    return cleared
