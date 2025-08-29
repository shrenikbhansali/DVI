"""W&B logging helpers."""
import os
from typing import Dict, Optional

try:
    import wandb as _wandb
except Exception:  # pragma: no cover - wandb optional
    _wandb = None

WANDB_DEFAULT_ENTITY = "sbhansali8-georgia-institute-of-technology"
WANDB_DEFAULT_PROJECT = "DVI-Testing"

__all__ = [
    "init_wandb",
    "wandb_watch_model",
    "wandb_log",
    "wandb_summary_update",
    "finish_wandb",
]


def init_wandb(args) -> Optional[object]:
    if args.no_wandb:
        print("[wandb] disabled via --no-wandb", flush=True)
        return None
    if _wandb is None:
        print("[wandb] package not available; continuing without W&B logging.", flush=True)
        return None
    entity = args.wandb_entity or WANDB_DEFAULT_ENTITY
    project = args.wandb_project or WANDB_DEFAULT_PROJECT
    name = args.run_name or f"{os.path.basename(args.model_id)}-L{args.early_layer}-seed{args.seed}"
    try:
        run = _wandb.init(
            entity=entity,
            project=project,
            name=name,
            config=vars(args),
            settings=_wandb.Settings(start_method="thread"),
        )
        print(f"[wandb] initialized: entity={entity} project={project} name={name}", flush=True)
        return run
    except Exception as e:  # pragma: no cover - init may fail offline
        print(f"[wandb] init failed: {e}; continuing without W&B.", flush=True)
        return None


def wandb_watch_model(model, log_freq: int = 25) -> None:
    if _wandb is None or _wandb.run is None:
        return
    try:
        _wandb.watch(model, log="gradients", log_freq=log_freq, log_graph=False)
    except TypeError:
        try:
            _wandb.run.watch(model, log="gradients", log_freq=log_freq, log_graph=False)
        except Exception as e:
            print(f"[wandb] watch skipped: {e}", flush=True)
    except Exception as e:
        print(f"[wandb] watch skipped: {e}", flush=True)


def _to_python_scalar(v):
    """Best-effort convert tensors/ndarrays to Python scalars."""
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def wandb_log(d: Dict, step: Optional[int] = None) -> None:
    if _wandb is None or _wandb.run is None:
        return
    clean = {k: _to_python_scalar(v) for k, v in d.items()}
    try:
        _wandb.log(clean, step=step)
    except Exception:
        pass


def wandb_summary_update(d: Dict) -> None:
    if _wandb is None or _wandb.run is None:
        return
    try:
        for k, v in d.items():
            _wandb.run.summary[k] = _to_python_scalar(v)
    except Exception:
        pass


def finish_wandb() -> None:
    if _wandb is None or _wandb.run is None:
        return
    try:
        _wandb.finish()
    except Exception:
        pass
