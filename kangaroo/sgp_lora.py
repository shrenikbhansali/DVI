"""Utilities for attaching and routing dual LoRA adapters.

This module exposes helpers to attach two sets of LoRA adapters:

``lora_draft``
    Trainable adapters placed on the shallow draft stack ``[0, k)``.

``lora_verify``
    Optional adapters for the deep verifier stack ``[k, L)``.  These are
    disabled by default (rank=0) but the plumbing is kept for completeness.

``set_active_adapter(model, name)`` can be used to switch which adapter is
active during forward passes.  Passing ``"none"`` disables all adapters.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Literal

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

logger = logging.getLogger("dual_lora")
logging.basicConfig(level=logging.INFO)

_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _parse_layer(name: str) -> int:
    """Extract layer index from parameter/module name."""
    if ".layers." not in name:
        return -1
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return -1


def attach_dual_lora(
    model: PreTrainedModel,
    *,
    split_layer: int,
    rank_s: int = 8,
    rank_v: int = 0,
    alpha: int = 16,
    dropout: float = 0.05,
) -> PreTrainedModel:
    """Attach ``lora_draft``/``lora_verify`` adapters to different layer ranges."""

    num_layers = model.config.num_hidden_layers
    if split_layer < 0 or split_layer >= num_layers:
        raise ValueError("split_layer out of range")

    weight = next(model.parameters())
    base_cfg = dict(
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ----- draft adapter on layers [0, split_layer) -----
    draft_cfg = LoraConfig(r=rank_s, layers_to_transform=list(range(split_layer)), **base_cfg)
    kwargs = {}
    if "device_map" in get_peft_model.__code__.co_varnames:
        kwargs["device_map"] = {"": weight.device}
    model = get_peft_model(model, draft_cfg, adapter_name="lora_draft", **kwargs)
    try:
        model._hf_peft_config_loaded = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # ----- verifier adapter on layers [split_layer, L) (optional) -----
    if rank_v and rank_v > 0:
        verify_cfg = LoraConfig(r=rank_v, layers_to_transform=list(range(split_layer, num_layers)), **base_cfg)
        if hasattr(model, "add_adapter"):
            add_kwargs = {}
            if "device_map" in model.add_adapter.__code__.co_varnames:
                add_kwargs["device_map"] = {"": weight.device}
            model.add_adapter("lora_verify", verify_cfg, **add_kwargs)
            # Older PEFT: ensure new LoRA modules are moved to the base weight device
            if not hasattr(model.add_adapter, "__code__") or "device_map" not in model.add_adapter.__code__.co_varnames:
                for _, mod in model.named_modules():
                    if hasattr(mod, "lora_A") or hasattr(mod, "lora_B"):
                        try:
                            mod.to(weight.device)
                        except Exception:
                            pass

    if "device_map" not in get_peft_model.__code__.co_varnames:
        for _, mod in model.named_modules():
            if hasattr(mod, "lora_A") or hasattr(mod, "lora_B"):
                mod.to(weight.device)

    summary = {}
    for name, p in model.named_parameters():
        if ".lora_" not in name:
            continue
        adapter = "lora_draft" if ".lora_draft." in name else "lora_verify"
        layer = _parse_layer(name)
        module = next((m for m in _TARGET_MODULES if f".{m}." in name), "?")
        key = (layer, module, adapter)
        summary[key] = summary.get(key, 0) + p.numel()

    if summary:
        logger.info("layer  module      adapter       #params")
        logger.info("-----  ----------  ------------ -------")
        for (layer, module, adapter), count in sorted(summary.items()):
            logger.info(f"{layer:>5}  {module:<10}  {adapter:<12} {count/1000:.0f}k")

    return model


def set_active_adapter(model: PreTrainedModel, name: Literal["draft", "verify", "none"]) -> None:
    """Activate a specific adapter or disable all adapters (robust across PEFT versions)."""
    if not hasattr(model, "peft_config"):
        return
    peft_cfg = getattr(model, "peft_config", {}) or {}

    def _disable():
        # Prefer the official API when available
        if hasattr(model, "disable_adapter"):
            try:
                model.disable_adapter()
                return
            except Exception:
                pass
        # Fallback used by some versions
        if hasattr(model, "set_adapter"):
            try:
                model.set_adapter([])
            except Exception:
                pass

    if name == "none":
        _disable()
        return

    mapping = {"draft": "lora_draft", "verify": "lora_verify"}
    target = mapping.get(name)
    if target not in peft_cfg:
        # Requested adapter not attached → ensure adapters are off
        _disable()
        return

    if hasattr(model, "set_adapter"):
        model.set_adapter(target)
    else:
        # If we cannot select explicitly, best effort: disable to avoid accidental bleed-through
        _disable()


def enable_lora_grads(model: PreTrainedModel, adapter_prefix: str, flag: bool) -> None:
    """Enable or disable ``requires_grad`` for parameters of a given adapter."""

    count = 0
    for name, p in model.named_parameters():
        if f".{adapter_prefix}." in name:
            if p.requires_grad != flag:
                p.requires_grad = flag
                count += 1
    logger.info(
        f"set requires_grad={flag} for {count} parameters with prefix {adapter_prefix}"
    )


def split_lora_params(
    model: PreTrainedModel,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    """Split parameters into draft/verify lists for optimisation."""

    draft, verify = [], []
    all_names: List[str] = []
    for name, p in model.named_parameters():
        if ".lora_" not in name:
            continue
        all_names.append(name)
        if ".lora_draft." in name and p.requires_grad:
            draft.append((name, p))
        elif ".lora_verify." in name:
            verify.append((name, p))

    if not draft and not verify:
        raise ValueError("no LoRA parameters found")

    union = {n for n, _ in draft}.union({n for n, _ in verify})
    if set(all_names) != union:
        raise ValueError("missing LoRA params in split")

    logger.info(f"split into draft={len(draft)} verify={len(verify)} params")
    return draft, verify


# Backwards compatibility ----------------------------------------------------

# Older code/tests expect ``inject_dual_lora(model, exit_layer=..., rank=...)``.
def inject_dual_lora(model: PreTrainedModel, exit_layer: int, rank: int = 8, **kwargs):
    rank_v = kwargs.get("rank_v", 0)
    alpha = kwargs.get("alpha", 16)
    dropout = kwargs.get("dropout", 0.05)
    return attach_dual_lora(
        model,
        split_layer=exit_layer,
        rank_s=rank,
        rank_v=rank_v,
        alpha=alpha,
        dropout=dropout,
    )


__all__ = [
    "attach_dual_lora",
    "inject_dual_lora",
    "set_active_adapter",
    "enable_lora_grads",
    "split_lora_params",
]

