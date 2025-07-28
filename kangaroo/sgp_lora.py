import logging
from typing import List, Tuple

from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model
import torch.nn as nn

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

def inject_dual_lora(model: PreTrainedModel, exit_layer: int, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> PreTrainedModel:
    """Attach lora_S/lora_D adapters to different layer ranges."""
    num_layers = model.config.num_hidden_layers
    if exit_layer < 0 or exit_layer >= num_layers - 1:
        raise ValueError("exit_layer out of range")
    weight = next(model.parameters())
    base_cfg = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_s_cfg = LoraConfig(**base_cfg, layers_to_transform=list(range(exit_layer + 1)))
    kwargs = {}
    if "device_map" in get_peft_model.__code__.co_varnames:
        kwargs["device_map"] = {"": weight.device}
    model = get_peft_model(model, lora_s_cfg, adapter_name="lora_S", **kwargs)

    lora_d_cfg = LoraConfig(**base_cfg, layers_to_transform=list(range(exit_layer + 1, num_layers)))
    if hasattr(model, "add_adapter"):
        add_kwargs = {}
        if "device_map" in model.add_adapter.__code__.co_varnames:
            add_kwargs["device_map"] = {"": weight.device}
        model.add_adapter("lora_D", lora_d_cfg, **add_kwargs)

    if "device_map" not in get_peft_model.__code__.co_varnames:
        for _, mod in model.named_modules():
            if hasattr(mod, "lora_A") or hasattr(mod, "lora_B"):
                mod.to(weight.device)

    summary = {}
    for name, p in model.named_parameters():
        if ".lora_" not in name:
            continue
        adapter = "lora_S" if ".lora_S." in name else "lora_D"
        layer = _parse_layer(name)
        module = next((m for m in _TARGET_MODULES if f".{m}." in name), "?")
        key = (layer, module, adapter)
        summary[key] = summary.get(key, 0) + p.numel()

    logger.info("layer  module      adapter   #params")
    logger.info("-----  ----------  --------  -------")
    for (layer, module, adapter), count in sorted(summary.items()):
        logger.info(f"{layer:>5}  {module:<10}  {adapter:<8}  {count/1000:.0f}k")

    return model

def split_lora_params(model: PreTrainedModel) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    fast, slow = [], []
    all_names = []
    for name, p in model.named_parameters():
        if ".lora_" not in name:
            continue
        all_names.append(name)
        if ".lora_S." in name and p.requires_grad:
            fast.append((name, p))
        elif ".lora_D." in name:
            slow.append((name, p))
    if not fast or not slow:
        raise ValueError("split resulted in empty group")
    union = {n for n, _ in fast}.union({n for n, _ in slow})
    inter = {n for n, _ in fast}.intersection({n for n, _ in slow})
    assert len(inter) == 0, "fast and slow sets intersect"
    assert set(all_names) == union, "missing LoRA params in split"
    logger.info(f"split into fast={len(fast)} slow={len(slow)} params")
    return fast, slow

def enable_lora_grads(model: PreTrainedModel, adapter_prefix: str, flag: bool):
    count = 0
    for name, p in model.named_parameters():
        if f".{adapter_prefix}." in name:
            if p.requires_grad != flag:
                p.requires_grad = flag
                count += 1
    logger.info(f"set requires_grad={flag} for {count} parameters with prefix {adapter_prefix}")
