"""Model assembly helpers for DVI training."""
import copy
from typing import Optional

import torch
import torch.nn as nn

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import inject_dual_lora, enable_lora_grads

__all__ = ["prepare_dvi_trainable", "build_optimizer"]


def prepare_dvi_trainable(model_id: str, early_layer: int, dtype: torch.dtype = torch.float16) -> EarlyExitLlamaForCausalLM:
    model = EarlyExitLlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", EARLY_STOP_LAYER=early_layer
    )
    model = inject_dual_lora(model, exit_layer=early_layer, rank=8)
    for p in model.parameters():
        p.requires_grad = False
    enable_lora_grads(model, "lora_S", True)
    enable_lora_grads(model, "lora_D", False)

    with torch.no_grad():
        w = model.lm_head.weight.detach().clone().float()

    model.exit_proj = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=torch.float32)
    model.exit_proj.weight.data.copy_(w)
    model.exit_proj.weight.requires_grad = True

    try:
        base_norm = model.model.norm
        model.exit_pre_norm = copy.deepcopy(base_norm).to(w.device)
    except Exception:
        model.exit_pre_norm = nn.LayerNorm(w.shape[1], elementwise_affine=True, device=w.device)
    for p in model.exit_pre_norm.parameters():
        p.requires_grad = True

    model.exit_logit_scale = nn.Parameter(torch.tensor(1.0, device=w.device))

    model.lm_head.weight.requires_grad = False
    if hasattr(model, "head_model"):
        for p in model.head_model.parameters():
            p.requires_grad = False
    return model


def build_optimizer(model, lr_exit=2e-4, lr_lora=5e-5, wd_exit=1e-2, wd_lora=0.0):
    head_params = [model.exit_proj.weight]
    if hasattr(model, "exit_pre_norm") and model.exit_pre_norm is not None:
        head_params += list(model.exit_pre_norm.parameters())
    if hasattr(model, "exit_logit_scale"):
        head_params += [model.exit_logit_scale]

    groups = [
        {"params": head_params, "lr": lr_exit, "weight_decay": wd_exit},
    ]
    lora_s = [p for n, p in model.named_parameters() if p.requires_grad and "lora_S" in n]
    if lora_s:
        groups.append({"params": lora_s, "lr": lr_lora, "weight_decay": wd_lora})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
