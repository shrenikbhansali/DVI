"""Model assembly helpers for DVI training."""
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import inject_dual_lora, enable_lora_grads

__all__ = [
    "prepare_dvi_trainable",
    "build_optimizer",
    "run_shallow_until_k",
    "run_deep_from_k",
    "exit_logits_from_hidden_k",
]


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


def _decoder_layers(model):
    """Return transformer decoder layers list for HF Llama-style models."""
    return model.model.layers


def run_shallow_until_k(
    model,
    *,
    input_ids: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: bool = True,
):
    """Run embedding + layers [:k] to obtain hidden at split and updated KVs."""
    k = getattr(model, "early_layer")
    assert isinstance(k, int) and k > 0, "model.early_layer must be set"

    layers = _decoder_layers(model)[:k]
    B, T = input_ids.shape
    device = input_ids.device

    hidden_states = model.model.embed_tokens(input_ids)

    past_len = 0
    if past_key_values and past_key_values[0] is not None:
        past_len = past_key_values[0][0].shape[2]

    if attention_mask is None:
        attention_mask = torch.ones((B, past_len + T), dtype=torch.bool, device=device)
    attn_mask = model.model._prepare_decoder_attention_mask(
        attention_mask, (B, T), hidden_states, past_len
    )
    position_ids = (
        torch.arange(past_len, past_len + T, device=device).unsqueeze(0).expand(B, T)
    )

    if past_key_values is None:
        past_key_values = tuple([None] * k)

    new_past = []
    for i, block in enumerate(layers):
        pkv = past_key_values[i] if i < len(past_key_values) else None
        out = block(
            hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=pkv,
            output_attentions=False,
            use_cache=use_cache,
        )
        hidden_states = out[0]
        if use_cache:
            new_past.append(out[1])

    return hidden_states, tuple(new_past) if use_cache else None


def run_deep_from_k(
    model,
    *,
    hidden_k: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
    use_cache: bool = True,
):
    """Run only layers [k:] given hidden states at split layer."""
    k = getattr(model, "early_layer")
    layers = _decoder_layers(model)[k:]

    B, T, _ = hidden_k.shape
    device = hidden_k.device

    past_len = 0
    if past_key_values and past_key_values[0] is not None:
        past_len = past_key_values[0][0].shape[2]

    attention_mask = torch.ones((B, past_len + T), dtype=torch.bool, device=device)
    attn_mask = model.model._prepare_decoder_attention_mask(
        attention_mask, (B, T), hidden_k, past_len
    )
    position_ids = (
        torch.arange(past_len, past_len + T, device=device).unsqueeze(0).expand(B, T)
    )

    if past_key_values is None:
        past_key_values = tuple([None] * len(layers))

    new_past = []
    hidden_states = hidden_k
    for idx, block in enumerate(layers):
        pkv = past_key_values[idx] if idx < len(past_key_values) else None
        out = block(
            hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=pkv,
            output_attentions=False,
            use_cache=use_cache,
        )
        hidden_states = out[0]
        if use_cache:
            new_past.append(out[1])

    normed = model.model.norm(hidden_states)
    logits = model.lm_head(normed)
    return logits, tuple(new_past) if use_cache else None


def exit_logits_from_hidden_k(model, hidden_k: torch.Tensor) -> torch.Tensor:
    """Project hidden states at split layer to draft logits."""
    h = hidden_k
    if hasattr(model, "exit_pre_norm") and model.exit_pre_norm is not None:
        h = model.exit_pre_norm(h)
    logits = model.exit_proj(h)
    if hasattr(model, "exit_logit_scale"):
        logits = model.exit_logit_scale * logits
    return logits
