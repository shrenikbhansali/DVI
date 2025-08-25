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

# ---------------------------------------------------------------------------
# Small utility to safely access nested attrs and inner LLaMA blocks
# ---------------------------------------------------------------------------

def _get_by_path(obj, path: str):
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _llama_model(model):
    """
    Return the inner LlamaModel (has `.layers`, `.embed_tokens`, `.norm`,
    and `_prepare_decoder_attention_mask`) across HF/PEFT layouts.
    """
    candidates = (
        "model",
        "model.model",
        "base_model.model",
        "base_model.model.model",
        "model.base_model.model",
    )
    for path in candidates:
        m = _get_by_path(model, path)
        if m is not None and hasattr(m, "layers") and hasattr(m, "_prepare_decoder_attention_mask"):
            return m
    raise AttributeError("Could not locate inner decoder model with `.layers`.")


def _resolve_early_layer(model) -> int:
    """
    Split index k from:
      - model.early_layer
      - model.early_exit_layer (EarlyExitLlamaForCausalLM)
      - model.config.EARLY_STOP_LAYER / model.config.early_layer
    """
    for attr in ("early_layer", "early_exit_layer"):
        k = getattr(model, attr, None)
        if isinstance(k, int) and k > 0:
            return int(k)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("EARLY_STOP_LAYER", "early_layer"):
            v = getattr(cfg, key, None)
            if isinstance(v, int) and v > 0:
                return int(v)
    raise AttributeError(
        "early_layer split index not set. "
        "Set model.early_layer or model.early_exit_layer, or config.EARLY_STOP_LAYER / config.early_layer."
    )

# ---------------------------------------------------------------------------
# A Linear that auto-casts inputs to its own weight dtype (prevents dtype crashes)
# ---------------------------------------------------------------------------

class _SafeLinear(nn.Linear):
    def forward(self, input):  # type: ignore[override]
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return super().forward(input)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_dvi_trainable(model_id: str, early_layer: int, dtype: torch.dtype = torch.float16) -> EarlyExitLlamaForCausalLM:
    model = EarlyExitLlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", EARLY_STOP_LAYER=early_layer
    )
    model = inject_dual_lora(model, exit_layer=early_layer, rank=8)

    # ---- expose split index on top-level and common submodules/configs ----
    try:
        model.early_layer = int(early_layer)
    except Exception:
        pass
    for attr_path in ("base_model", "model", "base_model.model"):
        sub = _get_by_path(model, attr_path)
        if sub is not None:
            try:
                setattr(sub, "early_layer", int(early_layer))
            except Exception:
                pass
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("early_layer", "EARLY_STOP_LAYER"):
            try:
                setattr(cfg, key, int(early_layer))
            except Exception:
                pass
    # ----------------------------------------------------------------------

    # Freeze base; enable only LoRA-S + exit head stack
    for p in model.parameters():
        p.requires_grad = False
    enable_lora_grads(model, "lora_S", True)
    enable_lora_grads(model, "lora_D", False)

    # ----- HEADS: keep the draft head in float32 and auto-cast inputs -----
    with torch.no_grad():
        base_w = model.lm_head.weight.detach().clone().float()
    head_device = base_w.device

    model.exit_proj = _SafeLinear(
        base_w.shape[1], base_w.shape[0], bias=False, device=head_device, dtype=torch.float32
    )
    with torch.no_grad():
        model.exit_proj.weight.copy_(base_w)
    model.exit_proj.weight.requires_grad = True

    # exit_pre_norm mirrors model norm; keep it float32 to match exit head
    try:
        base_norm = _llama_model(model).norm
        model.exit_pre_norm = copy.deepcopy(base_norm).to(device=head_device, dtype=torch.float32)
    except Exception:
        model.exit_pre_norm = nn.LayerNorm(
            base_w.shape[1], elementwise_affine=True, device=head_device, dtype=torch.float32
        )
    for p in model.exit_pre_norm.parameters():
        p.requires_grad = True

    # scale param in float32 as well
    model.exit_logit_scale = nn.Parameter(torch.tensor(1.0, device=head_device, dtype=torch.float32))

    # Freeze verifier head
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
    """Return transformer decoder layers list for HF/PEFT LLaMA-style models."""
    return _llama_model(model).layers


def run_shallow_until_k(
    model,
    *,
    input_ids: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: bool = True,
):
    """Run embedding + layers [:k] to obtain hidden at split and updated KVs."""
    k = _resolve_early_layer(model)
    lm = _llama_model(model)

    B, T = input_ids.shape
    device = input_ids.device

    hidden_states = lm.embed_tokens(input_ids)

    past_len = 0
    if past_key_values and past_key_values[0] is not None:
        past_len = past_key_values[0][0].shape[2]

    if attention_mask is None:
        attention_mask = torch.ones((B, past_len + T), dtype=torch.bool, device=device)
    attn_mask = lm._prepare_decoder_attention_mask(
        attention_mask, (B, T), hidden_states, past_len
    )
    position_ids = torch.arange(past_len, past_len + T, device=device).unsqueeze(0).expand(B, T)

    if past_key_values is None:
        past_key_values = tuple([None] * k)

    new_past = []
    for i, block in enumerate(lm.layers[:k]):
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
    k = _resolve_early_layer(model)
    lm = _llama_model(model)

    B, T, _ = hidden_k.shape
    device = hidden_k.device

    past_len = 0
    if past_key_values and past_key_values[0] is not None:
        past_len = past_key_values[0][0].shape[2]

    attention_mask = torch.ones((B, past_len + T), dtype=torch.bool, device=device)
    attn_mask = lm._prepare_decoder_attention_mask(
        attention_mask, (B, T), hidden_k, past_len
    )
    position_ids = torch.arange(past_len, past_len + T, device=device).unsqueeze(0).expand(B, T)

    deep_layers = lm.layers[k:]
    if past_key_values is None:
        past_key_values = tuple([None] * len(deep_layers))

    new_past = []
    hidden_states = hidden_k
    for idx, block in enumerate(deep_layers):
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

    normed = lm.norm(hidden_states)
    logits = model.lm_head(normed)
    return logits, tuple(new_past) if use_cache else None


def exit_logits_from_hidden_k(model, hidden_k: torch.Tensor) -> torch.Tensor:
    """Project hidden states at split layer to draft logits (dtype-safe)."""
    h = hidden_k

    # Run pre-norm in its own dtype (float32 here)
    if hasattr(model, "exit_pre_norm") and model.exit_pre_norm is not None and hasattr(model.exit_pre_norm, "weight"):
        pre_dtype = model.exit_pre_norm.weight.dtype
        if h.dtype != pre_dtype:
            h = h.to(pre_dtype)
        h = model.exit_pre_norm(h)

    # _SafeLinear will upcast if needed, but we align explicitly
    proj_dtype = model.exit_proj.weight.dtype
    if h.dtype != proj_dtype:
        h = h.to(proj_dtype)

    logits = model.exit_proj(h)

    if hasattr(model, "exit_logit_scale"):
        scale = model.exit_logit_scale
        if scale.dtype != logits.dtype:
            scale = scale.to(logits.dtype)
        logits = scale * logits
    return logits
