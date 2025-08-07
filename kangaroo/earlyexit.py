# kangaroo/earlyexit.py
import logging
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
from transformers.models.llama import LlamaForCausalLM


class SpecStep(NamedTuple):
    hidden: torch.Tensor   # (B, 1, H)  fp32  CPU
    logits: torch.Tensor   # (B, |V|)   fp32  CPU
    accept: torch.Tensor   # (B, 1)     uint8 CPU
    token:  torch.Tensor   # (B, 1)     int64 CPU


class EarlyExitLlamaForCausalLM(LlamaForCausalLM):
    """
    Llama wrapper that supports self‑speculative decoding:
        shallow 0..k‑1  → draft model
        deep    k..L‑1  → verifier model
    """

    # ------------------------------------------------------------------ #
    # INITIALISATION                                                     #
    # ------------------------------------------------------------------ #
    def __init__(self, config, EARLY_STOP_LAYER: int):
        super().__init__(config)

        self.early_exit_layer = EARLY_STOP_LAYER
        self.past_key_values   = None                        # filled lazily

        # ------------- draft head (independent, trainable) -------------
        # We *clone* lm_head weights so we start identical but the new
        # Parameter is trainable and not shared with the frozen lm_head.
        self.exit_proj = nn.Linear(config.hidden_size,
                                   config.vocab_size,
                                   bias=False)
        with torch.no_grad():
            self.exit_proj.weight.copy_(self.lm_head.weight)

        # The verifier continues to use the original full‑depth lm_head
        self.head_model = self.lm_head

    # ------------------------------------------------------------------ #
    # HELPER                                                             #
    # ------------------------------------------------------------------ #
    def _ensure_past_cache(self, n_layers: int):
        if self.past_key_values is None:
            self.past_key_values = [None] * n_layers

    # ------------------------------------------------------------------ #
    # FORWARD for either draft or verifier sub‑network                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_draft_or_large_model(
        self,
        *,
        in_tokens_small:  Optional[torch.LongTensor] = None,   # (B, T)
        in_features_large: Optional[torch.Tensor]     = None,   # (B, T, H)
        position_ids:      Optional[torch.Tensor]     = None,
    ):
        if (in_tokens_small is None) == (in_features_large is None):
            raise ValueError("Specify exactly one of in_tokens_small / "
                             "in_features_large")

        self._ensure_past_cache(len(self.model.layers))

        # -------- choose sub‑network ----------------------------------
        if in_tokens_small is not None:                       # draft path
            B, T = in_tokens_small.shape
            hidden = self.model.embed_tokens(in_tokens_small)
            layers = self.model.layers[: self.early_exit_layer]
            focus_offset = 0
        else:                                                 # verifier path
            B, T, _ = in_features_large.shape
            hidden = in_features_large
            layers = self.model.layers[self.early_exit_layer :]
            focus_offset = self.early_exit_layer

        device    = hidden.device
        pkv_first = self.past_key_values[focus_offset]
        past_len  = 0 if pkv_first is None else pkv_first[0].shape[2]

        # -------- positional ids & mask -------------------------------
        if position_ids is None:
            position_ids = (torch.arange(past_len, past_len + T, device=device)
                               .unsqueeze(0).expand(B, T))

        attn_mask = torch.ones((B, T + past_len),
                               dtype=torch.bool, device=device)
        attn_mask = self.model._prepare_decoder_attention_mask(
            attn_mask, (B, T), hidden, past_len)

        # -------- transformer layers ----------------------------------
        for i, layer in enumerate(layers):
            idx = i + focus_offset
            hidden, pkv = layer(
                hidden,
                attention_mask   = attn_mask,
                position_ids     = position_ids,
                past_key_value   = self.past_key_values[idx],
                output_attentions=False,
                use_cache        = True,
            )
            self.past_key_values[idx] = pkv

        # verifier also returns pre‑head hidden for layer‑norm’d logits
        if in_features_large is not None:
            norm = self.model.norm(hidden)
            if norm.dim() == 3 and norm.size(1) == 1:
                norm = norm.squeeze(1)
            return hidden, norm
        return hidden

    # ------------------------------------------------------------------ #
    # ONE SPECULATIVE MICRO‑STEP                                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def spec_decode_step(
        self,
        in_tokens: torch.LongTensor,          # (B,1)
        *,
        temperature:  float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        global_step:  int = 0,
    ) -> SpecStep:

        # ---------------- draft ---------------------------------------
        draft_hidden = self.forward_draft_or_large_model(
            in_tokens_small=in_tokens, position_ids=position_ids)   # (B,1,H)
        draft_logits = self.exit_proj(draft_hidden).squeeze(1)       # (B,V)

        if temperature > 0.0:
            probs = torch.softmax(draft_logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1)                      # (B,1)
        else:
            token = torch.argmax(draft_logits, dim=-1, keepdim=True) # (B,1)

        # ---------------- verifier ------------------------------------
        _, deep_hidden = self.forward_draft_or_large_model(
            in_features_large=draft_hidden, position_ids=position_ids)
        final_logits = self.head_model(deep_hidden).squeeze(1)       # (B,V)

        accept = (final_logits.argmax(dim=-1, keepdim=True) == token).to(torch.uint8)
        conf   = torch.softmax(final_logits, dim=-1).gather(-1, token)  # (B,1)

        logging.getLogger("debug_accept").debug(
            {"step": global_step,
             "accept": int(accept[0]),
             "conf": float(conf[0])})

        # ---------------- off‑load to CPU -----------------------------
        return SpecStep(
            draft_hidden.detach().cpu().clone().float(),  # (B,1,H)
            draft_logits.detach().cpu().clone().float(),  # (B,V)
            accept.detach().cpu().clone(),                # (B,1)
            token.detach().cpu().clone(),                 # (B,1)
        )
