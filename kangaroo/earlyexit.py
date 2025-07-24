import torch
from typing import List, Optional, Tuple
from transformers.models.llama import LlamaForCausalLM


class EarlyExitLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, EARLY_STOP_LAYER):
        super().__init__(config)
        self.past_key_values = None
        self.early_exit_layer = EARLY_STOP_LAYER

    # ------- new helper --------------------------------------------------
    def _ensure_past_cache(self, n_layers: int):
        if self.past_key_values is None:
            self.past_key_values = [None] * n_layers
    # --------------------------------------------------------------------

    @torch.no_grad()
    def forward_draft_or_large_model(
        self, *, in_tokens_small=None, in_features_large=None, position_ids=None
    ):
        self._ensure_past_cache(len(self.model.layers))

        use_cache = True
        if (in_tokens_small is None) == (in_features_large is None):
            raise ValueError("specify either in_tokens_small or in_features_large")

        if in_tokens_small is not None:
            batch_size, seq_length = in_tokens_small.shape
            device = in_tokens_small.device
        else:
            batch_size, seq_length, _ = in_features_large.shape
            device = in_features_large.device

        focus_layer = 0 if in_tokens_small is not None else self.early_exit_layer
        pkv = self.past_key_values[focus_layer]
        past_kv_len = 0 if pkv is None else pkv[0].shape[2]
        seq_len_with_past = seq_length + past_kv_len

        if position_ids is None:
            position_ids = (
                torch.arange(past_kv_len, seq_len_with_past, device=device)
                .unsqueeze(0)
                .expand(batch_size, seq_length)
            )

        # embeddings / inputs_embeds only needed for draft path
        if in_tokens_small is not None:
            inputs_embeds = self.model.embed_tokens(in_tokens_small)
            hidden_states = inputs_embeds
            layers = self.model.layers[: self.early_exit_layer]
            mask_src = inputs_embeds
        else:
            hidden_states = in_features_large
            layers = self.model.layers[self.early_exit_layer :]
            mask_src = in_features_large  # just for device

        # attention mask
        attn_mask = torch.ones(
            (batch_size, seq_len_with_past),
            dtype=torch.bool,
            device=mask_src.device,
        )
        attn_mask = self.model._prepare_decoder_attention_mask(
            attn_mask, (batch_size, seq_length), mask_src, past_kv_len
        )

        # main loop
        for off, layer in enumerate(layers):
            layer_idx = off if in_tokens_small is not None else off + self.early_exit_layer
            pkv = self.past_key_values[layer_idx]

            out = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = out[0]
            self.past_key_values[layer_idx] = out[1]

        if in_features_large is not None:
            return hidden_states, self.model.norm(hidden_states)
        return hidden_states
