import torch
from typing import List, Optional, Tuple, NamedTuple
import logging
from transformers.models.llama import LlamaForCausalLM


class SpecStep(NamedTuple):
    hidden: torch.Tensor  # (B, 1, H) on CPU float32
    logits: torch.Tensor  # (B, |V|) on CPU float32
    accept: torch.Tensor  # (B, 1) uint8
    token: torch.Tensor   # (B, 1) int64


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

    # ------------------------------------------------------------------
    @torch.no_grad()
    def spec_decode_step(
        self,
        in_tokens: torch.LongTensor,
        *,
        temperature: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> SpecStep:
        """Run one speculative decoding step.

        Parameters
        ----------
        in_tokens : Tensor
            Input token ids of shape (B, 1).
        temperature : float, optional
            Sampling temperature; 0 for greedy.
        position_ids : Tensor, optional
            Positional ids for the token.
        global_step : int, optional
            Step index for debug logging.

        Returns
        -------
        SpecStep
            Micro-step output detached to CPU.
        """

        logger = logging.getLogger("debug_accept")

        # --- Draft path -------------------------------------------------
        hidden = self.forward_draft_or_large_model(
            in_tokens_small=in_tokens, position_ids=position_ids
        )
        hidden_last = hidden[:, -1:, :]
        draft_logits = self.exit_proj(hidden_last).float()

        if temperature > 0:
            probs = torch.softmax(draft_logits / temperature, dim=-1)
            token = torch.multinomial(probs.view(probs.size(0), -1), 1)
        else:
            token = torch.argmax(draft_logits, dim=-1, keepdim=True)

        # --- Verifier path --------------------------------------------
        _, final_hidden = self.forward_draft_or_large_model(
            in_features_large=hidden_last, position_ids=position_ids
        )
        final_logits = self.head_model(final_hidden).float()

        accept = (
            final_logits.argmax(dim=-1, keepdim=True) == token
        ).to(torch.uint8)

        prob = (
            torch.softmax(final_logits, dim=-1)
            .gather(-1, token)
            .squeeze(-1)
        )

        for idx in range(accept.shape[0]):
            logger.debug(
                {
                    "step": global_step,
                    "accept": int(accept[idx].item()),
                    "conf": float(prob[idx].item()),
                }
            )

        return SpecStep(
            hidden_last.detach().to(torch.float32).cpu().clone(),
            draft_logits.squeeze(1).detach().to(torch.float32).cpu().clone(),
            accept.detach().cpu().clone(),
            token.detach().to(torch.int64).cpu().clone(),
        )
