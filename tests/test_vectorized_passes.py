import os
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import LlamaConfig
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.modeling import run_shallow_until_k, run_deep_from_k, exit_logits_from_hidden_k, adapter_guard

# --- helpers to build tiny model ---
class IdentityLayer(torch.nn.Module):
    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=True):
        if use_cache:
            B, S, H = x.shape
            past_len = 0
            if past_key_value is not None and past_key_value[0] is not None:
                past_len = past_key_value[0].shape[2]
            k = torch.zeros(B, 1, past_len + S, 1, device=x.device)
            v = torch.zeros(B, 1, past_len + S, 1, device=x.device)
            return x, (k, v)
        return x, past_key_value

class ToyModel(torch.nn.Module):
    def __init__(self, cfg, num_layers):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = torch.nn.ModuleList([IdentityLayer() for _ in range(num_layers)])
        self.norm = torch.nn.Identity()
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Mimic HF attention mask preparation for tests.

        The real implementation produces a 4D causal mask.  Here we just
        return ``None`` if no mask is provided, or a simple lower-triangular
        mask when ``attention_mask`` is given so that the vectorised helpers
        exercise mask logic without relying on transformers internals.
        """
        if attention_mask is None:
            return None
        B, S = input_shape
        L = S + past_key_values_length
        mask = torch.ones(B, 1, L, L, device=attention_mask.device)
        mask = torch.tril(mask)
        return mask[:, :, -S:, :]

def build_model():
    cfg = LlamaConfig(hidden_size=4, intermediate_size=8, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, vocab_size=10)
    model = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=1)
    model.model = ToyModel(cfg, num_layers=2)
    model.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model.exit_proj = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model.exit_proj.weight.data.copy_(model.lm_head.weight.data)
    model.head_model = model.lm_head
    model.past_key_values = None
    model.early_layer = 1
    return model

# --- reference stepwise implementations for tests ---

def _shallow_stepwise(model, input_ids, past_key_values=None, use_cache=True):
    k = model.early_layer
    lm = model.model
    B, T = input_ids.shape
    hidden = lm.embed_tokens(input_ids)
    if past_key_values is None:
        past_key_values = [None] * k
    layer_pkv = list(past_key_values)
    for layer_idx in range(k):
        block = lm.layers[layer_idx]
        pkv = layer_pkv[layer_idx]
        outputs_t = []
        cur_pkv = pkv
        for t in range(T):
            hs = hidden[:, t:t+1, :]
            past_len = cur_pkv[0].shape[2] if cur_pkv is not None else 0
            pos = torch.full((B,1), past_len, device=hs.device, dtype=torch.long)
            attn_mask = None
            out = block(hs, attention_mask=attn_mask, position_ids=pos, past_key_value=cur_pkv, output_attentions=False, use_cache=use_cache)
            hs = out[0]
            if use_cache:
                cur_pkv = out[1]
            outputs_t.append(hs)
        hidden = torch.cat(outputs_t, dim=1)
        if use_cache:
            layer_pkv[layer_idx] = cur_pkv
    return hidden, tuple(layer_pkv) if use_cache else None

def _deep_stepwise(model, hidden_k, past_key_values=None, use_cache=True):
    k = model.early_layer
    lm = model.model
    deep_layers = lm.layers[k:]
    B, T, _ = hidden_k.shape
    pkv = list(past_key_values) if past_key_values is not None else [None]*len(deep_layers)
    logits_chunks = []
    for t in range(T):
        hs = hidden_k[:, t:t+1, :]
        past_len = pkv[0][0].shape[2] if pkv[0] is not None else 0
        pos = torch.full((B,1), past_len, device=hs.device, dtype=torch.long)
        attn_mask = None
        new_past = []
        for i, block in enumerate(deep_layers):
            pkv_i = pkv[i]
            out = block(hs, attention_mask=attn_mask, position_ids=pos, past_key_value=pkv_i, output_attentions=False, use_cache=use_cache)
            hs = out[0]
            if use_cache:
                new_past.append(out[1])
        normed = lm.norm(hs)
        logits = model.lm_head(normed)
        logits_chunks.append(logits)
        if use_cache:
            pkv = new_past
    return torch.cat(logits_chunks, dim=1), tuple(pkv) if use_cache else None


def _generate_stepwise(model, max_new_tokens=5, draft_k=2):
    shallow_past = None
    deep_past = None
    last_token = torch.tensor([[1]])
    generated = []
    while len(generated) < max_new_tokens:
        tmp_shallow = shallow_past
        drafts = []
        prev = last_token
        for _ in range(draft_k):
            h, tmp_shallow = _shallow_stepwise(model, prev, past_key_values=tmp_shallow, use_cache=True)
            logits = exit_logits_from_hidden_k(model, h)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            drafts.append(nxt.squeeze(1))
            prev = nxt
        prop_seq = torch.stack(drafts, dim=1)
        h_block, shallow_full = _shallow_stepwise(model, prop_seq, past_key_values=shallow_past, use_cache=True)
        logits_deep, deep_full = _deep_stepwise(model, h_block, past_key_values=deep_past, use_cache=True)
        deep_argmax = logits_deep.argmax(dim=-1)
        matches0 = deep_argmax.eq(prop_seq)
        if matches0.all():
            accept_len = prop_seq.size(1)
        else:
            accept_len = int((~matches0).float().argmax().item())
        if accept_len > 0:
            accepted = prop_seq[:, :accept_len]
            generated.extend(accepted[0].tolist())
            past_len_s = 0 if shallow_past is None or shallow_past[0] is None else shallow_past[0][0].shape[2]
            new_shallow = []
            for (k, v) in shallow_full:
                new_shallow.append((k[:, :, : past_len_s + accept_len, :], v[:, :, : past_len_s + accept_len, :]))
            shallow_past = tuple(new_shallow)
            past_len_d = 0 if deep_past is None or deep_past[0] is None else deep_past[0][0].shape[2]
            new_deep = []
            for (k, v) in deep_full:
                new_deep.append((k[:, :, : past_len_d + accept_len, :], v[:, :, : past_len_d + accept_len, :]))
            deep_past = tuple(new_deep)
            last_token = accepted[:, -1:].clone()
        else:
            v1 = deep_argmax[:, 0:1]
            generated.extend(v1[0].tolist())
            last_token = v1
            h_fix, shallow_past = _shallow_stepwise(model, v1, past_key_values=shallow_past, use_cache=True)
            _, deep_past = _deep_stepwise(model, h_fix, past_key_values=deep_past, use_cache=True)
    return generated

# --- tests ---

def test_logit_equivalence():
    torch.manual_seed(0)
    model = build_model()
    prefix = torch.tensor([[1,2]])
    draft = torch.tensor([[3,4]])

    # stepwise baseline
    with adapter_guard(model, "draft"):
        _, shallow_past = _shallow_stepwise(model, prefix, past_key_values=None, use_cache=True)
        hidden_seq, _ = _shallow_stepwise(model, draft, past_key_values=shallow_past, use_cache=True)
    with adapter_guard(model, "verify"):
        logits_seq, _ = _deep_stepwise(model, hidden_seq, past_key_values=None, use_cache=False)

    # reset
    model.past_key_values = None

    # vectorised path
    with adapter_guard(model, "draft"):
        _, shallow_past_v = run_shallow_until_k(model, input_ids=prefix, past_key_values=None, use_cache=True)
        hidden_vec, _ = run_shallow_until_k(model, input_ids=draft, past_key_values=shallow_past_v, use_cache=True)
    with adapter_guard(model, "verify"):
        logits_vec, _ = run_deep_from_k(model, hidden_k=hidden_vec, past_key_values=None, use_cache=False)

    assert torch.allclose(logits_vec, logits_seq, atol=1e-5, rtol=1e-4)


def test_kv_growth():
    torch.manual_seed(0)
    model = build_model()
    prefix = torch.tensor([[1,2]])
    draft = torch.tensor([[3,4,5]])
    with adapter_guard(model, "draft"):
        _, past0 = run_shallow_until_k(model, input_ids=prefix, past_key_values=None, use_cache=True)
        kv_len0 = past0[0][0].shape[2]
        _, past1 = run_shallow_until_k(model, input_ids=draft, past_key_values=past0, use_cache=True)
        kv_len1 = past1[0][0].shape[2]
    assert kv_len1 - kv_len0 == draft.shape[1]
    with adapter_guard(model, "verify"):
        logits, deep_past = run_deep_from_k(model, hidden_k=torch.randn(1, draft.shape[1], model.config.hidden_size), past_key_values=None, use_cache=True)
    kv_len_deep = deep_past[0][0].shape[2]
    assert kv_len_deep == draft.shape[1]


def test_generation_equivalence():
    torch.manual_seed(0)
    model_step = build_model()
    torch.manual_seed(0)
    model_vec = build_model()
    ref = _generate_stepwise(model_step, max_new_tokens=5, draft_k=2)
    from training.spec_decode import generate_with_dvi_spec
    enc = {"input_ids": torch.tensor([[1]])}
    out_vec, _ = generate_with_dvi_spec(model_vec, None, enc=enc, max_new_tokens=5, draft_k=2, greedy=True, temperature=0.0)
    assert out_vec[0].tolist() == ref
