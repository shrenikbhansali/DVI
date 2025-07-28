import ast
import logging
import sys
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from transformers import LlamaConfig

class IdentityLayer(torch.nn.Module):
    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=True):
        return x, past_key_value

class ToyModel(torch.nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([IdentityLayer() for _ in range(num_layers)])
        self.norm = torch.nn.Identity()
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return None

def build_model(dtype=torch.float32):
    cfg = LlamaConfig(hidden_size=4, intermediate_size=8, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, vocab_size=10)
    model = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=1)
    model.model = ToyModel(cfg, num_layers=2)
    model.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)
    model.exit_proj = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)
    model.exit_proj.weight.data.copy_(model.lm_head.weight.data)
    model.head_model = model.lm_head
    if dtype == torch.float16:
        model.half()
    model.past_key_values = None
    return model, cfg

def test_deterministic_accept():
    model, cfg = build_model()
    step = model.spec_decode_step(torch.tensor([[1]]), temperature=0.0, global_step=0)
    assert step.accept.eq(torch.ones_like(step.accept)).all()

def test_confidence_sanity(caplog):
    model, cfg = build_model()
    toks = torch.tensor([[1],[2],[3]])
    with caplog.at_level(logging.DEBUG, logger="debug_accept"):
        step = model.spec_decode_step(toks, temperature=0.0, global_step=0)
    logs = [ast.literal_eval(r.message) for r in caplog.records]
    # recompute verifier logits from scratch for comparison
    model.past_key_values = None
    h = model.forward_draft_or_large_model(in_tokens_small=toks)
    _, final_h = model.forward_draft_or_large_model(in_features_large=h)
    final_logits = model.head_model(final_h)
    probs = torch.softmax(final_logits, dim=-1).gather(-1, step.token).squeeze(-1)
    for rec, p in zip(logs, probs):
        assert 0.0 <= rec["conf"] <= 1.0
        assert abs(rec["conf"] - p.item()) < 1e-5

def test_device_dtype_safety():
    model, cfg = build_model(dtype=torch.float16)
    step = model.spec_decode_step(torch.tensor([[1]]), temperature=0.0, global_step=0)
    assert step.hidden.device.type == "cpu" and step.hidden.dtype == torch.float32
    assert step.logits.device.type == "cpu" and step.logits.dtype == torch.float32
    assert step.token.device.type == "cpu" and step.token.dtype == torch.int64
    assert step.accept.device.type == "cpu" and step.accept.dtype == torch.uint8
