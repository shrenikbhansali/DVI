import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import LlamaConfig
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.spec_decode import generate_with_dvi_spec
import training.spec_decode as sd
import pytest


class DummyTok:
    def __call__(self, s, padding=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


def build_simple_model(vsz=32, h=16, L=4, ksplit=2):
    cfg = LlamaConfig(
        hidden_size=h,
        intermediate_size=2 * h,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vsz,
    )
    m = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=ksplit)
    m.early_layer = ksplit
    m.lm_head = torch.nn.Linear(h, vsz, bias=False)
    m.exit_proj = torch.nn.Linear(h, vsz, bias=False)
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.exit_proj.weight.zero_()
    return m


def test_nan_in_verifier_counts(monkeypatch):
    m = build_simple_model()
    if not hasattr(m.model, "_prepare_decoder_attention_mask"):
        pytest.skip("missing attention mask helper")
    tok = DummyTok()

    def fake_run_deep_from_k(model, hidden_k, past_key_values=None, use_cache=True):
        B, T, H = hidden_k.shape
        V = model.config.vocab_size
        logits = torch.full((B, T, V), float('nan'), device=hidden_k.device)
        return logits, past_key_values

    monkeypatch.setattr(sd, 'run_deep_from_k', fake_run_deep_from_k)
    out, metrics = generate_with_dvi_spec(m, tok, prompts=["x"], max_new_tokens=1, draft_k=2, greedy=True)
    assert metrics.nan_events == 1
    assert metrics.committed == 1
