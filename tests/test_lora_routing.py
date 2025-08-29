import os, sys, torch
import pytest
from transformers import LlamaConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import attach_dual_lora, set_active_adapter
from training.modeling import run_shallow_until_k, run_deep_from_k


class DummyTok:
    def __call__(self, s, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


def build_toy(k=2, L=4, vsz=32, h=16, r_s=4, r_v=0):
    cfg = LlamaConfig(
        hidden_size=h,
        intermediate_size=2 * h,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vsz,
    )
    m = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=k)
    m = attach_dual_lora(m, split_layer=k, rank_s=r_s, rank_v=r_v)
    return m


def test_param_partitioning_and_trainability():
    m = build_toy()
    names = [n for n, _ in m.named_parameters()]
    assert any("lora_draft" in n for n in names)
    assert not any("lora_verify" in n for n in names)  # rank_v=0
    assert all(p.requires_grad for n, p in m.named_parameters() if "lora_draft" in n)


def test_adapter_routing_noverify_grad():
    m = build_toy()
    if not hasattr(m.model, "_prepare_decoder_attention_mask"):
        pytest.skip("missing attention mask helper")
    tok = DummyTok()
    enc = tok("x", return_tensors="pt")
    set_active_adapter(m, "draft")
    h_k, spast = run_shallow_until_k(
        m, input_ids=enc["input_ids"], attention_mask=None, past_key_values=None, use_cache=True
    )
    set_active_adapter(m, "verify")
    with torch.no_grad():
        logits, dpast = run_deep_from_k(m, hidden_k=h_k, past_key_values=None, use_cache=True)
    assert not logits.requires_grad
