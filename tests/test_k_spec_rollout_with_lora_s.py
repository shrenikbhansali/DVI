import os, sys, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import LlamaConfig
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import attach_dual_lora
from training.rollout import rollout_collect_k_spec
from training.buffer import ReplayBuffer


class DummyTok:
    def __call__(self, s, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


def build_toy_tied(k=2, L=3, vsz=16, h=8):
    cfg = LlamaConfig(
        hidden_size=h,
        intermediate_size=2 * h,
        num_hidden_layers=L,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=vsz,
    )
    m = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=k)
    m.lm_head = torch.nn.Linear(h, vsz, bias=False)
    m.exit_proj = torch.nn.Linear(h, vsz, bias=False)
    with torch.no_grad():
        m.exit_proj.weight.copy_(m.lm_head.weight)
    m = attach_dual_lora(m, split_layer=k, rank_s=4, rank_v=0)
    return m


def test_k_spec_all_accepts_and_no_grad_on_vlogits():
    m = build_toy_tied()
    if not hasattr(m.model, "_prepare_decoder_attention_mask"):
        import pytest

        pytest.skip("missing attention mask helper")
    tok = DummyTok()
    buf = ReplayBuffer(capacity=64, device=torch.device("cpu"))
    n = rollout_collect_k_spec(
        m,
        tok,
        "x",
        buf,
        steps=3,
        k=2,
        greedy=True,
        temperature=0.0,
        spec_adaptive=False,
        eta=0.0,
    )
    assert n == 6
    assert len(buf) == 6
    assert buf.accepted_count() == 6
    for i in range(len(buf)):
        assert not buf._vlogits_buf[i].requires_grad
