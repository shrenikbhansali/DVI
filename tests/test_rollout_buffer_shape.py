import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import LlamaConfig
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.rollout import rollout_collect_k_spec
from training.buffer import ReplayBuffer
import pytest


class DummyTok:
    def __call__(self, s, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


def build_match_then_mismatch(vsz=32, h=16, L=4, ksplit=2):
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
        m.lm_head.weight[0, 0] = 1.0  # both agree on first token 0
        m.exit_proj.weight[0, 0] = 1.0
        m.lm_head.weight[1, 0] = 0.5
        m.exit_proj.weight[1, 0] = -0.5  # drafter prefers token1 at second pos
    return m


def test_rollout_keeps_prefix_plus_one_mismatch():
    m = build_match_then_mismatch()
    if not hasattr(m.model, "_prepare_decoder_attention_mask"):
        pytest.skip("missing attention mask helper")
    tok = DummyTok()
    buf = ReplayBuffer(capacity=16, device=torch.device("cpu"))
    n = rollout_collect_k_spec(m, tok, "x", buf, steps=1, k=2, greedy=True)
    assert n == 2
    rewards = buf._reward_buf[: len(buf)].tolist()
    assert rewards == [1.0, 0.0]
