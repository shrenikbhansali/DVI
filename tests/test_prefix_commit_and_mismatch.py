import os, sys, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import LlamaConfig
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.rollout import rollout_collect_k_spec
from training.buffer import ReplayBuffer


class DummyTok:
    def __call__(self, s, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


def build_toy_controlled(vsz=32, h=16, L=4, ksplit=2):
    cfg = LlamaConfig(
        hidden_size=h,
        intermediate_size=2 * h,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vsz,
    )
    m = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=ksplit)
    m.lm_head = torch.nn.Linear(h, vsz, bias=False)
    m.exit_proj = torch.nn.Linear(h, vsz, bias=False)
    with torch.no_grad():
        m.exit_proj.weight.copy_(m.lm_head.weight)
        m.exit_proj.weight[0].add_(0.1)
    return m


def test_prefix_and_mismatch_path_runs_and_logs():
    m = build_toy_controlled()
    if not hasattr(m.model, "_prepare_decoder_attention_mask"):
        import pytest
        pytest.skip("missing attention mask helper")

    tok = DummyTok()
    buf = ReplayBuffer(capacity=64, device=torch.device("cpu"))

    # Under token-quota semantics, `steps` is the minimum number of token records
    # to collect. With k=2 and B=1, steps=4 corresponds to two speculative blocks.
    n = rollout_collect_k_spec(m, tok, "x", buf, steps=4, k=2, greedy=True)

    # We return the number of token records appended; equals steps when steps is
    # a multiple of k × batch size.
    assert n == 4

    rewards = buf._reward_buf[: len(buf)]
    # Make sure at least one drafted position was rejected by the verifier.
    assert (rewards == 0).sum().item() >= 1
