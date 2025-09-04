import torch
import pytest
from torch import nn
from transformers import LlamaConfig

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.rollout import rollout_collect_k_spec
from training.buffer import ReplayBuffer


class DummyLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=True):
        if use_cache:
            B, T, H = x.shape
            pkv = (
                torch.zeros(B, 1, T, H, device=x.device, dtype=x.dtype),
                torch.zeros(B, 1, T, H, device=x.device, dtype=x.dtype),
            )
        else:
            pkv = past_key_value
        return x, pkv


class ToyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DummyLayer(cfg.hidden_size) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.Identity()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return None


class DummyTok:
    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


def make_model():
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=4, vocab_size=128)
    model = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=2)
    model.model = ToyModel(cfg)
    model.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model.exit_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model.exit_proj.weight = model.lm_head.weight
    model.head_model = model.lm_head
    model.past_key_values = None
    return model


def test_rollout_k_spec_accepts_all():
    model = make_model()
    tok = DummyTok()
    buf = ReplayBuffer(32, torch.device("cpu"))
    n = rollout_collect_k_spec(model, tok, "hi", buf, steps=4, k=2, greedy=True, temperature=0.0)
    # All k tokens from each step should be buffered (4 steps * 2 tokens)
    assert n == 8
    sample = buf.sample(8, accepted_only=False)
    # Every token was accepted so the reward sum equals the total count
    assert sample["reward"].sum().item() == pytest.approx(8.0)
    for p in model.parameters():
        assert p.grad is None
