import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from transformers import LlamaConfig
from training.buffer import ReplayBuffer
from training.rollout import rollout_collect_k_spec
from evaluation.acceptance import eval_acceptance
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM


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
        class Batch:
            def __init__(self):
                self.input_ids = torch.tensor([[1, 2]], dtype=torch.long)

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                return self

            def get(self, key, default=None):
                return getattr(self, key, default)

            def __getitem__(self, key):
                return getattr(self, key)

        return Batch()


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


def test_rollout_restores_adapter():
    model = make_model()
    tok = DummyTok()
    buf = ReplayBuffer(32, torch.device("cpu"))
    before = getattr(model, "_dvi_active_adapter", None)
    rollout_collect_k_spec(
        model,
        tok,
        "hi",
        buf,
        steps=2,
        k=2,
        greedy=True,
        temperature=0.0,
        spec_adaptive=False,
        eta=0.0,
    )
    assert getattr(model, "_dvi_active_adapter", None) == before


def test_eval_acceptance_restores_adapter():
    model = make_model()
    tok = DummyTok()
    before = getattr(model, "_dvi_active_adapter", None)
    eval_acceptance(model, tok, ["hi"], rollout_len=1, steps_per_prompt=1, quiet=True, k_max=1)
    assert getattr(model, "_dvi_active_adapter", None) == before
