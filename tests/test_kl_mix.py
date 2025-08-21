import sys
import math
import torch
import pytest
from torch import nn
from transformers import LlamaConfig

if __package__ is None:
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from training.kl_mix import exp_decay_lambda, cosine_decay_lambda
from train_dvi import mixed_update, prepare_model_for_training
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
        return x, past_key_value


class ToyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DummyLayer(cfg.hidden_size) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.Identity()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return None


def make_model():
    cfg = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=4, vocab_size=128)
    model = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=2)
    model.model = ToyModel(cfg)
    model.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model.exit_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model.exit_proj.weight = model.lm_head.weight
    model.head_model = model.lm_head
    model.past_key_values = None
    model = prepare_model_for_training(model, early_layer=2)
    return model, cfg


def test_exp_decay_lambda():
    vals = [exp_decay_lambda(s, 0.8, 10, 0.1) for s in range(5)]
    assert vals[0] == pytest.approx(0.8)
    assert all(0.1 <= v <= 1.0 for v in vals)
    assert all(x >= y for x, y in zip(vals, vals[1:]))


def test_cosine_decay_lambda():
    vals = [cosine_decay_lambda(s, 100, 0.9, 0.2) for s in range(0, 101, 10)]
    assert all(0.2 <= v <= 0.9 for v in vals)


def test_mixed_update_no_verifier_grads():
    model, cfg = make_model()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = {
        'hidden': torch.randn(2, 1, cfg.hidden_size),
        'token': torch.zeros(2, 1, dtype=torch.long),
        'reward': torch.ones(2),
        'conf': torch.zeros(2),
        'vlogits': torch.randn(2, cfg.vocab_size),
    }
    loss, gnorm, rl_loss, kl = mixed_update(model, opt, batch, baseline=0.0, clip=1.0, kl_lambda=0.5)
    assert math.isfinite(loss) and math.isfinite(gnorm)
    for n, p in model.named_parameters():
        if n.startswith('lora_D') or n == 'lm_head.weight':
            assert torch.equal(params_before[n], p)
