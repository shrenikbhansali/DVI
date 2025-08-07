import sys
import math
import torch
import pytest
from torch import nn
from transformers import LlamaConfig

if __package__ is None:
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from train_dvi import prepare_model_for_training, reinforce_update, update_baseline
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
    model.exit_proj.weight = model.lm_head.weight  # share parameter
    model.head_model = model.lm_head
    model.past_key_values = None
    model = prepare_model_for_training(model, early_layer=2)
    return model, cfg


def test_trainable_params():
    model, _ = make_model()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert 'exit_proj.weight' in trainable
    assert all('lora_S' in n or n == 'exit_proj.weight' for n in trainable)


def test_rollout_updates_params():
    model, cfg = make_model()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    params_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    buf = ReplayBuffer(8, torch.device('cpu'))
    ids = torch.randint(0, cfg.vocab_size, (1, 1))
    attempts = 0
    while buf.accepted_count() == 0 and attempts < 20:
        step = model.spec_decode_step(ids[:, -1:])
        buf.append(step.hidden.clone(), int(step.token), float(step.accept), 0.0)
        ids = torch.cat([ids, step.token], dim=-1)
        attempts += 1
    if buf.accepted_count() == 0:
        buf.append(torch.zeros(1, 1, cfg.hidden_size), 0, 1.0, 0.0)
    batch = buf.sample(1, accepted_only=True)
    reinforce_update(model, opt, batch, baseline=0.0, clip=1.0)
    changed = any(not torch.equal(params_before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert changed


def test_baseline_formula():
    baseline = 0.5
    rewards = torch.tensor([1.0, 0.0, 1.0])
    updated = update_baseline(baseline, rewards)
    assert updated == pytest.approx(0.9 * baseline + 0.1 * rewards.mean().item())


def test_reinforce_update_finite():
    model, cfg = make_model()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    batch = {
        'hidden': torch.randn(3, 1, cfg.hidden_size),
        'token': torch.zeros(3, 1, dtype=torch.long),
        'reward': torch.ones(3),
        'conf': torch.zeros(3),
    }
    loss, grad = reinforce_update(model, opt, batch, baseline=0.0, clip=1.0)
    assert math.isfinite(loss)
    assert math.isfinite(grad)
