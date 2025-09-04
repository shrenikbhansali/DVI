import torch
import torch
import torch.nn as nn
import contextlib
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import training.objectives as obj_mod
from training.objectives import policy_gradient_terms, one_policy_step


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.exit_proj = nn.Linear(2, 2, bias=False)
def test_pg_loss_sign_and_kl_scaling(monkeypatch):
    model = ToyModel()

    def fake_run_shallow_until_k(model, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        hidden = torch.nn.functional.one_hot(input_ids.squeeze(1), num_classes=2).float()
        return hidden.unsqueeze(1), []

    def fake_exit_logits_from_hidden_k(model, h_k):
        return model.exit_proj(h_k)

    monkeypatch.setattr(obj_mod, "run_shallow_until_k", fake_run_shallow_until_k)
    monkeypatch.setattr(obj_mod, "exit_logits_from_hidden_k", fake_exit_logits_from_hidden_k)
    monkeypatch.setattr(obj_mod, "adapter_guard", lambda *a, **k: contextlib.nullcontext())

    batch = {
        "state": torch.tensor([0, 1]),
        "token": torch.tensor([0, 1]),
        "accepted": torch.tensor([1.0, 0.0]),
        "vlogits": torch.zeros(2, 2),
    }
    pg1, kl, _ = policy_gradient_terms(model, batch, baseline_state={"b": 0.0}, baseline_ema=0.0)
    batch_flip = {
        "state": torch.tensor([0, 1]),
        "token": torch.tensor([0, 1]),
        "accepted": torch.tensor([0.0, 1.0]),
        "vlogits": torch.zeros(2, 2),
    }
    pg2, kl2, _ = policy_gradient_terms(model, batch_flip, baseline_state={"b": 0.0}, baseline_ema=0.0)
    assert torch.allclose(pg1, -pg2, atol=1e-6)

    rl_weight = 1.0
    beta1, beta2 = 0.1, 0.2
    loss1 = rl_weight * pg1 + beta1 * kl
    loss2 = rl_weight * pg1 + beta2 * kl
    assert torch.allclose(loss2 - rl_weight * pg1, (beta2 / beta1) * (loss1 - rl_weight * pg1), atol=1e-6)

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    w0 = model.exit_proj.weight.detach().clone()
    one_policy_step(
        model,
        opt,
        batch,
        rl_weight=1.0,
        beta=0.0,
        baseline_state={"b": 0.0},
        baseline_ema=0.0,
    )
    assert not torch.allclose(model.exit_proj.weight, w0)
