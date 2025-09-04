import torch
import torch.nn as nn
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from training.objectives import policy_gradient_terms


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.exit_proj = nn.Linear(2, 2, bias=False)


def test_pg_loss_sign_and_kl_scaling():
    model = ToyModel()
    batch = {
        "hidden": torch.zeros(2, 2),
        "token": torch.tensor([0, 1]),
        "accepted": torch.tensor([1.0, 0.0]),
        "vlogits": torch.zeros(2, 2),
    }
    pg1, kl, _ = policy_gradient_terms(model, batch, baseline_state={"b": 0.0}, baseline_ema=0.0)
    batch_flip = {
        "hidden": torch.zeros(2, 2),
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
