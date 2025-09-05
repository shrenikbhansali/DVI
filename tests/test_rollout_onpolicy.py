import torch
import types
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from training.buffer import ReplayBuffer
from training import rollout as rollout_mod
from training.align_telemetry import AlignTelemetryParams


class DummySpec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.past_key_values = None
        # dummy parameter so next(spec.parameters()) works
        self.dummy = torch.nn.Parameter(torch.zeros(1))


def make_tok():
    def _tok(prompt, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1]], dtype=torch.long)}
    return _tok


def test_rollout_labels_and_tokens(monkeypatch):
    spec = DummySpec()
    spec.draft_logits_list = [
        [0.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
        [0.0, 5.0, 1.0],
    ]
    spec.deep_logits_list = [
        [[0.0, 0.0, 0.0]],
        [[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
    ]

    def fake_run_shallow_until_k(spec, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        if spec.draft_logits_list:
            logits = spec.draft_logits_list.pop(0)
            hidden = torch.tensor(logits).view(1, 1, -1)
        else:
            hidden = torch.zeros(1, 1, 3)
        pkv = (torch.zeros(1, 1, 1, hidden.size(-1)), torch.zeros(1, 1, 1, hidden.size(-1)))
        return hidden, [pkv]

    def fake_exit_logits_from_hidden_k(spec, h_k):
        return h_k

    def fake_sample_from_logits(logits, greedy=False, temperature=1.0):
        return torch.argmax(logits, dim=-1)

    def fake_run_deep_from_k(spec, hidden_k, past_key_values=None, use_cache=True):
        if spec.deep_logits_list:
            logits = spec.deep_logits_list.pop(0)
            return torch.tensor(logits).unsqueeze(0), []
        V = hidden_k.size(-1)
        return torch.zeros(1, hidden_k.size(1), V), []

    monkeypatch.setattr(rollout_mod, "run_shallow_until_k", fake_run_shallow_until_k)
    monkeypatch.setattr(rollout_mod, "exit_logits_from_hidden_k", fake_exit_logits_from_hidden_k)
    monkeypatch.setattr(rollout_mod, "sample_from_logits", fake_sample_from_logits)
    monkeypatch.setattr(rollout_mod, "run_deep_from_k", fake_run_deep_from_k)
    monkeypatch.setattr(rollout_mod, "advance_kv_with_committed", lambda *args, **kwargs: None)

    tok = make_tok()
    buf = ReplayBuffer(16, torch.device("cpu"))
    rollout_mod.rollout_collect_k_spec(spec, tok, "hi", buf, steps=1, k=2, greedy=True)
    sample = buf.sample_on_policy(2)
    assert "state" in sample
    mapping = {int(tok): float(acc) for tok, acc in zip(sample["token"], sample["accepted"])}
    assert mapping[0] == 1.0 and mapping[1] == 0.0


def test_rollout_adaptive_stop(monkeypatch):
    spec = DummySpec()
    spec.draft_logits_list = [
        [0.0, 0.0],
        [10.0, 0.0],
        [0.0, 0.0],
    ]
    spec.deep_logits_list = [[[10.0, 0.0]]]

    def fake_run_shallow_until_k(spec, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        if spec.draft_logits_list:
            logits = spec.draft_logits_list.pop(0)
            hidden = torch.tensor(logits).view(1, 1, -1)
        else:
            hidden = torch.zeros(1, 1, 2)
        pkv = (torch.zeros(1, 1, 1, hidden.size(-1)), torch.zeros(1, 1, 1, hidden.size(-1)))
        return hidden, [pkv]

    def fake_exit_logits_from_hidden_k(spec, h_k):
        return h_k

    def fake_sample_from_logits(logits, greedy=False, temperature=1.0):
        return torch.argmax(logits, dim=-1)

    def fake_run_deep_from_k(spec, hidden_k, past_key_values=None, use_cache=True):
        if spec.deep_logits_list:
            logits = spec.deep_logits_list.pop(0)
            return torch.tensor(logits).unsqueeze(0), []
        V = hidden_k.size(-1)
        return torch.zeros(1, hidden_k.size(1), V), []

    monkeypatch.setattr(rollout_mod, "run_shallow_until_k", fake_run_shallow_until_k)
    monkeypatch.setattr(rollout_mod, "exit_logits_from_hidden_k", fake_exit_logits_from_hidden_k)
    monkeypatch.setattr(rollout_mod, "sample_from_logits", fake_sample_from_logits)
    monkeypatch.setattr(rollout_mod, "run_deep_from_k", fake_run_deep_from_k)
    monkeypatch.setattr(rollout_mod, "advance_kv_with_committed", lambda *args, **kwargs: None)

    tok = make_tok()
    buf = ReplayBuffer(16, torch.device("cpu"))
    rollout_mod.rollout_collect_k_spec(spec, tok, "hi", buf, steps=1, k=3, greedy=True, spec_adaptive=True, eta=0.9)
    assert len(buf) == 1


def test_rollout_collect_forwards_params(monkeypatch):
    captured = {}

    def fake_rollout_collect_k_spec(*args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(rollout_mod, "rollout_collect_k_spec", fake_rollout_collect_k_spec)

    spec = DummySpec()
    tok = make_tok()
    buf = ReplayBuffer(4, torch.device("cpu"))
    tele = AlignTelemetryParams(debug=1)

    rollout_mod.rollout_collect(
        spec,
        tok,
        "hi",
        buf,
        steps=1,
        telemetry=tele,
        spec_adaptive=True,
        eta=0.7,
        greedy=False,
        temperature=0.5,
    )

    assert captured.get("telemetry") is tele
    assert captured.get("spec_adaptive") is True
    assert captured.get("eta") == 0.7
    assert captured.get("greedy") is False
    assert captured.get("temperature") == 0.5
