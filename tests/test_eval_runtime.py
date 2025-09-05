import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.acceptance import eval_runtime_acceptance
import evaluation.acceptance as acc_mod
from training.spec_decode import SpecMetrics


def test_eval_runtime_returns_hist(monkeypatch):
    def fake_generate(*args, **kwargs):
        m = SpecMetrics(draft_k=2)
        m.prefix_hist = [1, 2, 3]
        return [], m

    monkeypatch.setattr(acc_mod, "generate_with_dvi_spec", fake_generate)
    metrics = eval_runtime_acceptance(None, None, ["hi"], draft_k=2)
    assert metrics["spec/prefix_hist_0"] == 1.0
    assert metrics["spec/prefix_hist_2"] == 3.0
