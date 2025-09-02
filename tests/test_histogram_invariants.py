import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.spec_decode import SpecMetrics


def test_histogram_accept_rate_alignment():
    m = SpecMetrics(draft_k=2)
    m.prefix_hist = [1, 1, 3]
    m.proposed = 10
    m.accepted = 7
    m.committed = 7
    d = m.to_dict()
    assert d["spec/hist_N"] == 5
    assert abs(d["spec/accept_rate"] - d["spec/accept_rate_from_hist"]) < 1e-6
