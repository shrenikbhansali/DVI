import os
import sys
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training.buffer import ReplayBuffer


def test_wraparound():
    buf = ReplayBuffer(3, torch.device('cpu'))
    drops = [
        buf.append(torch.full((2,), i), token=i, reward=i % 2, conf=0.0)
        for i in range(4)
    ]
    assert len(buf) == 3
    assert drops[-1] is True
    assert sorted(buf._token_buf[:len(buf)].tolist()) == [1, 2, 3]


def test_accepted_count():
    buf = ReplayBuffer(4, torch.device('cpu'))
    for i, r in enumerate([1.0, 0.0, 1.0]):
        buf.append(torch.zeros(1), i, r, 0.0)
    assert buf.accepted_count() == 2


def test_sample_shapes_dtypes():
    buf = ReplayBuffer(5, torch.device('cpu'))
    for i in range(5):
        buf.append(torch.zeros(4), i, 1.0, 0.5)
    batch = buf.sample(3)
    assert batch['hidden'].shape == (3, 4)
    assert batch['token'].dtype == torch.long
    assert batch['reward'].dtype == torch.float32
    assert batch['conf'].dtype == torch.float32


def test_insufficient_positives():
    buf = ReplayBuffer(2, torch.device('cpu'))
    buf.append(torch.zeros(1), 0, 0.0, 0.2)
    with pytest.raises(ValueError):
        buf.sample(1, accepted_only=True)
