import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from training.utils import ctar


def test_ctar_per_prompt():
    seqs = [[1, 1, 0], [1, 1, 1]]
    assert ctar(seqs, 2) == pytest.approx(0.75)
