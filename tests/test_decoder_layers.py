import torch

from training.utils import get_num_decoder_layers
from training import utils as u


def test_get_num_decoder_layers_plain():
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(5)])
            self.config = type("cfg", (), {"num_hidden_layers": 5})()

    m = Plain()
    assert get_num_decoder_layers(m) == 5


def test_get_num_decoder_layers_peft_like():
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(4)])
            self.config = type("cfg", (), {"num_hidden_layers": 4})()

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = type("base", (), {"model": Inner()})()
            self.config = self.base_model.model.config

    w = Wrapper()
    assert get_num_decoder_layers(w) == 4


def test_get_num_decoder_layers_fallback_warning(capsys):
    u._num_layers_warned = False

    class Fallback(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("cfg", (), {"num_hidden_layers": 7})()

    m = Fallback()
    assert get_num_decoder_layers(m) == 7
    captured = capsys.readouterr()
    assert "num_hidden_layers" in captured.out

