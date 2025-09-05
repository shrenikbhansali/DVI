import torch

from training import utils


class DummyTok:
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 0
    truncation_side = "right"
    model_max_length = 8

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=None):
        ids = torch.zeros((len(texts), 1), dtype=torch.long)
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": attn}


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.past_key_values = None
        self.config = type(
            "cfg",
            (),
            {
                "eos_token_id": 0,
                "pad_token_id": 0,
                "bos_token_id": 0,
                "use_cache": True,
            },
        )()

    def get_input_embeddings(self):
        return torch.nn.Embedding(1, 1)

    def generate(self, **kwargs):
        return kwargs["input_ids"]


def _run(debug_flag: bool, monkeypatch):
    calls = {"n": 0}

    def fake_purge(model):
        calls["n"] += 1

    monkeypatch.setattr(utils, "deep_kv_purge", fake_purge)
    model = DummyModel()
    tok = DummyTok()
    utils.measure_generate_walltime(
        model,
        tok,
        ["hi"],
        max_new_tokens=1,
        greedy=True,
        repeats=1,
        use_dvi_spec=False,
        debug_purge_kv=debug_flag,
    )
    return calls["n"]


def test_no_purge_by_default(monkeypatch):
    assert _run(False, monkeypatch) == 0


def test_debug_purge_enabled(monkeypatch):
    assert _run(True, monkeypatch) > 0

