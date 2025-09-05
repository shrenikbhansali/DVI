import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import LlamaConfig
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from training.spec_decode import generate_with_dvi_spec
from training.kv import estimate_kv_cache
import pytest


class DummyTok:
    def __call__(self, s, padding=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


def build_mismatch_model(vsz=32, h=16, L=4, ksplit=2):
    cfg = LlamaConfig(
        hidden_size=h,
        intermediate_size=2 * h,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vsz,
    )
    m = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=ksplit)
    m.early_layer = ksplit
    m.lm_head = torch.nn.Linear(h, vsz, bias=False)
    m.exit_proj = torch.nn.Linear(h, vsz, bias=False)
    with torch.no_grad():
        # Zero out the entire transformer stack and embeddings so the
        # hidden state is purely the token embedding.
        for p in m.model.parameters():
            p.zero_()
        m.model.embed_tokens.weight[:, 0] = 1.0  # constant positive feature
        m.model.norm.weight.fill_(1.0)

        # Force drafter/verifier disagreement.
        m.lm_head.weight.zero_()
        m.exit_proj.weight.zero_()
        m.lm_head.weight[1, 0] = 1.0  # verifier prefers token 1
        m.exit_proj.weight[0, 0] = 1.0  # drafter prefers token 0
    return m


def test_commit_one_on_miss_progress():
    m = build_mismatch_model()
    if not hasattr(m.model, "_prepare_decoder_attention_mask"):
        pytest.skip("missing attention mask helper")
    tok = DummyTok()
    out, metrics = generate_with_dvi_spec(m, tok, prompts=["x"], max_new_tokens=1, draft_k=2, greedy=True, early_layer=2)
    assert out[0].shape[0] == 1
    assert metrics.accepted == 0
    assert metrics.committed == 1
    assert metrics.prefix_hist[0] == 1
    assert metrics.steps == 1
    _, seq = estimate_kv_cache(m)
    assert seq == 3  # prompt (2) + committed (1)
