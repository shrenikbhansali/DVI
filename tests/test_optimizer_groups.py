import os, sys, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import LlamaConfig
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import attach_dual_lora


def build_model(vsz=32, h=16, L=4, k=2, r_s=8, r_v=0):
    cfg = LlamaConfig(
        hidden_size=h,
        intermediate_size=2 * h,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vsz,
    )
    m = EarlyExitLlamaForCausalLM(cfg, EARLY_STOP_LAYER=k)
    m.exit_proj = torch.nn.Linear(h, vsz, bias=False)
    m = attach_dual_lora(m, split_layer=k, rank_s=r_s, rank_v=r_v)
    return m


def test_optimizer_param_groups():
    m = build_model()
    draft = [p for n, p in m.named_parameters() if "lora_draft" in n and p.requires_grad]
    verify = [p for n, p in m.named_parameters() if "lora_verify" in n]
    exitp = [p for n, p in m.named_parameters() if "exit_proj" in n]
    assert len(draft) > 0
    assert len(verify) == 0
    assert len(exitp) > 0
    opt = torch.optim.AdamW(
        [{"params": exitp, "lr": 1e-3}, {"params": draft, "lr": 1e-4}],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    all_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert len(all_ids) == len(exitp) + len(draft)
