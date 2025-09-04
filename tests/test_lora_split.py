import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from transformers import LlamaConfig, AutoModelForCausalLM

from kangaroo.sgp_lora import inject_dual_lora, split_lora_params, enable_lora_grads


def test_toy_model_sanity():
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=4, num_attention_heads=4)
    model = AutoModelForCausalLM.from_config(config)
    model = inject_dual_lora(model, exit_layer=1, rank=4, rank_v=4)
    fast, slow = split_lora_params(model)
    assert all(".lora_draft." in n for n, _ in fast)
    assert all(".lora_verify." in n for n, _ in slow)
    assert len(fast) > 0 and len(slow) > 0


def test_layer_coverage():
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=4, num_attention_heads=4)
    model = AutoModelForCausalLM.from_config(config)
    model = inject_dual_lora(model, exit_layer=1, rank=4, rank_v=4)

    layer_adapters = {i: set() for i in range(config.num_hidden_layers)}
    for name, _ in model.named_parameters():
        if ".lora_" in name:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            if ".lora_draft." in name:
                layer_adapters[layer_idx].add("lora_draft")
            if ".lora_verify." in name:
                layer_adapters[layer_idx].add("lora_verify")
    assert layer_adapters[0] == {"lora_draft"}
    assert layer_adapters[1] == {"lora_verify"}
    assert layer_adapters[2] == {"lora_verify"}
    assert layer_adapters[3] == {"lora_verify"}


def test_enable_lora_grads_flip():
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=4, num_attention_heads=4)
    model = AutoModelForCausalLM.from_config(config)
    model = inject_dual_lora(model, exit_layer=1, rank=4, rank_v=4)
    fast, _ = split_lora_params(model)

    enable_lora_grads(model, "lora_draft", False)
    assert all(not p.requires_grad for _, p in fast)

    enable_lora_grads(model, "lora_draft", True)
    assert all(p.requires_grad for _, p in fast)
