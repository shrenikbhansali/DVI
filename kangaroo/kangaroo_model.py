import os, json, torch, torch.nn as nn
from fastchat.utils import str_to_torch_dtype
from transformers.models.llama import LlamaConfig
from transformers.utils import logging as hf_logging

from kangaroo.adapter import AdapterModel
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM

log = hf_logging.get_logger(__name__)

class KangarooModel(nn.Module):
    def __init__(
        self,
        base_model_name_or_path: str,
        adapter_model_path: str,
        args,
        EARLY_STOP_LAYER: int = 2,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # 1) load / shard base LLAMA
        # ------------------------------------------------------------
        self.base_model: EarlyExitLlamaForCausalLM = (
            EarlyExitLlamaForCausalLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=str_to_torch_dtype(args.dtype),
                device_map="auto",
                EARLY_STOP_LAYER=EARLY_STOP_LAYER,
            ).eval()
        )

        # Which GPU hosts the early‑exit layer?
        early_gpu = self.base_model.hf_device_map.get(
            f"model.layers.{EARLY_STOP_LAYER}", 0
        )
        self.early_device = torch.device(f"cuda:{early_gpu}")

        # ------------------------------------------------------------
        # 2) Build / load adapter
        # ------------------------------------------------------------
        adapter_state_dict = None
        if adapter_model_path:  # non‑empty → try loading
            try:
                # either a local dir or a HF repo containing config.json
                cfg_dir = (
                    adapter_model_path
                    if os.path.isdir(adapter_model_path)
                    else os.path.dirname(adapter_model_path)
                )
                adapter_cfg = LlamaConfig.from_pretrained(cfg_dir)
                adapter_state_dict = torch.load(
                    os.path.join(adapter_model_path, "pytorch_model.bin"),
                    map_location="cpu",
                )
                log.info(f"Loaded adapter weights from {adapter_model_path}")
            except Exception:
                # fallback to fresh if load fails
                adapter_cfg = LlamaConfig.from_pretrained(base_model_name_or_path)
                log.warning(
                    f"Could not load adapter from {adapter_model_path}; "
                    "initialising a random AdapterModel instead."
                )
        else:
            # empty string or None → fresh random adapter
            adapter_cfg = LlamaConfig.from_pretrained(base_model_name_or_path)
            log.warning(
                "No adapter_model_path provided – initialising a *random* AdapterModel. "
                "Speculative decoding will work but accept‑rate will be near 0 until you train the adapter."
            )

        # instantiate adapter on the early‑exit shard
        self.adapter_model = AdapterModel(adapter_cfg).eval().to(self.early_device)
        if adapter_state_dict is not None:
            self.adapter_model.load_state_dict(adapter_state_dict, strict=False)
        if args.dtype == "float16":
            self.adapter_model.half()

        # ------------------------------------------------------------
        # 3) Heads
        # ------------------------------------------------------------
        base_head = self.base_model.lm_head    # verifier head (where HF placed it)
        self.head_model = base_head

        # draft head must live on the early‑exit device
        self.exit_proj = nn.Linear(
            base_head.in_features,
            base_head.out_features,
            bias=False,
            device=self.early_device,
            dtype=base_head.weight.dtype,
        )
        # copy weights from the real lm_head
        self.exit_proj.weight.data.copy_(base_head.weight.data)

        # so EarlyExitLlamaForCausalLM can find them
        self.base_model.exit_proj = self.exit_proj
        self.base_model.head_model = self.head_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError("KangarooModel is a container; call .base_model.* methods instead.")
