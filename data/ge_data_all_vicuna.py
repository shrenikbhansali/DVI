# data/ge_data_all_vicuna.py  (final stable version)
# --------------------------------------------------------------------------- #
#  Build a mini hidden-state dataset for Draft–Verify from any Llama-2
#  checkpoint and a ShareGPT-style text source (local JSON or HF dataset id). #
# --------------------------------------------------------------------------- #
import argparse, os, json, itertools
from pathlib import Path
from typing  import Iterator

import torch
from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ---------------- CLI ------------------------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--start",  type=int, default=0)
p.add_argument("--end",    type=int, default=100)
p.add_argument("--index",  type=int, default=0)
p.add_argument("--gpu_index", nargs="+", type=int, default=[0])
p.add_argument("--outdir", type=str, default="outdir0")
p.add_argument("--model_name", type=str,
              default="meta-llama/Llama-2-7b-hf")
p.add_argument("--sharegpt_source", type=str,
              default="RyokoAI/ShareGPT52K",
              help="Local JSON/L or HF dataset id")
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

# ---------------- Dataset iterator ----------------------------------------- #
def iter_sharegpt(src:str) -> Iterator[dict]:
    """
    Yield raw ShareGPT items without ever converting to Arrow/Parquet.
    * Local file  : stream line-by-line (json or jsonl)
    * HF dataset  : load with streaming=True and islice()
    """
    if Path(src).expanduser().is_file():                        # ---- local
        file_path = Path(src).expanduser()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line.strip())
    else:                                                       # ---- hub
        ds_stream = load_dataset(src, split="train", streaming=True)
        for ex in itertools.islice(ds_stream, args.start, args.end):
            yield ex

# ---------------- Build processed slice ------------------------------------ #
def build_dataset(tok, max_len:int):
    tmpl_name = "llama-2"
    try: _ = get_conversation_template(tmpl_name)
    except ValueError: tmpl_name = "vicuna"

    processed = []
    for raw in iter_sharegpt(args.sharegpt_source):
        conv_raw = [m for m in raw["conversations"]
                      if m.get("from") in ("human","gpt")]
        if len(conv_raw) < 2: continue
        conv = get_conversation_template(tmpl_name)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        if conv_raw[0]["from"] != "human": conv_raw = conv_raw[1:]
        conv.messages = [ [roles[m["from"]], m["value"]] for m in conv_raw ]

        text = conv.get_prompt()
        ids  = tok(text, truncation=True, max_length=max_len,
                   return_tensors="pt").input_ids[0]

        mask = torch.ones_like(ids); cur = 1; mask[:cur]=0
        sep = conv.sep + conv.roles[1] + ": "
        for i, turn in enumerate(text.split(conv.sep2)):
            if not turn: break
            parts = turn.split(sep);  # user-Prompt + assistant reply
            if len(parts)!=2: break
            instr_len = len(tok(parts[0]).input_ids) - 2
            if i and not tok.legacy: instr_len -= 1
            mask[cur:cur+instr_len] = 0
            cur += len(tok(turn).input_ids)
            if i and not tok.legacy: cur -= 1
        mask[cur:] = 0
        processed.append({"input_ids": ids[None,:], "loss_mask": mask[None,:]})
    return processed

# ---------------- Load base model ----------------------------------------- #
print(f"→ Loading base model {args.model_name}")
tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                    trust_remote_code=True)
max_len = min(getattr(tok,"model_max_length",4096), 4096)
model   = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16,
            device_map="auto").eval()
n_layers   = model.config.num_hidden_layers
exit_layer = 2 if n_layers<=32 else 3
print(f"Tokenizer max_len={max_len} | exit layer={exit_layer}")

dataset = build_dataset(tok, max_len)
print(f"✓ Loaded {len(dataset)} samples")

out_root = Path(args.outdir)/str(args.index); out_root.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def encode(sample):
    ids  = sample["input_ids"].to("cuda")
    outs = model(ids, output_hidden_states=True, use_cache=False)
    return {
        "input_ids": ids.cpu()[0],
        "loss_mask": sample["loss_mask"].cpu()[0],
        f"hidden_state_layer{exit_layer}": outs.hidden_states[exit_layer].cpu()[0],
        "hidden_state": outs.hidden_states[-1].cpu()[0],
    }

for i,samp in enumerate(tqdm(dataset, desc="encoding")):
    torch.save(encode(samp), out_root/f"data_{i}.ckpt")
print(f"✓ Saved {i+1} ckpt files → {out_root}")
