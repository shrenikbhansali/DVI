# AGENTS.md
<!--
  Read me first!  This file is the on‑boarding document for any human
  *or* autonomous agent that touches the repo.
-->

---

## 0 · What is Kangaroo & What is Speculative Decoding? *(60 s primer)*

1. **Speculative Decoding (SD)**  
   An autoregressive acceleration trick:  
   *Draft* network proposes several tokens in one shot → *Verifier* network
   checks them in parallel.  
   Accepted tokens are committed; rejected ones are replaced.  
   Result: **1.5–2 ×** throughput without changing the final distribution.

2. **Kangaroo (Li et al., 2024)**  
   A *self‑speculative* variant: a **single Llama model** is split at layer *k*.  
   Layers 0‑k play the Draft role; layers k‑L play the Verifier role.  
   No auxiliary model, no extra memory.  *(We start from this codebase.)*

---

## 1 · Repo Origin & Immutable Core

This repository **forks the original → [`Equationliu/Kangaroo`](https://github.com/Equationliu/Kangaroo)**.  
We are layering **Draft → Verify → Improve RL (DVI‑RL)** on top.

> 🔒 **Files that must remain functionally intact (only *surgical* edits allowed):**
>
> | Path | Why it must be stable |
> |------|-----------------------|
> | `kangaroo/earlyexit.py` | Houses the micro‑step speculative decoder; even small logic drift can break accept/reject signals. |
> | `kangaroo/adapter.py`   | Re‑implements a slim 1‑layer decoder and **patches several HF internals** (RMSNorm, rotary cache, causal masks). Many helper methods live here; other modules must call these not HF defaults. |
> | `evaluation/inference_kangaroo.py` | Benchmark reference; changes here mask regression in speed/quality. |

Any PR that rewrites the algorithms inside these files **must** include:

1. A benchmark replicating Kangaroo’s original speed‑up on `--max_new_tokens 64`.
2. Unit tests (`pytest -q`) proving unchanged accept‑rate on a fixed seed batch.

---

## 2 · Always Check `adapter.py` First ❗
We are **NOT** using the latest version of huggingface / transformers. We are using transformers=4.33.3. The version that you have installed *is* is the latest version. Therefore, you should check against the web or check against `kangaroo/adapter.py`. 

`kangaroo/adapter.py` is **not** a thin wrapper:

* It is the transformers=4.33.3 implementations of **LlamaAttention**, **LlamaRotaryEmbedding**, causal masking, etc.
* Caps KV‑cache to 64 tokens for memory.
* Provides `_set_cos_sin_cache()`, `forward_early_stop()`, `_prepare_decoder_attention_mask()`—utilities that the rest of the codebase *will* call.

➡️ **Before you import or extend any HF module, search `adapter.py` to avoid
silent incompatibilities.**

---

## 3 · Big‑Picture Vision

This repo turns speculative decoding into a **training‑time signal**, not just an inference trick.

### 🔁 Draft → Verify → Improve (DVI-RL)

| Role | Description |
|------|-------------|
| **Draft** | Shallow layers predict next token with `lora_S` adapters |
| **Verify** | Deep layers accept/reject the draft with `lora_D` adapters |
| **Improve** | Draft gets REINFORCE signal on accept=1 / reject=0 |

This creates a **self-contained RL loop**:  
no external reward, no labels, no human supervision.

### 🤖 Practical impact

Train from **streaming conversation** logs (e.g. ShareGPT):  
no replay buffer, no human annotators, and constant compute overhead.

---

## 4 · Codebase Map

| Folder | Description |
|--------|-------------|
| `kangaroo/` | Core speculative decoding utilities: modified model, adapter patches, early exit logic |
| `evaluation/` | Inference benchmarking, logging, acceptance tracing |
| `training/` | SGP, DVI-RL trainers, slow-update routines |
| `lora/` | LoRA injection & separation of shallow/deep adapters |
| `buffer/` | Replay buffer used by fast and slow optimizers |
| `scripts/` | CLI entrypoints for train/eval/trace |
| `configs/` | YAML files for experiment reproducibility |

---

## 5 · Interfaces & Contracts

### `EarlyExitLlamaForCausalLM` (in `earlyexit.py`)

- `forward_draft_or_large_model(...)`: core forward pass, either shallow or deep
- `spec_decode_step(...)`: single step of speculative decoding with accept bit

### `adapter.py`

- Overloads HF modules: do not bypass these!
- Provides `LlamaForCausalLMWithAdapters` and `inject_adapter_hooks(...)`

### Replay Buffer (`buffer.py`)

- Fixed‑capacity FIFO buffer that stores accepted token transitions
- Used for fast policy gradient and slow verifier fine-tuning

---

## 6 · Development Guidelines

- ✅ Follow modularity: isolate logic in `trainer_dvi.py`, keep inference clean
- ✅ Add metrics to `evaluation/trace_logger.py` rather than printing manually
- ✅ Add new adapters via `inject_dual_lora(...)` (see `sgp_lora.py`)
- ✅ Unit test slow‑update schedules separately
- ❗ Respect **immutable files** (`earlyexit.py`, `adapter.py`, `inference_kangaroo.py`)
- ❗ Always check if a helper exists in `adapter.py` before modifying or re‑writing core logic (especially attention, masks, or embeddings)

---
