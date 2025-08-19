# tests/dvi_bestcase_kl_rl.py
# DVI best-case with richer logging & debuggability.
# Phase schedule: KL warm-start → mixed (KL+RL) ramp → RL.
# Adds: human-readable logs, JSONL dumps, accepted-only toggle with backoff, scale-safe exit head.
# Patches:
#   - Later default exit layer (24)
#   - Exit pre-norm + learnable logit scale before exit head
#   - Policy gradient uses log-prob (-E[log π_s(v)]) for stronger early gradients
#   - Sharper teacher by default (T=1.0) and stronger CE during warm-up
#   - Optimizer includes new pre-norm & scale params
#   - **FIX** tok/s measurement: CUDA synchronize + count actual collected tokens
#   - **NEW** KV cache size reporting each step (bytes/MB + estimated seq_len)
#   - **NEW** Weights & Biases tracking (entity/project defaults set)
#   - **NO-CACHE PROBE (HARD)**: probe snapshots & restores all known KV holders so nothing persists
#   - **RATE-LIMIT** KV mutation warnings (avoid spam)
#   - **W&B STEP** Always log with explicit step to keep steps monotonic.

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc, math, random, argparse, json, time, copy
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, __version__ as transformers_ver

# W&B (optional)
try:
    import wandb as _wandb
except Exception:
    _wandb = None

# repo-local
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import inject_dual_lora, enable_lora_grads
from training.buffer import ReplayBuffer

WANDB_DEFAULT_ENTITY  = "sbhansali8-georgia-institute-of-technology"
WANDB_DEFAULT_PROJECT = "DVI-Testing"

# ----------------------- utils -----------------------

def set_seed(seed: int = 1234):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dirs(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)

def env_dump(outdir: Optional[str] = None):
    info = dict(
        torch_version=torch.__version__,
        cuda=torch.version.cuda if torch.cuda.is_available() else None,
        transformers=transformers_ver,
        gpus=torch.cuda.device_count(),
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None),
    )
    msg = "[env] " + json.dumps(info, indent=2)
    print(msg, flush=True)
    if outdir:
        with open(os.path.join(outdir, "logs", "env.json"), "w") as f:
            json.dump(info, f, indent=2)

def build_prompts_from_alpaca(limit: int) -> List[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for ex in ds:
        inst = ex.get("instruction", "").strip()
        inp  = ex.get("input", "").strip()
        if inp:
            text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            text = f"### Instruction:\n{inst}\n\n### Response:\n"
        if len(text) > 20:
            prompts.append(text)
        if len(prompts) >= limit:
            break
    try:
        ds.cleanup_cache_files()
    except Exception:
        pass
    return prompts

def ctar(bits: List[int], w: int) -> float:
    if len(bits) < w: return 0.0
    tot = len(bits) - w + 1
    ok = 0
    for i in range(tot):
        if all(bits[i + j] == 1 for j in range(w)):
            ok += 1
    return ok / max(1, tot)

def free_cuda(note=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    if note: print(f"[mem] cleared caches {note}", flush=True)

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _human_bytes(n: int) -> str:
    if n is None: return "0 B"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units)-1:
        x /= 1024.0; i += 1
    return f"{x:.2f} {units[i]}"

# -------------------- KV helpers --------------------

def _kv_snapshot(spec):
    """
    Snapshot references (not clones) of all likely KV attributes so we can
    restore them exactly even if the underlying forward replaces them.
    """
    slots = []
    for base_name, obj in (("spec", spec), ("model", getattr(spec, "model", None))):
        if obj is None: continue
        for attr in ("past_key_values", "_past_key_values"):
            if hasattr(obj, attr):
                slots.append((obj, attr, getattr(obj, attr, None)))
    return slots

def _kv_restore(slots):
    for obj, attr, val in slots:
        try:
            setattr(obj, attr, val)
        except Exception:
            pass

def _first_nonempty_pkv(spec: EarlyExitLlamaForCausalLM):
    """
    Look across common KV holders.
    """
    for obj in (spec, getattr(spec, "model", None)):
        if obj is None: continue
        for name in ("past_key_values", "_past_key_values"):
            if hasattr(obj, name):
                pkv = getattr(obj, name)
                if pkv:  # non-empty container
                    return pkv
    return None

def estimate_kv_cache(spec: EarlyExitLlamaForCausalLM) -> Tuple[int, int]:
    """
    Estimate current KV cache size in bytes and approximate seq_len.
    """
    pkv = _first_nonempty_pkv(spec)
    if not pkv:
        return 0, 0
    total_bytes = 0
    est_seq_len = 0
    try:
        for layer in pkv:
            for t in layer:
                if isinstance(t, torch.Tensor):
                    total_bytes += t.element_size() * t.nelement()
                    if t.ndim >= 2:
                        est_seq_len = max(est_seq_len, int(t.shape[-2]))
    except Exception:
        pass
    return int(total_bytes), int(est_seq_len)

def clear_all_kv(spec, verbose: bool = False, tag: str = ""):
    """
    Wipes *all* likely KV holders (spec & spec.model, both names).
    """
    touched = []
    for obj in (spec, getattr(spec, "model", None)):
        if obj is None: continue
        for name in ("past_key_values", "_past_key_values"):
            if hasattr(obj, name):
                try:
                    setattr(obj, name, None)
                    touched.append(f"{obj.__class__.__name__}.{name}")
                except Exception:
                    pass
    if verbose:
        print(f"[kv-clear]{(' '+tag) if tag else ''} -> {', '.join(touched) if touched else 'none'}", flush=True)

# -------------------- draft path helpers --------------------

@torch.inference_mode()
def prime_kv_full(spec: EarlyExitLlamaForCausalLM, input_ids: torch.Tensor):
    # fresh KV for both drafter & verifier paths
    clear_all_kv(spec)
    h = spec.forward_draft_or_large_model(in_tokens_small=input_ids)
    _ , _ = spec.forward_draft_or_large_model(in_features_large=h)

@torch.inference_mode()
def advance_kv_with_committed(spec: EarlyExitLlamaForCausalLM, token_ids: torch.Tensor):
    # commit the teacher token into KV (this path SHOULD use cache)
    h = spec.forward_draft_or_large_model(in_tokens_small=token_ids)
    _ , _ = spec.forward_draft_or_large_model(in_features_large=h)

def _top1(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1, keepdim=True)[0,0].item())

# -------------------- strictly side-effect free drafter probe --------------------

@torch.inference_mode()
def drafter_hidden_no_cache(spec: EarlyExitLlamaForCausalLM, ids_last: torch.Tensor) -> torch.Tensor:
    """
    Compute drafter hidden for next token WITHOUT persisting any KV changes.
    Strategy:
      1) Snapshot all known KV holders.
      2) Try to call with use_cache=False.
      3) If unsupported, temporarily toggle config.use_cache=False.
      4) ALWAYS restore the exact KV objects we snapshotted in (1).
    Returns [B,1,H] hidden.
    """
    slots = _kv_snapshot(spec)

    # First try: pass the flag down if the wrapper supports it
    try:
        out = spec.forward_draft_or_large_model(in_tokens_small=ids_last, use_cache=False)
        return out
    except TypeError:
        pass
    except Exception:
        pass
    finally:
        # If forward wrote anything, this puts it back immediately.
        _kv_restore(slots)

    # Fallback: temporarily disable model-level use_cache
    toggled = []
    def _toggle(obj):
        if obj is not None and hasattr(obj, "config") and hasattr(obj.config, "use_cache"):
            toggled.append((obj, obj.config.use_cache))
            obj.config.use_cache = False
    _toggle(spec)
    _toggle(getattr(spec, "model", None))
    try:
        out = spec.forward_draft_or_large_model(in_tokens_small=ids_last)
    finally:
        for obj, prev in toggled:
            try: obj.config.use_cache = prev
            except Exception: pass
        _kv_restore(slots)
    return out

# -------------------- exit logits helper --------------------

def model_exit_logits(spec: EarlyExitLlamaForCausalLM, h: Optional[torch.Tensor] = None,
                      logits_in: Optional[torch.Tensor] = None, preproj_h: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply pre-norm + exit head with optional global scale.
    """
    assert (h is not None) ^ (preproj_h is not None), "Provide either h or preproj_h"
    x = h.float() if preproj_h is None else preproj_h.float()
    if preproj_h is None and hasattr(spec, "exit_pre_norm") and spec.exit_pre_norm is not None:
        x = spec.exit_pre_norm(x)
    out = spec.exit_proj(x)
    if hasattr(spec, "exit_logit_scale"):
        out = spec.exit_logit_scale * out
    return out

# -------------------- evaluation (acceptance metric) --------------------

_KV_WARN_COUNT = 0
_KV_WARN_LIMIT = 8

@torch.inference_mode()
def eval_acceptance(spec: EarlyExitLlamaForCausalLM, tok: AutoTokenizer, prompts: List[str],
                    rollout_len: int, steps_per_prompt: int = 1,
                    dump_debug: bool = False, dump_path: Optional[str] = None, topk: int = 5,
                    quiet: bool = False) -> Tuple[float, Dict[int,float]]:
    global _KV_WARN_COUNT
    spec.eval()
    dev = next(spec.parameters()).device
    accepts = []
    dbg_out = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(dev)
        prime_kv_full(spec, enc["input_ids"])
        ids_last = enc["input_ids"][:, -1:]
        for _ in range(steps_per_prompt * rollout_len):
            v_logits = spec.verifier_logits_for_next(ids_last)
            if not torch.isfinite(v_logits).all():
                accepts.append(0); break
            v_top1 = _top1(v_logits)

            # Take KV measurement, probe (side-effect free), measure again
            kv_bytes_b, kv_seq_b = estimate_kv_cache(spec)
            d_hidden = drafter_hidden_no_cache(spec, ids_last)  # [B,1,H], restores KV inside
            kv_bytes_a, kv_seq_a = estimate_kv_cache(spec)
            if (kv_bytes_b != kv_bytes_a or kv_seq_b != kv_seq_a) and _KV_WARN_COUNT < _KV_WARN_LIMIT:
                print("[kv] WARNING: probe seems to have mutated past_key_values during eval.", flush=True)
                _KV_WARN_COUNT += 1

            # exit logits
            h = d_hidden.squeeze(1).float()
            if hasattr(spec, "exit_pre_norm") and spec.exit_pre_norm is not None:
                h = spec.exit_pre_norm(h)
            d_logits = spec.exit_proj(h)
            if hasattr(spec, "exit_logit_scale"):
                d_logits = spec.exit_logit_scale * d_logits

            if not torch.isfinite(d_logits).all():
                accepts.append(0)
                d_top1 = -1
            else:
                d_top1 = _top1(d_logits)
                accepts.append(int(d_top1 == v_top1))

            if dump_debug and dump_path:
                try:
                    v_prob, v_id = torch.topk(torch.softmax(v_logits.float(), dim=-1)[0], k=topk)
                    d_prob, d_id = torch.topk(torch.softmax(d_logits.float(), dim=-1)[0], k=topk)
                    dbg_out.append({
                        "draft_top1": int(d_top1),
                        "verifier_top1": int(v_top1),
                        "accept": int(d_top1 == v_top1),
                        "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                        "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                    })
                except Exception:
                    pass

            # commit verifier token (this updates/cache as intended)
            v_token = torch.tensor([[v_top1]], device=ids_last.device, dtype=ids_last.dtype)
            advance_kv_with_committed(spec, v_token)
            ids_last = v_token

        # end-of-prompt cleanup: drop KV & clear caches
        clear_all_kv(spec)
        if not quiet:
            free_cuda("(eval prompt done)")
        else:
            free_cuda("")

    acc = sum(accepts)/max(1, len(accepts))
    c = {w: ctar(accepts, w) for w in (1,2,3,4)}

    if dump_debug and dump_path and dbg_out:
        with open(dump_path, "a") as f:
            for rec in dbg_out:
                f.write(json.dumps(rec) + "\n")
    return acc, c

# -------------------- model (train drafter only) --------------------

def prepare_dvi_trainable(model_id: str, early_layer: int, dtype=torch.float16):
    model = EarlyExitLlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", EARLY_STOP_LAYER=early_layer
    )
    model = inject_dual_lora(model, exit_layer=early_layer, rank=8)
    for p in model.parameters(): p.requires_grad = False
    enable_lora_grads(model, "lora_S", True)    # drafter learns
    enable_lora_grads(model, "lora_D", False)   # deep path frozen

    with torch.no_grad():
        w = model.lm_head.weight.detach().clone().float()

    # Exit head
    model.exit_proj = nn.Linear(w.shape[1], w.shape[0], bias=False,
                                device=w.device, dtype=torch.float32)
    model.exit_proj.weight.data.copy_(w)
    model.exit_proj.weight.requires_grad = True

    # add pre-norm + learnable global logit scale
    try:
        base_norm = model.model.norm  # Llama final RMSNorm
        model.exit_pre_norm = copy.deepcopy(base_norm).to(w.device)
    except Exception:
        model.exit_pre_norm = nn.LayerNorm(w.shape[1], elementwise_affine=True, device=w.device)
    for p in model.exit_pre_norm.parameters():
        p.requires_grad = True

    model.exit_logit_scale = nn.Parameter(torch.tensor(1.0, device=w.device))

    model.lm_head.weight.requires_grad = False
    if hasattr(model, "head_model"):
        for p in model.head_model.parameters(): p.requires_grad = False
    return model

def build_optimizer(model, lr_exit=2e-4, lr_lora=5e-5, wd_exit=1e-2, wd_lora=0.0):
    head_params = [model.exit_proj.weight]
    if hasattr(model, "exit_pre_norm") and model.exit_pre_norm is not None:
        head_params += list(model.exit_pre_norm.parameters())
    if hasattr(model, "exit_logit_scale"):
        head_params += [model.exit_logit_scale]

    groups = [
        {"params": head_params, "lr": lr_exit, "weight_decay": wd_exit},
    ]
    lora_s = [p for n,p in model.named_parameters() if p.requires_grad and "lora_S" in n]
    if lora_s:
        groups.append({"params": lora_s, "lr": lr_lora, "weight_decay": wd_lora})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

# -------------------- objectives --------------------

def _maybe_clamp_exit_head(model, init_fro: float, max_fro: float, max_fro_ratio: float):
    """Clamp Frobenius norm only if a bound is set; prefer relative cap."""
    if (max_fro is None or max_fro <= 0.0) and (max_fro_ratio is None or max_fro_ratio <= 0.0):
        return  # disabled
    with torch.no_grad():
        W = model.exit_proj.weight
        n = torch.linalg.vector_norm(W).float().item()
        bound = None
        if max_fro and max_fro > 0:
            bound = max_fro
        if max_fro_ratio and max_fro_ratio > 0:
            bound_rel = init_fro * max_fro_ratio
            bound = min(bound, bound_rel) if bound is not None else bound_rel
        if bound is not None and math.isfinite(n) and n > bound:
            W.mul_(bound / n)

def _logit_stats(s_logits: torch.Tensor, t_logits: torch.Tensor):
    s = s_logits.float()
    t = t_logits.float()
    return float(s.std().item()), float(t.std().item())

def one_mixed_step(model, opt, batch,
                   temperature=1.0, clip=1.0,
                   ce_weight=0.20,
                   lam_pg=0.0, lam_kl=1.0, ent_weight=0.0,
                   init_fro: float = None, max_fro: float = 0.0, max_fro_ratio: float = 0.0):
    """
    Mixed objective:
      - RL surrogate: maximize E[log π_s(v)]  => loss_pg = -E[log π_s(v)]
      - KL(teacher -> student) with softened teacher
      - CE: anchor on teacher top-1 (stabilizer)
      - ENT: optional entropy bonus (maximize entropy)
    """
    dev = next(model.parameters()).device
    hidden  = batch["hidden"].to(dev)          # [B, H]
    vlogits = batch["vlogits"].to(dev)         # [B, V] (fp16 ok)
    tokens  = batch["token"].to(dev).view(-1)  # [B]

    # pre-norm + scale
    h = hidden.float()
    if hasattr(model, "exit_pre_norm") and model.exit_pre_norm is not None:
        h = model.exit_pre_norm(h)
    slogits = model.exit_proj(h)               # [B, V] fp32
    if hasattr(model, "exit_logit_scale"):
        slogits = model.exit_logit_scale * slogits

    slogp   = F.log_softmax(slogits, dim=-1)
    sp      = slogp.exp()

    # RL (expected acceptance, log-prob surrogate)
    pi_v = sp.gather(1, tokens.view(-1,1)).squeeze(1)  # [B]
    loss_pg = -torch.log(pi_v.clamp_min(1e-8)).mean()

    # KL (teacher -> student)
    tlogits = vlogits.float() / float(temperature)
    tlogp   = F.log_softmax(tlogits, dim=-1)
    tp      = tlogp.exp()
    kl = F.kl_div(input=slogp, target=tp, reduction="batchmean", log_target=False)

    # CE anchor
    ce = F.nll_loss(slogp, tokens, reduction="mean") if ce_weight > 0.0 else torch.tensor(0.0, device=dev)

    # Entropy bonus (maximize entropy -> subtract from loss)
    ent = -(sp * slogp).sum(-1).mean()

    # Weighted total + per-term contributions
    loss = lam_pg*loss_pg + lam_kl*kl + ce_weight*ce - ent_weight*ent
    contrib_pg  = float((lam_pg*loss_pg).detach().item())
    contrib_kl  = float((lam_kl*kl).detach().item())
    contrib_ce  = float((ce_weight*ce).detach().item()) if ce_weight > 0.0 else 0.0
    contrib_ent = float((-ent_weight*ent).detach().item()) if ent_weight > 0.0 else 0.0

    # NaN/Inf guard
    is_finite = (torch.isfinite(loss) & torch.isfinite(kl) & torch.isfinite(loss_pg) & torch.isfinite(ce) & torch.isfinite(ent)).item()
    if not bool(is_finite):
        opt.zero_grad(set_to_none=True)
        return dict(ok=False, loss=float("nan"), kl=float("nan"), pg=float("nan"),
                    ce=float("nan"), ent=float("nan"), grad=float("nan"),
                    c_pg=contrib_pg, c_kl=contrib_kl, c_ce=contrib_ce, c_ent=contrib_ent,
                    pi_v=float(pi_v.mean().item()), std_s_t=(0.0, 0.0))

    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), clip)
    opt.step(); opt.zero_grad(set_to_none=True)

    # Gentle, configurable clamp (disabled by default)
    _maybe_clamp_exit_head(model, init_fro, max_fro, max_fro_ratio)

    std_s, std_t = _logit_stats(slogits.detach(), vlogits.detach())
    return dict(ok=True, loss=float(loss.item()), kl=float(kl.item()), pg=float(loss_pg.item()),
                ce=float(ce.item()), ent=float(ent.item()), grad=float(grad),
                c_pg=contrib_pg, c_kl=contrib_kl, c_ce=contrib_ce, c_ent=contrib_ent,
                pi_v=float(pi_v.mean().item()), std_s_t=(std_s, std_t))

# -------------------- rollout & buffer --------------------

@torch.inference_mode()
def rollout_collect(spec: EarlyExitLlamaForCausalLM, tok: AutoTokenizer, prompt: str,
                    buf: ReplayBuffer, steps: int,
                    debug_out: Optional[List[Dict]] = None, topk: int = 5) -> int:
    """
    Collect up to `steps` tokens; returns the actual number collected.
    """
    spec.eval()
    dev = next(spec.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(dev)
    prime_kv_full(spec, enc["input_ids"])
    last = enc["input_ids"][:, -1:]

    n_collected = 0
    for _ in range(steps):
        v_logits = spec.verifier_logits_for_next(last)
        if not torch.isfinite(v_logits).all():
            break
        v_top1   = _top1(v_logits)

        # --- probe drafter WITHOUT persisting KV changes ---
        d_hidden   = drafter_hidden_no_cache(spec, last)      # [B,1,H]
        # pre-norm + scale
        h = d_hidden.squeeze(1).float()
        if hasattr(spec, "exit_pre_norm") and spec.exit_pre_norm is not None:
            h = spec.exit_pre_norm(h)
        d_logits   = spec.exit_proj(h)                        # [B,V]
        if hasattr(spec, "exit_logit_scale"):
            d_logits = spec.exit_logit_scale * d_logits

        d_top1     = _top1(d_logits) if torch.isfinite(d_logits).all() else -1
        accept_bit = int(d_top1 == v_top1)

        buf.append(
            hidden=d_hidden.squeeze(0).squeeze(0).cpu(),  # [H]
            token=int(v_top1),                             # teacher top-1 id
            reward=float(accept_bit),                     # acceptance bit
            conf=0.0,
            vlogits=v_logits.squeeze(0).cpu(),            # teacher logits
        )

        if debug_out is not None:
            try:
                v_prob, v_id = torch.topk(torch.softmax(v_logits.float(), dim=-1)[0], k=topk)
                d_prob, d_id = torch.topk(torch.softmax(d_logits.float(), dim=-1)[0], k=topk)
                debug_out.append({
                    "draft_top1": int(d_top1),
                    "verifier_top1": int(v_top1),
                    "accept": accept_bit,
                    "verifier_topk": [[int(i), float(p)] for p, i in zip(v_prob.tolist(), v_id.tolist())],
                    "draft_topk": [[int(i), float(p)] for p, i in zip(d_prob.tolist(), d_id.tolist())],
                })
            except Exception:
                pass

        # commit verifier token into KV for next step
        v_token = torch.tensor([[v_top1]], device=last.device, dtype=last.dtype)
        advance_kv_with_committed(spec, v_token)
        last = v_token
        n_collected += 1

    return n_collected

def buf_debug(buf: ReplayBuffer, k: int = 16):
    size = len(buf)
    acc  = buf.accepted_count()
    stats = {"total": size, "accepted": acc, "accept_rate_est": (acc/size) if size else 0.0}
    print("[buf]", json.dumps(stats, indent=2))
    if size == 0: return
    show = min(k, size)
    try:
        samp = buf.sample(show, accepted_only=False)
        for i in range(show):
            print(f"  sample[{i:02d}] tok={int(samp['token'][i])} r={float(samp['reward'][i])}")
    except ValueError:
        pass

# -------------------- schedules --------------------

def mix_schedule(step, warmup_kl:int, ramp:int, kl_min:float, pg_max:float):
    """Linear: KL 1→kl_min; PG 0→pg_max after warmup."""
    if step < warmup_kl:
        return 0.0, 1.0
    t = min(1.0, (step - warmup_kl) / max(1, ramp))
    lam_pg = pg_max * t
    lam_kl = (1.0 - t) * 1.0 + t * kl_min
    return lam_pg, lam_kl

def phase_of_step(step, warmup_kl, ramp):
    if step < warmup_kl:
        return "WARMUP(KL)"
    elif step < warmup_kl + ramp:
        return "MIXED"
    else:
        return "RL"

# --- W&B helpers -------------------------------------------------

def init_wandb(args) -> Optional[object]:
    if args.no_wandb:
        print("[wandb] disabled via --no-wandb", flush=True)
        return None
    if _wandb is None:
        print("[wandb] package not available; continuing without W&B logging.", flush=True)
        return None

    entity  = args.wandb_entity or WANDB_DEFAULT_ENTITY
    project = args.wandb_project or WANDB_DEFAULT_PROJECT
    name    = args.run_name or f"{os.path.basename(args.model_id)}-L{args.early_layer}-seed{args.seed}-{int(time.time())}"

    try:
        run = _wandb.init(
            entity=entity,
            project=project,
            name=name,
            config=vars(args),
            settings=_wandb.Settings(start_method="thread"),
        )
        print(f"[wandb] initialized: entity={entity} project={project} name={name}", flush=True)
        return run
    except Exception as e:
        print(f"[wandb] init failed: {e}; continuing without W&B.", flush=True)
        return None

def wandb_watch_model(model, log_freq: int = 25):
    """Call this AFTER the model is constructed to watch gradients/params."""
    if _wandb is None or _wandb.run is None:
        return
    try:
        _wandb.watch(model, log="gradients", log_freq=log_freq, log_graph=False)
    except TypeError:
        try:
            _wandb.run.watch(model, log="gradients", log_freq=log_freq, log_graph=False)
        except Exception as e:
            print(f"[wandb] watch skipped: {e}", flush=True)
    except Exception as e:
        print(f"[wandb] watch skipped: {e}", flush=True)

def wandb_log(d: Dict, step: Optional[int] = None):
    if _wandb is None or _wandb.run is None:
        return
    try:
        _wandb.log(d, step=step)
    except Exception:
        pass
# ----------------------------------------------------------------

# -------------------- training loop --------------------

def train_bestcase_kl_rl(model, tok, prompts_train, prompts_eval,
                         steps: int, rollout_len: int, batch_size: int,
                         lr_exit: float, lr_lora: float, temperature: float,
                         ce_weight: float, eval_every: int,
                         warmup_kl: int, ramp_steps: int,
                         kl_min: float, pg_max: float, ent_weight: float,
                         outdir: str, accepted_only_flag: bool,
                         debug_dump_every: int, debug_topk: int,
                         max_fro: float, max_fro_ratio: float,
                         quiet_eval: bool):

    metrics_path = os.path.join(outdir, "logs", "train_metrics.jsonl")
    samples_path = os.path.join(outdir, "logs", "rollout_samples.jsonl")

    opt = build_optimizer(model, lr_exit=lr_exit, lr_lora=lr_lora, wd_exit=1e-2, wd_lora=0.0)
    buf = ReplayBuffer(capacity=max(4096, batch_size*rollout_len*8), device=torch.device("cpu"))

    # Pre-train eval & initial norms
    model.eval()
    with torch.no_grad():
        init_fro = float(torch.linalg.vector_norm(model.exit_proj.weight).item())
        lm_norm  = float(torch.linalg.vector_norm(model.lm_head.weight).item())
    print(f"[sanity] head norms: ||lm||={lm_norm:.3f} ||exit||={init_fro:.3f}", flush=True)

    # W&B initial log (explicit step=0)
    wandb_log({"sanity/lm_head_norm": lm_norm, "sanity/exit_head_norm": init_fro}, step=0)

    print("\n[e2e] measuring acceptance (pre-train)…", flush=True)
    free_cuda("(pre-eval)")
    acc0, ctar0 = eval_acceptance(
        model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
        dump_debug=True, dump_path=samples_path, topk=debug_topk, quiet=quiet_eval
    )
    print(f"[e2e] PRE  | acc={acc0:.3f} | CTAR1={ctar0[1]:.3f} CTAR2={ctar0[2]:.3f} CTAR3={ctar0[3]:.3f} CTAR4={ctar0[4]:.3f}", flush=True)
    wandb_log({"eval/pre/acc": acc0,
               "eval/pre/ctar1": ctar0[1], "eval/pre/ctar2": ctar0[2],
               "eval/pre/ctar3": ctar0[3], "eval/pre/ctar4": ctar0[4]}, step=0)

    ptr = 0
    tokens_total = 0
    ema_tok_s = None

    for g in range(steps):
        model.eval()  # rollout in eval mode
        p = prompts_train[ptr % len(prompts_train)]; ptr += 1

        # Accurate tok/s: synchronize + count actual collected tokens
        dbg_roll = [] if (debug_dump_every and (g % debug_dump_every == 0)) else None
        _cuda_sync()
        t_roll_s = time.perf_counter()
        n_collected = rollout_collect(model, tok, p, buf, steps=rollout_len, debug_out=dbg_roll, topk=debug_topk)
        _cuda_sync()
        t_roll_e = time.perf_counter()
        elapsed = max(1e-6, (t_roll_e - t_roll_s))
        tok_s = float(n_collected) / elapsed
        tokens_total += n_collected
        ema_tok_s = tok_s if ema_tok_s is None else (0.9*ema_tok_s + 0.1*tok_s)

        # KV cache & GPU memory stats (post-rollout; KV holds last state)
        kv_bytes, kv_seq = estimate_kv_cache(model)
        kv_mb = kv_bytes / (1024**2)
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            mem_reserved = torch.cuda.memory_reserved() / (1024**2)
            mem_max = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            mem_alloc = mem_reserved = mem_max = 0.0

        if g % 10 == 0:
            buf_debug(buf, k=16)

        # wait for enough data
        if len(buf) < batch_size:
            print(f"[train] step {g:04d} | {phase_of_step(g, warmup_kl, ramp_steps):>10} | "
                  f"buf={len(buf)}/{batch_size} | tok/s={tok_s:.2f} (EMA {ema_tok_s:.2f}) | "
                  f"KV={kv_mb:.2f} MB (seq≈{kv_seq}) | waiting for batch…", flush=True)
            wandb_log({"roll/tok_s": tok_s, "roll/tok_s_ema": ema_tok_s,
                       "kv/bytes": kv_bytes, "kv/MB": kv_mb, "kv/seq_est": kv_seq,
                       "cuda/mem_alloc_MB": mem_alloc, "cuda/mem_reserved_MB": mem_reserved,
                       "cuda/mem_max_alloc_MB": mem_max,
                       "buf/size": len(buf)}, step=g)
            continue

        # schedule
        lam_pg, lam_kl = mix_schedule(g, warmup_kl, ramp_steps, kl_min, pg_max)

        # sampling policy
        phase = phase_of_step(g, warmup_kl, ramp_steps)
        accepted_only = accepted_only_flag and (phase != "WARMUP(KL)") and (lam_pg > 0.0)
        try:
            batch = buf.sample(batch_size, accepted_only=accepted_only)
        except ValueError:
            if accepted_only:
                print(f"[train][backoff] step {g:04d} | accepted-only requested but pool too small; sampling all tokens this step.")
                batch = buf.sample(batch_size, accepted_only=False)
            else:
                print(f"[train] step {g:04d} | buffer insufficient (total={len(buf)}); collecting more…")
                continue

        # one train step
        model.train()
        _cuda_sync()
        t_trn_s = time.perf_counter()
        out = one_mixed_step(
            model, opt, batch,
            temperature=temperature, clip=1.0,
            ce_weight=ce_weight, lam_pg=lam_pg, lam_kl=lam_kl,
            ent_weight=ent_weight,
            init_fro=init_fro, max_fro=max_fro, max_fro_ratio=max_fro_ratio
        )
        _cuda_sync()
        t_trn_e = time.perf_counter()
        step_time = (t_trn_e - t_trn_s)

        # quick metrics
        accept_roll = float(batch["reward"].float().mean().item())
        buf_size = len(buf); buf_acc = buf.accepted_count()
        buf_rate = (buf_acc / buf_size) if buf_size else 0.0
        with torch.no_grad():
            exit_norm = float(torch.linalg.vector_norm(model.exit_proj.weight).item())

        # safety recovery
        if not out["ok"] or not math.isfinite(out["loss"]) or not math.isfinite(out["kl"]):
            print(f"[train][nan] step {g:04d} | {phase:>10} | "
                  f"loss {out['loss']} | KL {out['kl']} | PG {out['pg']} | CE {out['ce']} | "
                  f"grad {out['grad']} -> clearing opt state",
                  flush=True)
            opt.state.clear()
            free_cuda("(nan-recovery)")
            continue

        # human-readable line
        std_s, std_t = out["std_s_t"]
        print(
            f"[train] step {g:04d} | {phase:>10} | "
            f"loss {out['loss']:+7.4f} "
            f"(PG {out['pg']:+7.4f}→{out['c_pg']:+7.4f} , KL {out['kl']:+7.4f}→{out['c_kl']:+7.4f} , "
            f"CE {out['ce']:+6.4f}→{out['c_ce']:+6.4f} , ENT {out['ent']:+6.4f}→{out['c_ent']:+6.4f}) | "
            f"grad {out['grad']:6.2f} | tok/s={tok_s:.2f} (EMA {ema_tok_s:.2f}) | step_time {step_time*1e3:6.1f} ms | "
            f"π_s(v)≈{out['pi_v']:.4f} | std_s={std_s:.3f} std_t={std_t:.3f} | "
            f"||exit||={exit_norm:.2f} | roll_acc≈{accept_roll:.4f} | "
            f"buf_size={buf_size} acc_rate≈{buf_rate:.3f} | λ_pg={lam_pg:.3f} λ_kl={lam_kl:.3f} | accepted_only={accepted_only} | "
            f"KV={kv_mb:.2f} MB (seq≈{kv_seq}) | CUDA alloc={mem_alloc:.1f} MB",
            flush=True
        )

        # JSONL metrics dump
        metrics = dict(
            step=g, phase=phase,
            loss=out["loss"], kl=out["kl"], pg=out["pg"], ce=out["ce"], ent=out["ent"],
            c_pg=out["c_pg"], c_kl=out["c_kl"], c_ce=out["c_ce"], c_ent=out["c_ent"],
            grad=float(out["grad"]), tok_s=tok_s, tok_s_ema=ema_tok_s, step_time_ms=step_time*1e3,
            pi_s_v=out["pi_v"], std_s=std_s, std_t=std_t,
            exit_norm=exit_norm,
            roll_acc=accept_roll, buf_size=buf_size, buf_acc_rate=buf_rate,
            lam_pg=lam_pg, lam_kl=lam_kl, accepted_only=accepted_only,
            kv_bytes=kv_bytes, kv_mb=kv_mb, kv_seq_est=kv_seq,
            cuda_mem_alloc_mb=mem_alloc, cuda_mem_reserved_mb=mem_reserved, cuda_mem_max_alloc_mb=mem_max,
            tokens_total=tokens_total,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # W&B log (explicit step=g)
        wandb_log({
            "loss/total": out["loss"], "loss/pg": out["pg"], "loss/kl": out["kl"],
            "loss/ce": out["ce"], "loss/ent": out["ent"],
            "contrib/pg": out["c_pg"], "contrib/kl": out["c_kl"],
            "contrib/ce": out["c_ce"], "contrib/ent": out["c_ent"],
            "grad/norm": float(out["grad"]),
            "roll/tok_s": tok_s, "roll/tok_s_ema": ema_tok_s,
            "time/step_ms": step_time*1e3,
            "policy/pi_s_v": out["pi_v"], "logits/std_s": std_s, "logits/std_t": std_t,
            "head/exit_norm": exit_norm,
            "roll/accept_rate_batch": accept_roll,
            "buf/size": buf_size, "buf/accept_rate": buf_rate,
            "schedule/lam_pg": lam_pg, "schedule/lam_kl": lam_kl,
            "sampling/accepted_only": int(accepted_only),
            "kv/bytes": kv_bytes, "kv/MB": kv_mb, "kv/seq_est": kv_seq,
            "cuda/mem_alloc_MB": mem_alloc, "cuda/mem_reserved_MB": mem_reserved, "cuda/mem_max_alloc_MB": mem_max,
            "tokens/total": tokens_total,
        }, step=g)

        # rollout sample dump if requested
        if dbg_roll:
            with open(samples_path, "a") as f:
                for rec in dbg_roll:
                    f.write(json.dumps({"step": g, **rec}) + "\n")

        # mid-train eval
        if (g+1) % eval_every == 0:
            model.eval()
            free_cuda("(pre mid-train eval)")
            print("[e2e] measuring acceptance (mid-train)…", flush=True)
            acc_mid, ctar_mid = eval_acceptance(
                model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
                dump_debug=True, dump_path=samples_path, topk=debug_topk, quiet=quiet_eval
            )
            print(f"[e2e] MID  | step {g+1:04d} | acc={acc_mid:.3f} | "
                  f"CTAR1={ctar_mid[1]:.3f} CTAR2={ctar_mid[2]:.3f} CTAR3={ctar_mid[3]:.3f} CTAR4={ctar_mid[4]:.3f}",
                  flush=True)
            wandb_log({"eval/mid/acc": acc_mid,
                       "eval/mid/ctar1": ctar_mid[1], "eval/mid/ctar2": ctar_mid[2],
                       "eval/mid/ctar3": ctar_mid[3], "eval/mid/ctar4": ctar_mid[4]}, step=g+1)

    # Final eval — free training state first
    del opt
    del buf
    free_cuda("(pre post-train eval)")
    model.eval()

    print("\n[e2e] measuring acceptance (post-train)…", flush=True)
    acc1, ctar1 = eval_acceptance(
        model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
        dump_debug=True, dump_path=samples_path, topk=debug_topk, quiet=quiet_eval
    )
    print(f"[e2e] POST | acc={acc1:.3f} | Δ={acc1-acc0:+.3f} | "
          f"CTAR1={ctar1[1]:.3f} CTAR2={ctar1[2]:.3f} CTAR3={ctar1[3]:.3f} CTAR4={ctar1[4]:.3f}", flush=True)
    wandb_log({"eval/post/acc": acc1,
               "eval/post/ctar1": ctar1[1], "eval/post/ctar2": ctar1[2],
               "eval/post/ctar3": ctar1[3], "eval/post/ctar4": ctar1[4],
               "eval/delta_acc": (acc1-acc0)}, step=steps)

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--early-layer", type=int, default=4)  # later default exit
    ap.add_argument("--train-prompts", type=int, default=512)
    ap.add_argument("--eval-prompts", type=int, default=64)

    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rollout", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)

    # patched defaults: stronger head LR; sharper teacher; stronger CE
    ap.add_argument("--lr-exit", type=float, default=5e-4)
    ap.add_argument("--lr-lora", type=float, default=5e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.20)
    ap.add_argument("--ent-weight", type=float, default=0.00)

    # KL->RL schedule (shorter warmup & ramp)
    ap.add_argument("--warmup-kl", type=int, default=40)
    ap.add_argument("--ramp-steps", type=int, default=80)
    ap.add_argument("--kl-min", type=float, default=0.05)
    ap.add_argument("--pg-max", type=float, default=1.0)

    # Debugging / logging
    ap.add_argument("--outdir", type=str, default="minbench_out")
    ap.add_argument("--debug-dump-every", type=int, default=25, help="Dump rollout samples every N steps (0 to disable).")
    ap.add_argument("--debug-topk", type=int, default=5, help="Top-k to include in sample dumps.")
    ap.add_argument("--accepted-only", action="store_true",
                    help="Prefer training only on accepted tokens (activated in MIXED/RL; backoffs to all when insufficient).")

    # Exit-head scale safety (disabled by default)
    ap.add_argument("--max-fro", type=float, default=0.0, help="Absolute clamp for ‖exit_proj‖. 0 disables.")
    ap.add_argument("--max-fro-ratio", type=float, default=0.0, help="Relative clamp vs initial ‖exit_proj‖ (e.g., 1.5). 0 disables.")

    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)

    # W&B args
    ap.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    ap.add_argument("--wandb-entity", type=str, default=WANDB_DEFAULT_ENTITY, help="W&B entity (team or user).")
    ap.add_argument("--wandb-project", type=str, default=WANDB_DEFAULT_PROJECT, help="W&B project name.")
    ap.add_argument("--run-name", type=str, default=None, help="Optional W&B run name.")

    # Noise control for eval
    ap.add_argument("--quiet-eval", action="store_true", help="Silence per-prompt cache clear prints during eval.")

    args = ap.parse_args()

    ensure_dirs(args.outdir)
    env_dump(args.outdir)
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    print("[e2e] loading prompts…", flush=True)
    prompts_train = build_prompts_from_alpaca(args.train_prompts)
    prompts_eval  = build_prompts_from_alpaca(args.eval_prompts)

    print("[e2e] loading model…", flush=True)
    model = prepare_dvi_trainable(args.model_id, args.early_layer, dtype=torch.float16)

    # sanity: head parity & norms (before any training)
    with torch.no_grad():
        lm_nrm   = model.lm_head.weight.detach().float().norm().item()
        exit_nrm = model.exit_proj.weight.detach().float().norm().item()
        diff_nrm = (model.lm_head.weight.detach().float() - model.exit_proj.weight.detach().float()).norm().item()
    print("[sanity] head parity:", {"||lm||": lm_nrm, "||exit||": exit_nrm, "||lm-exit||": diff_nrm}, flush=True)

    # W&B init after model is ready (so watch can attach)
    run = init_wandb(args)
    wandb_log({"sanity/head_parity_lm": lm_nrm,
               "sanity/head_parity_exit": exit_nrm,
               "sanity/head_parity_diff": diff_nrm}, step=0)
    wandb_watch_model(model, log_freq=25)

    train_bestcase_kl_rl(
        model, tok, prompts_train, prompts_eval,
        steps=args.steps, rollout_len=args.rollout, batch_size=args.batch_size,
        lr_exit=args.lr_exit, lr_lora=args.lr_lora, temperature=args.temperature,
        ce_weight=args.ce_weight, eval_every=args.eval_every,
        warmup_kl=args.warmup_kl, ramp_steps=args.ramp_steps,
        kl_min=args.kl_min, pg_max=args.pg_max, ent_weight=args.ent_weight,
        outdir=args.outdir, accepted_only_flag=args.accepted_only,
        debug_dump_every=args.debug_dump_every, debug_topk=args.debug_topk,
        max_fro=args.max_fro, max_fro_ratio=args.max_fro_ratio,
        quiet_eval=args.quiet_eval
    )

    if run is not None:
        try:
            run.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()
