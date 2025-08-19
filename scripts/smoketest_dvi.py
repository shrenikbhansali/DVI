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

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc, math, random, argparse, json, time, copy
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, __version__ as transformers_ver

# repo-local
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM
from kangaroo.sgp_lora import inject_dual_lora, enable_lora_grads
from training.buffer import ReplayBuffer

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

# -------------------- KV helpers --------------------

@torch.inference_mode()
def prime_kv_full(spec: EarlyExitLlamaForCausalLM, input_ids: torch.Tensor):
    spec.past_key_values = None
    h = spec.forward_draft_or_large_model(in_tokens_small=input_ids)
    _ , _ = spec.forward_draft_or_large_model(in_features_large=h)

@torch.inference_mode()
def advance_kv_with_committed(spec: EarlyExitLlamaForCausalLM, token_ids: torch.Tensor):
    h = spec.forward_draft_or_large_model(in_tokens_small=token_ids)
    _ , _ = spec.forward_draft_or_large_model(in_features_large=h)

def _top1(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1, keepdim=True)[0,0].item())

# -------------------- evaluation (acceptance metric) --------------------

@torch.inference_mode()
def eval_acceptance(spec: EarlyExitLlamaForCausalLM, tok: AutoTokenizer, prompts: List[str],
                    rollout_len: int, steps_per_prompt: int = 1,
                    dump_debug: bool = False, dump_path: Optional[str] = None, topk: int = 5) -> Tuple[float, Dict[int,float]]:
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
            v_top1   = _top1(v_logits)

            # side-effect-free probe of drafter
            pkv_backup = [None if pkv is None else tuple(t.clone() for t in pkv)
                          for pkv in (spec.past_key_values or [])]
            d_hidden   = spec.forward_draft_or_large_model(in_tokens_small=ids_last)  # [B,1,H]
            # --- patched: pre-norm + scale before exit head ---
            h = d_hidden.squeeze(1).float()
            if hasattr(spec, "exit_pre_norm") and spec.exit_pre_norm is not None:
                h = spec.exit_pre_norm(h)
            d_logits = spec.exit_proj(h)
            if hasattr(spec, "exit_logit_scale"):
                d_logits = spec.exit_logit_scale * d_logits
            # ---------------------------------------------------

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

            spec.past_key_values = pkv_backup
            v_token = torch.tensor([[v_top1]], device=ids_last.device, dtype=ids_last.dtype)
            advance_kv_with_committed(spec, v_token)
            ids_last = v_token

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

    # --- patched: add pre-norm + learnable global logit scale ---
    # Try to clone the model's final RMSNorm if available; otherwise fallback to LayerNorm
    try:
        base_norm = model.model.norm  # Llama final RMSNorm
        model.exit_pre_norm = copy.deepcopy(base_norm).to(w.device)
    except Exception:
        model.exit_pre_norm = nn.LayerNorm(w.shape[1], elementwise_affine=True, device=w.device)
    for p in model.exit_pre_norm.parameters():
        p.requires_grad = True

    model.exit_logit_scale = nn.Parameter(torch.tensor(1.0, device=w.device))
    # -------------------------------------------------------------

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
      - RL surrogate: maximize E[log π_s(v)]  => loss_pg = -E[log π_s(v)]   (patched)
      - KL(teacher -> student) with softened teacher
      - CE: anchor on teacher top-1 (stabilizer)
      - ENT: optional entropy bonus (maximize entropy)
    """
    dev = next(model.parameters()).device
    hidden  = batch["hidden"].to(dev)          # [B, H]
    vlogits = batch["vlogits"].to(dev)         # [B, V] (fp16 ok)
    tokens  = batch["token"].to(dev).view(-1)  # [B]

    # --- patched: pre-norm + scale ---
    h = hidden.float()
    if hasattr(model, "exit_pre_norm") and model.exit_pre_norm is not None:
        h = model.exit_pre_norm(h)
    slogits = model.exit_proj(h)               # [B, V] fp32
    if hasattr(model, "exit_logit_scale"):
        slogits = model.exit_logit_scale * slogits
    # ---------------------------------

    slogp   = F.log_softmax(slogits, dim=-1)
    sp      = slogp.exp()

    # RL (expected acceptance, log-prob surrogate)
    pi_v = sp.gather(1, tokens.view(-1,1)).squeeze(1)  # [B]
    loss_pg = -torch.log(pi_v.clamp_min(1e-8)).mean()  # patched

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
                    debug_out: Optional[List[Dict]] = None, topk: int = 5):
    spec.eval()
    dev = next(spec.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(dev)
    prime_kv_full(spec, enc["input_ids"])
    last = enc["input_ids"][:, -1:]

    for _ in range(steps):
        v_logits = spec.verifier_logits_for_next(last)
        if not torch.isfinite(v_logits).all():
            break
        v_top1   = _top1(v_logits)

        # probe drafter without state mutation
        pkv_backup = [None if pkv is None else tuple(t.clone() for t in pkv)
                      for pkv in (spec.past_key_values or [])]
        d_hidden   = spec.forward_draft_or_large_model(in_tokens_small=last)      # [B,1,H]
        # --- patched: pre-norm + scale ---
        h = d_hidden.squeeze(1).float()
        if hasattr(spec, "exit_pre_norm") and spec.exit_pre_norm is not None:
            h = spec.exit_pre_norm(h)
        d_logits   = spec.exit_proj(h)                                            # [B,V]
        if hasattr(spec, "exit_logit_scale"):
            d_logits = spec.exit_logit_scale * d_logits
        # ----------------------------------
        d_top1     = _top1(d_logits) if torch.isfinite(d_logits).all() else -1
        accept_bit = int(d_top1 == v_top1)

        buf.append(
            hidden=d_hidden.squeeze(0).squeeze(0).cpu(),  # [H] (pre-norm is applied in training step)
            token=int(v_top1),                             # teacher top-1 id
            reward=float(accept_bit),                     # acceptance bit (metric)
            conf=0.0,
            vlogits=v_logits.squeeze(0).cpu(),            # teacher logits (for KL)
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

        spec.past_key_values = pkv_backup
        v_token = torch.tensor([[v_top1]], device=last.device, dtype=last.dtype)
        advance_kv_with_committed(spec, v_token)
        last = v_token

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

# -------------------- training loop --------------------

def train_bestcase_kl_rl(model, tok, prompts_train, prompts_eval,
                         steps: int, rollout_len: int, batch_size: int,
                         lr_exit: float, lr_lora: float, temperature: float,
                         ce_weight: float, eval_every: int,
                         warmup_kl: int, ramp_steps: int,
                         kl_min: float, pg_max: float, ent_weight: float,
                         outdir: str, accepted_only_flag: bool,
                         debug_dump_every: int, debug_topk: int,
                         max_fro: float, max_fro_ratio: float):

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

    print("\n[e2e] measuring acceptance (pre-train)…", flush=True)
    free_cuda("(pre-eval)")
    acc0, ctar0 = eval_acceptance(
        model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
        dump_debug=True, dump_path=samples_path, topk=debug_topk
    )
    print(f"[e2e] PRE  | acc={acc0:.3f} | CTAR1={ctar0[1]:.3f} CTAR2={ctar0[2]:.3f} CTAR3={ctar0[3]:.3f} CTAR4={ctar0[4]:.3f}", flush=True)

    ptr = 0
    for g in range(steps):
        model.eval()  # rollout in eval mode
        p = prompts_train[ptr % len(prompts_train)]; ptr += 1

        # --- data collection timing for tok/s
        t_roll_s = time.time()
        dbg_roll = [] if (debug_dump_every and (g % debug_dump_every == 0)) else None
        rollout_collect(model, tok, p, buf, steps=rollout_len, debug_out=dbg_roll, topk=debug_topk)
        t_roll_e = time.time()
        tok_s = float(rollout_len) / max(1e-6, (t_roll_e - t_roll_s))

        if g % 10 == 0:
            buf_debug(buf, k=16)

        # wait for enough data
        if len(buf) < batch_size:
            print(f"[train] step {g:04d} | {phase_of_step(g, warmup_kl, ramp_steps):>10} | "
                  f"buf={len(buf)}/{batch_size} | tok/s≈{tok_s:.2f} | waiting for batch…", flush=True)
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
        t_trn_s = time.time()
        out = one_mixed_step(
            model, opt, batch,
            temperature=temperature, clip=1.0,
            ce_weight=ce_weight, lam_pg=lam_pg, lam_kl=lam_kl,
            ent_weight=ent_weight,
            init_fro=init_fro, max_fro=max_fro, max_fro_ratio=max_fro_ratio
        )
        t_trn_e = time.time()
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
                  f"grad {out['grad']} -> restoring exit head weights & clearing opt",
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
            f"grad {out['grad']:6.2f} | tok/s≈{tok_s:.2f} | step_time {step_time*1e3:6.1f} ms | "
            f"π_s(v)≈{out['pi_v']:.4f} | std_s={std_s:.3f} std_t={std_t:.3f} | "
            f"||exit||={exit_norm:.2f} | roll_acc≈{accept_roll:.4f} | "
            f"buf_size={buf_size} acc_rate≈{buf_rate:.3f} | λ_pg={lam_pg:.3f} λ_kl={lam_kl:.3f} | accepted_only={accepted_only}",
            flush=True
        )

        # JSONL metrics dump
        metrics = dict(
            step=g, phase=phase,
            loss=out["loss"], kl=out["kl"], pg=out["pg"], ce=out["ce"], ent=out["ent"],
            c_pg=out["c_pg"], c_kl=out["c_kl"], c_ce=out["c_ce"], c_ent=out["c_ent"],
            grad=float(out["grad"]), tok_s=tok_s, step_time_ms=step_time*1e3,
            pi_s_v=out["pi_v"], std_s=std_s, std_t=std_t,
            exit_norm=exit_norm,
            roll_acc=accept_roll, buf_size=buf_size, buf_acc_rate=buf_rate,
            lam_pg=lam_pg, lam_kl=lam_kl, accepted_only=accepted_only,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

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
                dump_debug=True, dump_path=samples_path, topk=debug_topk
            )
            print(f"[e2e] MID  | step {g+1:04d} | acc={acc_mid:.3f} | "
                  f"CTAR1={ctar_mid[1]:.3f} CTAR2={ctar_mid[2]:.3f} CTAR3={ctar_mid[3]:.3f} CTAR4={ctar_mid[4]:.3f}",
                  flush=True)

    # Final eval — free training state first
    del opt
    del buf
    free_cuda("(pre post-train eval)")
    model.eval()

    print("\n[e2e] measuring acceptance (post-train)…", flush=True)
    acc1, ctar1 = eval_acceptance(
        model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
        dump_debug=True, dump_path=samples_path, topk=debug_topk
    )
    print(f"[e2e] POST | acc={acc1:.3f} | Δ={acc1-acc0:+.3f} | "
          f"CTAR1={ctar1[1]:.3f} CTAR2={ctar1[2]:.3f} CTAR3={ctar1[3]:.3f} CTAR4={ctar1[4]:.3f}", flush=True)

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--early-layer", type=int, default=24)  # patched: later default exit
    ap.add_argument("--train-prompts", type=int, default=256)
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

    # KL->RL schedule (patched: shorter warmup & ramp)
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

    train_bestcase_kl_rl(
        model, tok, prompts_train, prompts_eval,
        steps=args.steps, rollout_len=args.rollout, batch_size=args.batch_size,
        lr_exit=args.lr_exit, lr_lora=args.lr_lora, temperature=args.temperature,
        ce_weight=args.ce_weight, eval_every=args.eval_every,
        warmup_kl=args.warmup_kl, ramp_steps=args.ramp_steps,
        kl_min=args.kl_min, pg_max=args.pg_max, ent_weight=args.ent_weight,
        outdir=args.outdir, accepted_only_flag=args.accepted_only,
        debug_dump_every=args.debug_dump_every, debug_topk=args.debug_topk,
        max_fro=args.max_fro, max_fro_ratio=args.max_fro_ratio
    )

if __name__ == "__main__":
    main()
