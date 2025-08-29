"""DVI best-case trainer with KL→RL schedule.

Mapping of the original smoketest script to modular components:

* Model assembly & dual-LoRA enablement → ``training.modeling``
* KV hygiene helpers → ``training.kv``
* Rollout & buffer operations → ``training.rollout``
* Losses & schedules → ``training.objectives`` and ``training.schedule``
* Evaluation (acceptance & CTARs) → ``evaluation.acceptance``
* W&B + JSONL logging → ``training.logging``
* General utilities (seed, prompts, env dump) → ``training.utils``
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import time
import gc
from typing import List

import torch

from training.utils import (
    set_seed,
    ensure_dirs,
    env_dump,
    build_prompts_from_alpaca,
    free_cuda,
    _cuda_sync,
    # added for compression + timing
    count_transformer_layers,
    theoretical_compression,
)
from training.mem import deep_kv_purge, timing_trace
from training.modeling import prepare_dvi_trainable, build_optimizer
from training.kv import estimate_kv_cache
from training.rollout import rollout_collect, rollout_collect_k_spec, buf_debug
from training.objectives import one_mixed_step
from training.schedule import mix_schedule, phase_of_step
from training.logging import init_wandb, wandb_watch_model, wandb_log, WANDB_DEFAULT_ENTITY, WANDB_DEFAULT_PROJECT
from training.buffer import ReplayBuffer
from evaluation.acceptance import eval_acceptance


def train_bestcase_kl_rl(model, tok, prompts_train: List[str], prompts_eval: List[str],
                        steps: int, rollout_len: int, batch_size: int,
                        lr_exit: float, lr_lora: float, temperature: float,
                        ce_weight: float, eval_every: int,
                        warmup_kl: int, ramp_steps: int,
                        kl_min: float, pg_max: float, ent_weight: float,
                        outdir: str, accepted_only_flag: bool,
                        debug_dump_every: int, debug_topk: int,
                        max_fro: float, max_fro_ratio: float,
                        quiet_eval: bool,
                        early_layer: int,
                        train_k_spec: int = 1,
                        spec_train_greedy: bool = False,
                        spec_train_temp: float = 1.0,
                        ce_mask_by_reward: bool = False,
                        kl_warmup_scale: float = 1.0,
                        eval_k_max: int = None,
                        ):
    metrics_path = os.path.join(outdir, "logs", "train_metrics.jsonl")
    samples_path = os.path.join(outdir, "logs", "rollout_samples.jsonl")

    buf = ReplayBuffer(capacity=max(4096, batch_size * rollout_len * 8), device=torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        init_fro = float(torch.linalg.vector_norm(model.exit_proj.weight).item())
        lm_norm = float(torch.linalg.vector_norm(model.lm_head.weight).item())
    print(f"[sanity] head norms: ||lm||={lm_norm:.3f} ||exit||={init_fro:.3f}", flush=True)
    wandb_log({"sanity/lm_head_norm": lm_norm, "sanity/exit_head_norm": init_fro}, step=0)

    # layer count for compression estimates
    total_layers = count_transformer_layers(model)
    print(f"[info] total decoder layers detected: {total_layers}", flush=True)

    eval_k_max = eval_k_max or train_k_spec
    if eval_k_max > train_k_spec:
        print(f"[warn] eval_k_max {eval_k_max} > train_k_spec {train_k_spec}", flush=True)

    print("\n[e2e] measuring acceptance (pre-train)…", flush=True)
    free_cuda("(pre-eval)")
    acc0, ctar0 = eval_acceptance(
        model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
        dump_debug=True, dump_path=samples_path, topk=debug_topk, quiet=quiet_eval, k_max=eval_k_max
    )
    ctar_msg = " ".join([f"CTAR{w}={ctar0.get(w,0):.3f}" for w in range(1, eval_k_max + 1)])
    print(f"[e2e] PRE  | acc={acc0:.3f} | {ctar_msg}", flush=True)
    log_dict = {"eval/pre/acc": acc0}
    for w in range(1, eval_k_max + 1):
        log_dict[f"eval/pre/ctar{w}"] = ctar0.get(w, 0)
    wandb_log(log_dict, step=0)

    # Build optimizer after any dtype casting that may occur during evaluation.
    opt = build_optimizer(model, lr_exit=lr_exit, lr_lora=lr_lora, wd_exit=1e-2, wd_lora=0.0)

    ptr = 0
    tokens_total = 0
    ema_tok_s = None

    for g in range(steps):
        model.eval()
        p = prompts_train[ptr % len(prompts_train)]
        ptr += 1
        dbg_roll = [] if (debug_dump_every and (g % debug_dump_every == 0)) else None
        _cuda_sync()
        t_roll_s = time.perf_counter()
        if train_k_spec and train_k_spec > 1:
            n_collected = rollout_collect_k_spec(
                model,
                tok,
                p,
                buf,
                steps=rollout_len,
                k=train_k_spec,
                greedy=spec_train_greedy,
                temperature=spec_train_temp,
                debug_out=dbg_roll,
                topk=debug_topk,
            )
        else:
            n_collected = rollout_collect(
                model, tok, p, buf, steps=rollout_len, debug_out=dbg_roll, topk=debug_topk
            )
        _cuda_sync()
        t_roll_e = time.perf_counter()
        elapsed = max(1e-6, (t_roll_e - t_roll_s))
        tok_s = float(n_collected) / elapsed
        tokens_total += n_collected
        ema_tok_s = tok_s if ema_tok_s is None else (0.9 * ema_tok_s + 0.1 * tok_s)

        kv_bytes, kv_seq = estimate_kv_cache(model)
        kv_mb = kv_bytes / (1024 ** 2)
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            mem_max = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            mem_alloc = mem_reserved = mem_max = 0.0

        if g % 10 == 0:
            buf_debug(buf, k=16)

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

        lam_pg, lam_kl = mix_schedule(g, warmup_kl, ramp_steps, kl_min, pg_max)
        lam_kl *= kl_warmup_scale
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

        model.train()
        _cuda_sync()
        t_trn_s = time.perf_counter()
        out = one_mixed_step(
            model, opt, batch,
            temperature=temperature, clip=1.0,
            ce_weight=ce_weight, lam_pg=lam_pg, lam_kl=lam_kl,
            ent_weight=ent_weight,
            init_fro=init_fro, max_fro=max_fro, max_fro_ratio=max_fro_ratio,
            ce_mask_by_reward=ce_mask_by_reward,
        )
        _cuda_sync()
        t_trn_e = time.perf_counter()
        step_time = (t_trn_e - t_trn_s)

        accept_roll = float(batch["reward"].float().mean().item())
        # per-step compression & speedup estimate (frequent logging)
        comp_ratio, speedup_est = theoretical_compression(accept_roll, early_layer, total_layers)

        buf_size = len(buf)
        buf_acc = buf.accepted_count()
        buf_rate = (buf_acc / buf_size) if buf_size else 0.0
        with torch.no_grad():
            exit_norm = float(torch.linalg.vector_norm(model.exit_proj.weight).item())

        if not out["ok"] or not math.isfinite(out["loss"]) or not math.isfinite(out["kl"]):
            print(f"[train][nan] step {g:04d} | {phase:>10} | "
                  f"loss {out['loss']} | KL {out['kl']} | PG {out['pg']} | CE {out['ce']} | "
                  f"grad {out['grad']} -> clearing opt state", flush=True)
            opt.state.clear()
            free_cuda("(nan-recovery)")
            continue

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
            f"KV={kv_mb:.2f} MB (seq≈{kv_seq}) | CUDA alloc={mem_alloc:.1f} MB | "
            f"comp≈{comp_ratio:.3f} (speedup≈{speedup_est:.2f}×)",
            flush=True,
        )

        metrics = dict(
            step=g, phase=phase,
            loss=out["loss"], kl=out["kl"], pg=out["pg"], ce=out["ce"], ent=out["ent"],
            c_pg=out["c_pg"], c_kl=out["c_kl"], c_ce=out["c_ce"], c_ent=out["c_ent"],
            grad=float(out["grad"]), tok_s=tok_s, tok_s_ema=ema_tok_s, step_time_ms=step_time * 1e3,
            pi_s_v=out["pi_v"], std_s=std_s, std_t=std_t,
            exit_norm=exit_norm,
            roll_acc=accept_roll, buf_size=buf_size, buf_acc_rate=buf_rate,
            lam_pg=lam_pg, lam_kl=lam_kl, accepted_only=accepted_only,
            kv_bytes=kv_bytes, kv_mb=kv_mb, kv_seq_est=kv_seq,
            cuda_mem_alloc_mb=mem_alloc, cuda_mem_reserved_mb=mem_reserved, cuda_mem_max_alloc_mb=mem_max,
            tokens_total=tokens_total,
            # added metrics
            comp_ratio_est=comp_ratio,
            comp_speedup_est=speedup_est,
            total_layers=total_layers,
            early_layer=early_layer,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        wandb_log({
            "loss/total": out["loss"], "loss/pg": out["pg"], "loss/kl": out["kl"],
            "loss/ce": out["ce"], "loss/ent": out["ent"],
            "contrib/pg": out["c_pg"], "contrib/kl": out["c_kl"],
            "contrib/ce": out["c_ce"], "contrib/ent": out["c_ent"],
            "grad/norm": float(out["grad"]),
            "roll/tok_s": tok_s, "roll/tok_s_ema": ema_tok_s,
            "time/step_ms": step_time * 1e3,
            "policy/pi_s_v": out["pi_v"], "logits/std_s": std_s, "logits/std_t": std_t,
            "head/exit_norm": exit_norm,
            "roll/accept_rate_batch": accept_roll,
            "buf/size": buf_size, "buf/accept_rate": buf_rate,
            "schedule/lam_pg": lam_pg, "schedule/lam_kl": lam_kl,
            "sampling/accepted_only": int(accepted_only),
            "kv/bytes": kv_bytes, "kv/MB": kv_mb, "kv/seq_est": kv_seq,
            "cuda/mem_alloc_MB": mem_alloc, "cuda/mem_reserved_MB": mem_reserved, "cuda/mem_max_alloc_MB": mem_max,
            "tokens/total": tokens_total,
            # added: frequent compression logging
            "comp/ratio_est": comp_ratio,
            "comp/speedup_est": speedup_est,
        }, step=g)

        if dbg_roll:
            with open(samples_path, "a") as f:
                for rec in dbg_roll:
                    f.write(json.dumps({"step": g, **rec}) + "\n")

        if (g + 1) % eval_every == 0:
            model.eval()
            free_cuda("(pre mid-train eval)")
            print("[e2e] measuring acceptance (mid-train)…", flush=True)
            acc_mid, ctar_mid = eval_acceptance(
                model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
                dump_debug=True, dump_path=samples_path, topk=debug_topk, quiet=quiet_eval, k_max=eval_k_max
            )
            ctar_msg = " ".join([f"CTAR{w}={ctar_mid.get(w,0):.3f}" for w in range(1, eval_k_max + 1)])
            print(f"[e2e] MID  | step {g+1:04d} | acc={acc_mid:.3f} | {ctar_msg}", flush=True)
            log_mid = {"eval/mid/acc": acc_mid}
            for w in range(1, eval_k_max + 1):
                log_mid[f"eval/mid/ctar{w}"] = ctar_mid.get(w, 0)
            wandb_log(log_mid, step=g + 1)

    del opt
    del buf
    free_cuda("(pre post-train eval)")
    model.eval()
    print("\n[e2e] measuring acceptance (post-train)…", flush=True)
    acc1, ctar1 = eval_acceptance(
        model, tok, prompts_eval, rollout_len, steps_per_prompt=1,
        dump_debug=True, dump_path=samples_path, topk=debug_topk, quiet=quiet_eval, k_max=eval_k_max
    )
    ctar_msg = " ".join([f"CTAR{w}={ctar1.get(w,0):.3f}" for w in range(1, eval_k_max + 1)])
    print(f"[e2e] POST | acc={acc1:.3f} | Δ={acc1-acc0:+.3f} | {ctar_msg}", flush=True)
    log_post = {"eval/post/acc": acc1,
                "eval/delta_acc": (acc1 - acc0)}
    for w in range(1, eval_k_max + 1):
        log_post[f"eval/post/ctar{w}"] = ctar1.get(w, 0)
    wandb_log(log_post, step=steps)

    deep_kv_purge(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--early-layer", type=int, default=16, help="Early-exit split layer k (prefer deeper, e.g. 16 or 24 on 7B)")
    ap.add_argument("--train-prompts", type=int, default=512)
    ap.add_argument("--eval-prompts", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rollout", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-exit", type=float, default=5e-4)
    ap.add_argument("--lr-lora", type=float, default=5e-5)
    ap.add_argument("--lora-s-rank", type=int, default=8)
    ap.add_argument("--lora-v-rank", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.10)
    ap.add_argument("--ce-mask-by-reward", action="store_true")
    ap.add_argument("--ent-weight", type=float, default=0.00)
    ap.add_argument("--warmup-kl", type=int, default=40)
    ap.add_argument("--ramp-steps", type=int, default=80)
    ap.add_argument("--kl-min", type=float, default=0.05)
    ap.add_argument("--pg-max", type=float, default=1.0)
    ap.add_argument("--kl-warmup-scale", type=float, default=1.0)
    ap.add_argument("--outdir", type=str, default="minbench_out")
    ap.add_argument("--debug-dump-every", type=int, default=25)
    ap.add_argument("--debug-topk", type=int, default=5)
    ap.add_argument("--accepted-only", action="store_true")
    ap.add_argument("--max-fro", type=float, default=0.0)
    ap.add_argument("--max-fro-ratio", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    # timing / compression CLI
    ap.add_argument("--time-prompts", type=int, default=64,
                    help="#prompts for post-train walltime measurement")
    ap.add_argument("--time-max-new-tokens", type=int, default=64,
                    help="new tokens to generate for timing")
    ap.add_argument("--time-repeats", type=int, default=3,
                    help="repeat timing and take median")
    ap.add_argument("--timing-greedy", action="store_true",
                    help="use greedy decoding for walltime timing (default: sampling)")
    ap.add_argument("--spec-draft-k", type=int, default=4, help="block size for self-spec drafting")
    ap.add_argument("--train-k-spec", type=int, default=1, help="k tokens per speculative draft during training")
    ap.add_argument("--spec-train-greedy", action="store_true", help="use greedy drafting during spec training")
    ap.add_argument("--spec-train-temp", type=float, default=1.0, help="temperature for spec training drafting")
    ap.add_argument("--eval-k-max", type=int, default=None, help="max CTAR depth to report")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-entity", type=str, default=WANDB_DEFAULT_ENTITY)
    ap.add_argument("--wandb-project", type=str, default=WANDB_DEFAULT_PROJECT)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--quiet-eval", action="store_true")
    args = ap.parse_args()
    args.eval_k_max = args.eval_k_max or args.train_k_spec

    ensure_dirs(args.outdir)
    env_dump(args.outdir)
    set_seed(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[e2e] loading prompts…", flush=True)
    prompts_train = build_prompts_from_alpaca(args.train_prompts)
    prompts_eval  = build_prompts_from_alpaca(args.eval_prompts)

    print("[e2e] loading model…", flush=True)
    model = prepare_dvi_trainable(
        args.model_id,
        args.early_layer,
        rank_s=args.lora_s_rank,
        rank_v=args.lora_v_rank,
        dtype=torch.float16,
    )

    with torch.no_grad():
        lm_nrm   = model.lm_head.weight.detach().float().norm().item()
        exit_nrm = model.exit_proj.weight.detach().float().norm().item()
        diff_nrm = (model.lm_head.weight.detach().float() - model.exit_proj.weight.detach().float()).norm().item()
    print("[sanity] head parity:", {"||lm||": lm_nrm, "||exit||": exit_nrm, "||lm-exit||": diff_nrm}, flush=True)

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
        quiet_eval=args.quiet_eval,
        early_layer=args.early_layer,
        train_k_spec=args.train_k_spec,
        spec_train_greedy=args.spec_train_greedy,
        spec_train_temp=args.spec_train_temp,
        ce_mask_by_reward=args.ce_mask_by_reward,
        kl_warmup_scale=args.kl_warmup_scale,
        eval_k_max=args.eval_k_max,
    )
    deep_kv_purge(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------- Post-train walltime timing --------
    print("\n[e2e] timing DVI (SPEC) vs baseline…", flush=True)
    from training.utils import measure_generate_walltime, theoretical_compression, count_transformer_layers
    timing_prompts = build_prompts_from_alpaca(args.time_prompts)

    model.eval()
    dvi_device = next(model.parameters()).device
    dvi_dtype = next(model.parameters()).dtype
    total_layers = count_transformer_layers(model)
    print(
        f"[time] decode_mode=DVI(SPEC); device={dvi_device}; dtype={dvi_dtype}; temperature={args.temperature}",
        flush=True,
    )

    dvi_res = measure_generate_walltime(
        model, tok, timing_prompts,
        max_new_tokens=args.time_max_new_tokens,
        greedy=args.timing_greedy,
        repeats=args.time_repeats,
        use_dvi_spec=True,
        draft_k=args.spec_draft_k,
        temperature=max(1e-6, args.temperature),
        early_layer_override=args.early_layer,
        quiet=True,
    )
    dvi_time, spec_metrics = dvi_res
    print(f"[time] DVI(SPEC) generate: {dvi_time:.3f}s", flush=True)
    if spec_metrics is None:
        spec_metrics = {}
    acc_rt = float(spec_metrics.get("spec/accept_rate", 0.0))
    comp_rt, _ = theoretical_compression(acc_rt, args.early_layer, total_layers)
    print(f"[spec] runtime_accept_rate={acc_rt:.3f} | runtime_comp_est≈{comp_rt:.3f}", flush=True)

    deep_kv_purge(model)
    try:
        import wandb as _wandb
        if _wandb is not None:
            try:
                _wandb.unwatch(model)
            except Exception:
                pass
    except Exception:
        pass
    if run is not None:
        try:
            run.finish()
            timing_trace("wandb run finished before baseline load")
        except Exception:
            pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_cuda("(before baseline timing)")

    from transformers import AutoModelForCausalLM
    baseline = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dvi_dtype
    ).to(dvi_device)
    baseline.eval()
    if getattr(baseline.config, "use_cache", True) is False:
        baseline.config.use_cache = True

    deep_kv_purge(baseline)

    print(
        f"[time] decode_mode=BASELINE(vanilla); device={dvi_device}; dtype={dvi_dtype}; temperature={args.temperature}",
        flush=True,
    )
    base_time = measure_generate_walltime(
        baseline,
        tok,
        timing_prompts,
        max_new_tokens=args.time_max_new_tokens,
        greedy=args.timing_greedy,
        repeats=args.time_repeats,
        use_dvi_spec=False,
        temperature=max(1e-6, args.temperature),
    )
    print(f"[time] Baseline generate: {base_time:.3f}s", flush=True)

    speedup_wall = (base_time / dvi_time) if dvi_time > 0 else float("inf")
    print(f"[time] Walltime speedup (baseline/DVI SPEC): {speedup_wall:.3f}×", flush=True)

    wandb_log({
        "speed/walltime_dvi_spec_s": dvi_time,
        "speed/walltime_base_s": base_time,
        "speed/speedup_walltime_spec": speedup_wall,
        "spec/proposed": spec_metrics.get("spec/proposed", 0.0),
        "spec/accepted": spec_metrics.get("spec/accepted", 0.0),
        "spec/committed": spec_metrics.get("spec/committed", 0.0),
        "spec/accept_rate": spec_metrics.get("spec/accept_rate", 0.0),
        "spec/deep_tokens": spec_metrics.get("spec/deep_tokens", 0.0),
        "spec/deep_to_commit": spec_metrics.get("spec/deep_to_commit", 0.0),
        "comp/runtime_est": comp_rt,
        "comp/theoretical_from_train": theoretical_compression(
            spec_metrics.get("spec/accept_rate", 0.0), args.early_layer, total_layers
        )[0],
    }, step=args.steps)

    deep_kv_purge(baseline)
    del baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_cuda("(timing done)")


if __name__ == "__main__":
    main()
