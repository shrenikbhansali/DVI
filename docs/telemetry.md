# DVI Alignment Telemetry

`AlignTelemetryParams` provides lightweight diagnostics for the drafter–verifier
alignment in both training and runtime experiments. Telemetry is **off by
default** and is controlled entirely through CLI flags in `train_bestcase.py`.

Policy-gradient training can be enabled with `--use-policy-grad` and is further
configured via `--rl-weight`, `--pg-baseline-ema`, `--kl-beta0`,
`--kl-beta-min` and `--kl-anneal-steps`. When enabled, telemetry JSONs also
record averaged advantages/kl terms and the drafting `eta`/`prefix_hist` per
block.

When enabled it can:

* emit up to `--telemetry-prints-budget` concise lines to stdout
  (one per block),
* write at most `--telemetry-max-blocks` JSON files and optional `.pt`
  tensor blobs to `--telemetry-save-dir`, and
* optionally search both offsets `{0,+1}` when computing accepted prefix
  length with `--telemetry-auto-offset 1`.

## Example usage

### Baseline diagnosis (auto-offset off)

```bash
CUDA_VISIBLE_DEVICES=0 python train_bestcase.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --early-layer 4 \
  --steps 60 --rollout 2 --batch-size 64 \
  --lr-exit 5e-4 --lr-lora 5e-5 \
  --train-k-spec 2 --spec-train-greedy \
  --outdir runs/k2_telemetry \
  --eval-every 30 \
  --spec-k-max 2 \
  --time-prompts 16 --time-max-new-tokens 64 --time-repeats 1 \
  --timing-greedy --quiet-eval \
  --telemetry-debug 1 \
  --telemetry-prints-budget 6 \
  --telemetry-dump-tensors 0 \
  --telemetry-max-blocks 12 \
  --telemetry-save-dir runs/k2_telemetry/align_dumps \
  --telemetry-auto-offset 0
```

### Mitigation trial (auto-offset on)

```bash
--telemetry-auto-offset 1
```

## Inspecting dumps

Telemetry files are stored in the directory passed via `--telemetry-save-dir`
and are named `{run_id}_{phase}_step####.json`.  Each JSON file contains a
`diag` block with fields such as `match_0`, `match_p1`, `best_offset` and
`accept_len_default`, together with KV-cache lengths before/after the block.
When policy-gradient training is enabled the dumps also contain a
`diag_extra` section with `eta`, a per-block `prefix_hist`, and averaged
`policy_mean_advantage`/`policy_mean_kl` statistics.
These additional fields help correlate runtime behaviour with policy
updates.  A couple of sample JSONs are usually enough to diagnose
alignment issues.
