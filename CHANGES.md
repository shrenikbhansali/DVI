# Changelog

## Smoketest modularisation

- Ported the monolithic `scripts/smoketest_dvi.py` into reusable modules:
  - Model assembly → `training/modeling.py`
  - KV hygiene → `training/kv.py`
  - Rollout & buffer helpers → `training/rollout.py`
  - Loss functions → `training/objectives.py`
  - Scheduling → `training/schedule.py`
  - W&B utilities → `training/logging.py`
  - General utilities → `training/utils.py`
  - Acceptance eval/CTARs → `evaluation/acceptance.py`
- Added `train_bestcase.py` as the consolidated trainer using the above modules.
- Replaced `scripts/smoketest_dvi.py` with a thin wrapper around the new trainer.


## Multi-Token Speculation During Training

### Before (k=1)

Previously, training used **single-token rollouts**:

* Draft proposes **1 token**.
* Verifier accepts/rejects.
* That single bit updates the draft.

**Problem:** inference uses **multi-token speculation** (e.g. draft k=4), but training didn’t. This mismatch meant the draft wasn’t trained under the same dynamics it faces during decoding.

### Proposal: Train with `train-k-spec > 1`

Now, training supports **multi-token speculative rollouts**:

1. Draft proposes **k tokens** in parallel.
2. Verifier accepts/rejects each.
3. All k outcomes are logged and used as reinforcement signals.

This mirrors inference exactly.

### Why It Matters

* **Closer match to inference:** training = deployment.
* **More feedback per step:** k accept/reject signals instead of 1.
* **Online reinforcement:** every token contributes a reward.
* **Better correlation to runtime:** acceptance rate improvements → speedups.

### Feasibility & Trade-offs

* **Feasible:** rollout collector extended to handle k>1.
* **Efficiency:** rollout cost grows with k; tune `rollout_len × train-k-spec`.
* **Noise:** large k may yield many rejections early; mitigate with baselines or accepted-only sampling.
* **Hyperparams:** start with `train-k-spec=4, rollout=2`.

---
