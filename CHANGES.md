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
