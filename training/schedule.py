"""Scheduling helpers for KL→RL mix."""

def mix_schedule(step, warmup_kl: int, ramp: int, kl_min: float, pg_max: float):
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
