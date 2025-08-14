import math


def exp_decay_lambda(step: int, lambda0: float, tau: float, min_lambda: float = 0.0) -> float:
    lam = lambda0 * math.exp(-step / max(tau, 1e-6))
    return max(min_lambda, min(1.0, lam))


def cosine_decay_lambda(step: int, total_steps: int, lambda0: float, min_lambda: float = 0.0) -> float:
    frac = min(1.0, step / max(total_steps, 1))
    lam = min_lambda + (lambda0 - min_lambda) * 0.5 * (1.0 + math.cos(math.pi * frac))
    return max(min_lambda, min(1.0, lam))
