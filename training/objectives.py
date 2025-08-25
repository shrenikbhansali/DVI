"""Losses and head regularisation for DVI training."""
import math
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F

__all__ = ["one_mixed_step", "policy_kl_ent_multi_step"]


def _maybe_clamp_exit_head(model, init_fro: float, max_fro: float, max_fro_ratio: float) -> None:
    if (max_fro is None or max_fro <= 0.0) and (max_fro_ratio is None or max_fro_ratio <= 0.0):
        return
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


def _logit_stats(s_logits: torch.Tensor, t_logits: torch.Tensor) -> Tuple[float, float]:
    s = s_logits.float()
    t = t_logits.float()
    return float(s.std().item()), float(t.std().item())


def one_mixed_step(model, opt, batch,
                   temperature=1.0, clip=1.0,
                   ce_weight=0.20,
                   lam_pg=0.0, lam_kl=1.0, ent_weight=0.0,
                   init_fro: float = None, max_fro: float = 0.0, max_fro_ratio: float = 0.0):
    dev = next(model.parameters()).device
    hidden = batch["hidden"].to(dev)
    vlogits = batch["vlogits"].to(dev)
    tokens = batch["token"].to(dev).view(-1)

    h = hidden.float()
    if hasattr(model, "exit_pre_norm") and model.exit_pre_norm is not None:
        h = model.exit_pre_norm(h)
    slogits = model.exit_proj(h)
    if hasattr(model, "exit_logit_scale"):
        slogits = model.exit_logit_scale * slogits

    slogp = F.log_softmax(slogits, dim=-1)
    sp = slogp.exp()

    pi_v = sp.gather(1, tokens.view(-1, 1)).squeeze(1)
    loss_pg = -torch.log(pi_v.clamp_min(1e-8)).mean()

    tlogits = vlogits.float() / float(temperature)
    tlogp = F.log_softmax(tlogits, dim=-1)
    tp = tlogp.exp()
    kl = F.kl_div(input=slogp, target=tp, reduction="batchmean", log_target=False)

    ce = F.nll_loss(slogp, tokens, reduction="mean") if ce_weight > 0.0 else torch.tensor(0.0, device=dev)
    ent = -(sp * slogp).sum(-1).mean()

    loss = lam_pg * loss_pg + lam_kl * kl + ce_weight * ce - ent_weight * ent
    contrib_pg = float((lam_pg * loss_pg).detach().item())
    contrib_kl = float((lam_kl * kl).detach().item())
    contrib_ce = float((ce_weight * ce).detach().item()) if ce_weight > 0.0 else 0.0
    contrib_ent = float((-ent_weight * ent).detach().item()) if ent_weight > 0.0 else 0.0

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

    _maybe_clamp_exit_head(model, init_fro, max_fro, max_fro_ratio)

    std_s, std_t = _logit_stats(slogits.detach(), vlogits.detach())
    return dict(ok=True, loss=float(loss.item()), kl=float(kl.item()), pg=float(loss_pg.item()),
                ce=float(ce.item()), ent=float(ent.item()), grad=float(grad),
                c_pg=contrib_pg, c_kl=contrib_kl, c_ce=contrib_ce, c_ent=contrib_ent,
                pi_v=float(pi_v.mean().item()), std_s_t=(std_s, std_t))


def policy_kl_ent_multi_step(
    step_records: List[Dict[str, torch.Tensor]],
    *,
    kl_weight: float = 1.0,
    ent_weight: float = 0.0,
    baseline: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute multi-step policy gradient with KL and entropy regularisation.

    Each element of ``step_records`` should contain ``logp_exit`` (log prob of
    the taken action), ``logits_exit`` and ``logits_verifier`` for KL,
    and ``reward`` (scalar reward for that position).
    """

    if len(step_records) == 0:
        raise ValueError("step_records must be non-empty")

    pg_terms = []
    kl_terms = []
    ent_terms = []
    for rec in step_records:
        logp = rec["logp_exit"]
        reward = rec["reward"]
        adv = reward - baseline
        pg_terms.append(-adv.detach() * logp)

        if "logits_exit" in rec and "logits_verifier" in rec:
            s_logp = F.log_softmax(rec["logits_exit"], dim=-1)
            t_logp = F.log_softmax(rec["logits_verifier"].detach(), dim=-1)
            kl_terms.append(F.kl_div(s_logp, t_logp.exp(), reduction="batchmean", log_target=False))
            ent_terms.append(-(s_logp.exp() * s_logp).sum(-1).mean())

    pg = torch.stack(pg_terms).mean()
    kl = torch.stack(kl_terms).mean() if kl_terms else torch.tensor(0.0, device=pg.device)
    ent = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=pg.device)
    loss = pg + kl_weight * kl - ent_weight * ent
    return {"loss": loss, "pg": pg, "kl": kl, "ent": ent}
