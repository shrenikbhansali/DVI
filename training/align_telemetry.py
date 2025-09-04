from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json, time, pathlib
import torch


@dataclass
class AlignTelemetryParams:
    # Logging
    debug: int = 0                 # >0 enables concise per-block prints
    prints_budget: int = 3         # max lines to print
    topk: int = 5                  # optional caller-side top-k if needed
    # Dumps
    save_dir: str = "./dvi_align_dumps"
    run_id: Optional[str] = None   # default: timestamp
    max_blocks: int = 5            # cap number of block dumps
    dump_tensors: int = 0          # >0 saves small .pt blobs (sample 0 only)
    # Diagnostics
    gold: int = 0                  # >0 enables optional slow "gold" verifier (if available)
    # Mitigation (DEFAULT OFF so unit tests keep passing)
    auto_offset: int = 0           # >0 uses best of {offset 0, +1} to compute acceptance


class AlignLogger:
    _prints_emitted = 0
    _blocks_dumped = 0

    def __init__(self, cfg: Optional[AlignTelemetryParams] = None):
        self.cfg = cfg or AlignTelemetryParams()
        if not self.cfg.run_id:
            self.cfg.run_id = time.strftime("%Y%m%d-%H%M%S")
        self._save_base = pathlib.Path(self.cfg.save_dir)
        if self.should_dump():
            self._save_base.mkdir(parents=True, exist_ok=True)

    # ---- guards ----
    def enabled(self) -> bool:
        return bool(self.cfg and (self.cfg.debug or self.cfg.dump_tensors or self.cfg.gold))

    def should_log(self) -> bool:
        return bool(self.cfg.debug)

    def should_dump(self) -> bool:
        return bool(self.cfg.max_blocks > 0 and (self.cfg.dump_tensors or self.cfg.debug))

    # ---- helpers ----
    def print_once(self, line: str) -> None:
        if self._prints_emitted < self.cfg.prints_budget and self.should_log():
            print(line, flush=True)
            self._prints_emitted += 1

    @staticmethod
    def kv_len_from_past(past) -> int:
        if not past or past[0] is None:
            return 0
        try:
            return int(past[0][0].shape[2])  # k/v: [B, heads, T, dim]
        except Exception:
            return 0

    @staticmethod
    def compute_diag_metrics(prop_seq: torch.Tensor, deep_argmax: torch.Tensor) -> Dict[str, Any]:
        # prop_seq, deep_argmax: [B,k]
        B, k = prop_seq.shape
        def rate(off: int) -> float:
            L = max(0, k - off)
            if L == 0:
                return 0.0
            d = deep_argmax[:, off:off+L]
            p = prop_seq[:, :L]
            return float((d == p).float().mean().item())
        m00 = rate(0)
        m10 = rate(1)
        best = 0 if m00 >= m10 else 1
        return {"match_0": m00, "match_p1": m10, "best_offset": best}

    # ---- persistence ----
    def _save_json(self, name: str, obj: Dict[str, Any]) -> None:
        if not self.should_dump() or self._blocks_dumped >= self.cfg.max_blocks:
            return
        with open(self._save_base / name, "w") as f:
            json.dump(obj, f, indent=2)

    def _save_pt(self, name: str, obj: Dict[str, Any]) -> None:
        if not self.should_dump() or self._blocks_dumped >= self.cfg.max_blocks:
            return
        torch.save(obj, self._save_base / name)

    # ---- main API ----
    def block_report(
        self,
        *,
        phase: str,                # "spec" or "rollout"
        step_idx: int,
        k: int,
        B: int,
        greedy: bool,
        temperature: float,
        prop_seq: torch.Tensor,                # [B,k]
        deep_logits: Optional[torch.Tensor],   # [B,k,V] or None
        deep_argmax: Optional[torch.Tensor],   # [B,k] or None
        accept_len_default: int,
        kv_len_shallow_before: int,
        kv_len_deep_before: int,
        kv_len_shallow_after: int,
        kv_len_deep_after: int,
        gold: Optional[Dict[str, Any]] = None,
        sample0_tensors: Optional[Dict[str, torch.Tensor]] = None,
        eta: Optional[float] = None,
        prefix_hist: Optional[Any] = None,
        policy_mean_adv: Optional[float] = None,
        policy_mean_kl: Optional[float] = None,
    ) -> None:
        if deep_logits is not None and deep_argmax is None:
            deep_argmax = deep_logits.argmax(dim=-1)

        diag = None
        if deep_argmax is not None:
            try:
                diag = self.compute_diag_metrics(prop_seq, deep_argmax)
            except Exception:
                diag = None

        if self.should_log() and diag is not None:
            self.print_once(
                f"[align/{phase}] step={step_idx} k={k} B={B} "
                f"m00={diag['match_0']:.3f} m+1={diag['match_p1']:.3f} "
                f"best_off={diag['best_offset']} "
                f"acc_len={accept_len_default} "
                f"KV(sh,deep) {kv_len_shallow_before}->{kv_len_shallow_after},"
                f"{kv_len_deep_before}->{kv_len_deep_after}"
            )

        if self.should_dump() and self._blocks_dumped < self.cfg.max_blocks:
            meta = {
                "phase": phase,
                "run_id": self.cfg.run_id,
                "step_idx": int(step_idx),
                "k": int(k),
                "B": int(B),
                "greedy": bool(greedy),
                "temperature": float(temperature),
                "accept_len_default": int(accept_len_default),
                "kv_len_shallow_before": int(kv_len_shallow_before),
                "kv_len_deep_before": int(kv_len_deep_before),
                "kv_len_shallow_after": int(kv_len_shallow_after),
                "kv_len_deep_after": int(kv_len_deep_after),
                "diag": diag,
                "gold": gold,
            }
            if eta is not None:
                meta.setdefault("diag_extra", {})["eta"] = float(eta)
            if prefix_hist is not None:
                meta.setdefault("diag_extra", {})["prefix_hist"] = prefix_hist
            if policy_mean_adv is not None:
                meta.setdefault("diag_extra", {})["policy_mean_advantage"] = float(policy_mean_adv)
            if policy_mean_kl is not None:
                meta.setdefault("diag_extra", {})["policy_mean_kl"] = float(policy_mean_kl)
            self._save_json(f"{self.cfg.run_id}_{phase}_step{step_idx:04d}.json", meta)
            if self.cfg.dump_tensors and sample0_tensors:
                safe = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                        for k, v in sample0_tensors.items()}
                self._save_pt(f"{self.cfg.run_id}_{phase}_step{step_idx:04d}_tensors.pt", safe)
            self._blocks_dumped += 1
