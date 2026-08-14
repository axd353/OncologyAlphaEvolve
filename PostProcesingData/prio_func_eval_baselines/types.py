from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineResult:
    auc_roc: float
    heldout_scores: tuple[float, ...]