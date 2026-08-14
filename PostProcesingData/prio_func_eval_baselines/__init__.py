from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from PostProcesingData.prio_func_eval_baselines.independent_learning_scheme import (
    evaluate_independent_learning_scheme,
)
from PostProcesingData.prio_func_eval_baselines.mixture_learning import evaluate_mixture_learning
from PostProcesingData.prio_func_eval_baselines.tl_gdes import evaluate_tl_gdes
from PostProcesingData.prio_func_eval_baselines.tl_pr import evaluate_tl_pr
from PostProcesingData.prio_func_eval_baselines.types import BaselineResult


BaselineEvaluator = Callable[
    [
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        dict[str, Any],
    ],
    BaselineResult,
]

_BASELINE_REGISTRY: dict[str, tuple[str, BaselineEvaluator]] = {
    "mixture learning": ("Mixture Learning", evaluate_mixture_learning),
    "mixture_learning": ("Mixture Learning", evaluate_mixture_learning),
    "independent learning scheme": (
        "Independent Learning Scheme",
        evaluate_independent_learning_scheme,
    ),
    "independent_learning_scheme": (
        "Independent Learning Scheme",
        evaluate_independent_learning_scheme,
    ),
    "tl-gdes": ("TL-GDES", evaluate_tl_gdes),
    "tl_gdes": ("TL-GDES", evaluate_tl_gdes),
    "tl-pr": ("TL-PR", evaluate_tl_pr),
    "tl_pr": ("TL-PR", evaluate_tl_pr),
}


def normalize_baseline_name(name: str) -> str:
    normalized_name = name.strip().lower()
    if normalized_name not in _BASELINE_REGISTRY:
        supported = ", ".join(sorted({value[0] for value in _BASELINE_REGISTRY.values()}))
        raise ValueError(f"Unsupported baseline {name!r}. Supported baselines: {supported}.")
    return _BASELINE_REGISTRY[normalized_name][0]


def evaluate_baseline(
    name: str,
    *,
    training_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
    heldout_data: pd.DataFrame,
    training_ancestry_groups: tuple[str, ...],
    calibration_ancestry_groups: tuple[str, ...],
    heldout_ancestry_groups: tuple[str, ...],
    options: dict[str, Any] | None = None,
) -> BaselineResult:
    normalized_name = name.strip().lower()
    if normalized_name not in _BASELINE_REGISTRY:
        supported = ", ".join(sorted({value[0] for value in _BASELINE_REGISTRY.values()}))
        raise ValueError(f"Unsupported baseline {name!r}. Supported baselines: {supported}.")
    _, evaluator = _BASELINE_REGISTRY[normalized_name]
    return evaluator(
        training_data,
        calibration_data,
        heldout_data,
        training_ancestry_groups,
        calibration_ancestry_groups,
        heldout_ancestry_groups,
        options or {},
    )