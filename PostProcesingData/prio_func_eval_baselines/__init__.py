from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from PostProcesingData.prio_func_eval_baselines.mixture_learning import evaluate_mixture_learning


BaselineEvaluator = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]], float]

_BASELINE_REGISTRY: dict[str, tuple[str, BaselineEvaluator]] = {
    "mixture learning": ("Mixture Learning", evaluate_mixture_learning),
    "mixture_learning": ("Mixture Learning", evaluate_mixture_learning),
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
    options: dict[str, Any] | None = None,
) -> float:
    normalized_name = name.strip().lower()
    if normalized_name not in _BASELINE_REGISTRY:
        supported = ", ".join(sorted({value[0] for value in _BASELINE_REGISTRY.values()}))
        raise ValueError(f"Unsupported baseline {name!r}. Supported baselines: {supported}.")
    _, evaluator = _BASELINE_REGISTRY[normalized_name]
    return float(
        evaluator(
            training_data,
            calibration_data,
            heldout_data,
            options or {},
        )
    )