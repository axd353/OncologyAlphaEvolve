from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from PostProcesingData.prio_func_eval_baselines.mixture_learning import _extract_features
from PostProcesingData.prio_func_eval_baselines.mixture_learning import _extract_labels
from PostProcesingData.prio_func_eval_baselines.types import BaselineResult
from funsearch_pipeline.evaluation.procedure2 import _safe_roc_auc


def _ancestry_mask(ancestry_groups: tuple[str, ...], ancestry_group: str) -> np.ndarray:
    return np.asarray([group == ancestry_group for group in ancestry_groups], dtype=bool)


def _select_rows(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return frame.loc[mask].reset_index(drop=True)


def evaluate_independent_learning_scheme(
    training_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
    heldout_data: pd.DataFrame,
    training_ancestry_groups: tuple[str, ...],
    calibration_ancestry_groups: tuple[str, ...],
    heldout_ancestry_groups: tuple[str, ...],
    options: dict[str, Any],
) -> BaselineResult:
    if len(training_data) != len(training_ancestry_groups):
        raise ValueError("Training ancestry assignments did not match training row count.")
    if len(calibration_data) != len(calibration_ancestry_groups):
        raise ValueError("Calibration ancestry assignments did not match calibration row count.")
    if len(heldout_data) != len(heldout_ancestry_groups):
        raise ValueError("Heldout ancestry assignments did not match heldout row count.")

    alpha = float(options.get("alpha", 1.0))
    combined_training = pd.concat([training_data, calibration_data], ignore_index=True)
    combined_ancestry_groups = training_ancestry_groups + calibration_ancestry_groups
    discovered_ancestries = tuple(dict.fromkeys(heldout_ancestry_groups))
    heldout_scores = np.empty(len(heldout_data), dtype=float)

    for ancestry_group in discovered_ancestries:
        training_mask = _ancestry_mask(combined_ancestry_groups, ancestry_group)
        if not np.any(training_mask):
            raise ValueError(
                f"Independent Learning Scheme found heldout ancestry {ancestry_group!r} "
                "without any matching training or calibration samples."
            )
        heldout_mask = _ancestry_mask(heldout_ancestry_groups, ancestry_group)
        ancestry_training = _select_rows(combined_training, training_mask)
        ancestry_heldout = _select_rows(heldout_data, heldout_mask)

        model = Ridge(alpha=alpha)
        model.fit(_extract_features(ancestry_training), _extract_labels(ancestry_training))
        heldout_scores[heldout_mask] = model.predict(_extract_features(ancestry_heldout))

    return BaselineResult(
        auc_roc=float(_safe_roc_auc(_extract_labels(heldout_data), heldout_scores)),
        heldout_scores=tuple(float(value) for value in heldout_scores),
    )