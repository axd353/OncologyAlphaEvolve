from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import apply_dosage_transform
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import extract_labels
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import fit_dosage_transform
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import fit_logistic_gd
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import get_dosage_cols
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import has_enough_binary_signal
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import select_tl_gdes_iteration
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import sigmoid
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import stratified_train_cal_split
from PostProcesingData.prio_func_eval_baselines.types import BaselineResult
from funsearch_pipeline.evaluation.procedure2 import _safe_roc_auc


def _mask_for_ancestry(ancestry_groups: tuple[str, ...], ancestry_group: str) -> np.ndarray:
    return np.asarray([group == ancestry_group for group in ancestry_groups], dtype=bool)


def _mask_for_not_ancestry(ancestry_groups: tuple[str, ...], ancestry_group: str) -> np.ndarray:
    return np.asarray([group != ancestry_group for group in ancestry_groups], dtype=bool)


def evaluate_tl_gdes(
    training_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
    heldout_data: pd.DataFrame,
    training_ancestry_groups: tuple[str, ...],
    calibration_ancestry_groups: tuple[str, ...],
    heldout_ancestry_groups: tuple[str, ...],
    options: dict[str, Any],
) -> BaselineResult:
    combined_training = pd.concat([training_data, calibration_data], ignore_index=True)
    combined_ancestry_groups = training_ancestry_groups + calibration_ancestry_groups
    if len(combined_training) != len(combined_ancestry_groups):
        raise ValueError("Combined training ancestry assignments did not match row count.")
    if len(heldout_data) != len(heldout_ancestry_groups):
        raise ValueError("Heldout ancestry assignments did not match row count.")

    dosage_cols = get_dosage_cols(combined_training)
    heldout_scores = np.empty(len(heldout_data), dtype=np.float64)
    discovered_ancestries = tuple(dict.fromkeys(heldout_ancestry_groups))

    max_iter = int(options.get("max_iter", 100))
    learning_rate = float(options.get("learning_rate", 0.05))
    source_n_iter = int(options.get("source_n_iter", 3000))
    source_learning_rate = float(options.get("source_learning_rate", 0.05))
    source_l2 = float(options.get("source_l2", 1e-4))
    target_l2 = float(options.get("target_l2", 0.0))
    cal_fraction = float(options.get("cal_fraction", 0.25))
    min_target_n = int(options.get("min_target_n", 20))
    min_class_count = int(options.get("min_class_count", 2))
    class_weight = options.get("class_weight")
    center_dosages = bool(options.get("center_dosages", True))
    scale_dosages = bool(options.get("scale_dosages", False))
    seed = int(options.get("seed", 0))

    for ancestry_group in discovered_ancestries:
        source_mask = _mask_for_not_ancestry(combined_ancestry_groups, ancestry_group)
        target_mask = _mask_for_ancestry(combined_ancestry_groups, ancestry_group)
        heldout_mask = _mask_for_ancestry(heldout_ancestry_groups, ancestry_group)
        if not np.any(source_mask):
            raise ValueError(
                f"TL-GDES needs non-target source samples for ancestry {ancestry_group!r}."
            )
        if not np.any(heldout_mask):
            continue

        source_df = combined_training.loc[source_mask].reset_index(drop=True)
        target_df = combined_training.loc[target_mask].reset_index(drop=True)
        transform = fit_dosage_transform(
            source_df,
            dosage_cols,
            center=center_dosages,
            scale=scale_dosages,
        )
        x_source = apply_dosage_transform(source_df, transform)
        y_source = extract_labels(source_df)
        if not has_enough_binary_signal(y_source, min_class_count=1):
            raise ValueError(
                f"TL-GDES source pool for ancestry {ancestry_group!r} does not contain both labels."
            )
        beta_source, intercept_source = fit_logistic_gd(
            x_source,
            y_source,
            n_iter=source_n_iter,
            learning_rate=source_learning_rate,
            l2=source_l2,
            class_weight=class_weight,
        )

        use_source_only = (
            len(target_df) < min_target_n
            or not has_enough_binary_signal(extract_labels(target_df), min_class_count=min_class_count)
        )
        if use_source_only:
            beta_final = beta_source
            intercept_final = intercept_source
        else:
            x_target = apply_dosage_transform(target_df, transform)
            y_target = extract_labels(target_df)
            ancestry_seed_offset = sum((index + 1) * ord(char) for index, char in enumerate(ancestry_group))
            adapt_idx, cal_idx = stratified_train_cal_split(
                y_target,
                cal_fraction=cal_fraction,
                seed=seed + ancestry_seed_offset,
            )
            best_iter, _ = select_tl_gdes_iteration(
                x_adapt=x_target[adapt_idx],
                y_adapt=y_target[adapt_idx],
                x_cal=x_target[cal_idx],
                y_cal=y_target[cal_idx],
                beta_prior=beta_source,
                intercept_prior=intercept_source,
                max_iter=max_iter,
                learning_rate=learning_rate,
                l2=target_l2,
                class_weight=class_weight,
            )
            beta_final, intercept_final = fit_logistic_gd(
                x_target,
                y_target,
                n_iter=best_iter,
                learning_rate=learning_rate,
                l2=target_l2,
                class_weight=class_weight,
                beta_init=beta_source,
                intercept_init=intercept_source,
            )

        x_heldout = apply_dosage_transform(
            heldout_data.loc[heldout_mask].reset_index(drop=True),
            transform,
        )
        heldout_scores[heldout_mask] = intercept_final + x_heldout @ beta_final

    return BaselineResult(
        auc_roc=float(_safe_roc_auc(extract_labels(heldout_data), heldout_scores)),
        heldout_scores=tuple(float(value) for value in heldout_scores),
    )