from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import apply_dosage_transform
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import extract_labels
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import fit_binary_tlpr_elastic_net
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import fit_dosage_transform
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import fit_logistic_gd
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import get_dosage_cols
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import has_enough_binary_signal
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import select_tlpr_hyperparameters
from PostProcesingData.prio_func_eval_baselines.transfer_learning_common import sigmoid
from PostProcesingData.prio_func_eval_baselines.types import BaselineResult
from funsearch_pipeline.evaluation.procedure2 import _safe_roc_auc


def _mask_for_ancestry(ancestry_groups: tuple[str, ...], ancestry_group: str) -> np.ndarray:
    return np.asarray([group == ancestry_group for group in ancestry_groups], dtype=bool)


def _mask_for_not_ancestry(ancestry_groups: tuple[str, ...], ancestry_group: str) -> np.ndarray:
    return np.asarray([group != ancestry_group for group in ancestry_groups], dtype=bool)


def evaluate_tl_pr(
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

    alpha_grid = tuple(float(value) for value in options.get("alpha_grid", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
    n_lambdas = int(options.get("n_lambdas", 30))
    lambda_min_ratio = float(options.get("lambda_min_ratio", 0.01))
    ridge_grid_max = float(options.get("ridge_grid_max", 10000.0))
    ridge_grid_min = float(options.get("ridge_grid_min", 1.0))
    cv_folds = int(options.get("cv_folds", 2))
    max_iter = int(options.get("max_iter", 600))
    learning_rate = float(options.get("learning_rate", 0.05))
    tol = float(options.get("tol", 1e-6))
    source_n_iter = int(options.get("source_n_iter", 3000))
    source_learning_rate = float(options.get("source_learning_rate", 0.05))
    source_l2 = float(options.get("source_l2", 1e-4))
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
                f"TL-PR needs non-target source samples for ancestry {ancestry_group!r}."
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
        source_features = apply_dosage_transform(source_df, transform)
        source_labels = extract_labels(source_df)
        if not has_enough_binary_signal(source_labels, min_class_count=1):
            raise ValueError(
                f"TL-PR source pool for ancestry {ancestry_group!r} does not contain both labels."
            )
        beta_source, intercept_source = fit_logistic_gd(
            source_features,
            source_labels,
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
            target_features = apply_dosage_transform(target_df, transform)
            target_labels = extract_labels(target_df)
            ancestry_seed_offset = sum((index + 1) * ord(char) for index, char in enumerate(ancestry_group))
            best_alpha, best_lambda, _ = select_tlpr_hyperparameters(
                feature_matrix=target_features,
                labels=target_labels,
                beta_prior=beta_source,
                intercept_prior=intercept_source,
                alpha_grid=alpha_grid,
                n_lambdas=n_lambdas,
                lambda_min_ratio=lambda_min_ratio,
                ridge_grid_max=ridge_grid_max,
                ridge_grid_min=ridge_grid_min,
                cv_folds=cv_folds,
                n_iter=max_iter,
                learning_rate=learning_rate,
                class_weight=class_weight,
                seed=seed + ancestry_seed_offset,
                tol=tol,
            )
            beta_final, intercept_final, _ = fit_binary_tlpr_elastic_net(
                target_features,
                target_labels,
                beta_prior=beta_source,
                intercept_prior=intercept_source,
                alpha=best_alpha,
                lambda_value=best_lambda,
                n_iter=max_iter,
                learning_rate=learning_rate,
                class_weight=class_weight,
                tol=tol,
            )

        heldout_features = apply_dosage_transform(
            heldout_data.loc[heldout_mask].reset_index(drop=True),
            transform,
        )
        heldout_scores[heldout_mask] = intercept_final + heldout_features @ beta_final

    return BaselineResult(
        auc_roc=float(_safe_roc_auc(extract_labels(heldout_data), heldout_scores)),
        heldout_scores=tuple(float(value) for value in heldout_scores),
    )