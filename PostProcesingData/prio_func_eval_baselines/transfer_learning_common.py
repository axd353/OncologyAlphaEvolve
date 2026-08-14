from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from GenomicsHelpers.oracle_data_adapter import DEFAULT_LABEL_FIELD
from GenomicsHelpers.oracle_data_adapter import DOSAGE_COLUMN_PREFIX


@dataclass(frozen=True)
class DosageTransform:
    dosage_cols: tuple[str, ...]
    impute_values: np.ndarray
    center_values: np.ndarray
    scale_values: np.ndarray


def get_dosage_cols(frame: pd.DataFrame) -> list[str]:
    dosage_cols = [
        str(column)
        for column in frame.columns
        if str(column).startswith(DOSAGE_COLUMN_PREFIX)
    ]
    if not dosage_cols:
        raise ValueError("Transfer-learning baselines require at least one dosage column.")
    return dosage_cols


def extract_labels(frame: pd.DataFrame) -> np.ndarray:
    if DEFAULT_LABEL_FIELD not in frame.columns:
        raise ValueError(
            "Transfer-learning baselines require the label column "
            f"{DEFAULT_LABEL_FIELD!r}."
        )
    labels = frame[DEFAULT_LABEL_FIELD].to_numpy(dtype=int)
    observed = sorted(np.unique(labels).tolist())
    if not set(observed).issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1, observed {observed}.")
    return labels


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def binary_logp_correct(labels: np.ndarray, p_case: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    p_case = np.clip(np.asarray(p_case, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    p_correct = np.where(labels == 1, p_case, 1.0 - p_case)
    return np.log(p_correct)


def mean_logp_correct(labels: np.ndarray, p_case: np.ndarray) -> float:
    return float(np.mean(binary_logp_correct(labels, p_case)))


def make_sample_weight(labels: np.ndarray, class_weight: Optional[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if class_weight != "balanced":
        return np.ones(len(labels), dtype=np.float64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = np.ones(len(labels), dtype=np.float64)
    for label_value in (0, 1):
        if counts[label_value] > 0:
            weights[labels == label_value] = len(labels) / (2.0 * counts[label_value])
    return weights


def has_enough_binary_signal(labels: np.ndarray, min_class_count: int) -> bool:
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size < 2:
        return False
    counts = np.bincount(labels, minlength=2)
    return bool(np.all(counts >= int(min_class_count)))


def fit_dosage_transform(
    frame: pd.DataFrame,
    dosage_cols: Sequence[str],
    *,
    center: bool = True,
    scale: bool = False,
) -> DosageTransform:
    matrix = frame.loc[:, list(dosage_cols)].to_numpy(dtype=np.float64)
    impute_values = np.nanmean(matrix, axis=0)
    impute_values = np.where(np.isfinite(impute_values), impute_values, 0.0)
    center_values = impute_values.copy() if center else np.zeros(len(dosage_cols), dtype=np.float64)
    if scale:
        imputed_matrix = np.where(np.isnan(matrix), impute_values, matrix)
        scale_values = np.nanstd(imputed_matrix, axis=0, ddof=0)
        scale_values = np.where(
            (np.isfinite(scale_values)) & (scale_values > 1e-8),
            scale_values,
            1.0,
        )
    else:
        scale_values = np.ones(len(dosage_cols), dtype=np.float64)
    return DosageTransform(
        dosage_cols=tuple(str(column) for column in dosage_cols),
        impute_values=impute_values,
        center_values=center_values,
        scale_values=scale_values,
    )


def apply_dosage_transform(frame: pd.DataFrame, transform: DosageTransform) -> np.ndarray:
    matrix = frame.loc[:, list(transform.dosage_cols)].to_numpy(dtype=np.float64)
    matrix = np.where(np.isnan(matrix), transform.impute_values, matrix)
    return (matrix - transform.center_values) / transform.scale_values


def fit_logistic_gd(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    n_iter: int,
    learning_rate: float,
    l2: float = 0.0,
    class_weight: Optional[str] = None,
    beta_init: Optional[np.ndarray] = None,
    intercept_init: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    beta = (
        np.zeros(feature_matrix.shape[1], dtype=np.float64)
        if beta_init is None
        else np.asarray(beta_init, dtype=np.float64).copy()
    )
    intercept = logit(float(np.mean(labels))) if intercept_init is None else float(intercept_init)
    weights = make_sample_weight(labels.astype(int), class_weight)
    weight_sum = float(np.sum(weights))

    for _ in range(int(n_iter)):
        p_case = sigmoid(intercept + feature_matrix @ beta)
        residual = (p_case - labels) * weights
        grad_beta = (feature_matrix.T @ residual) / weight_sum + float(l2) * beta
        grad_intercept = float(np.sum(residual) / weight_sum)
        beta -= float(learning_rate) * grad_beta
        intercept -= float(learning_rate) * grad_intercept

    return beta, intercept


def select_tl_gdes_iteration(
    *,
    x_adapt: np.ndarray,
    y_adapt: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    beta_prior: np.ndarray,
    intercept_prior: float,
    max_iter: int,
    learning_rate: float,
    l2: float,
    class_weight: Optional[str],
) -> tuple[int, float]:
    beta = beta_prior.copy()
    intercept = float(intercept_prior)
    weights = make_sample_weight(y_adapt, class_weight)
    weight_sum = float(np.sum(weights))

    if len(y_cal) > 0:
        best_score = mean_logp_correct(y_cal, sigmoid(intercept + x_cal @ beta))
    else:
        best_score = mean_logp_correct(y_adapt, sigmoid(intercept + x_adapt @ beta))
    best_iter = 0

    for iteration in range(1, int(max_iter) + 1):
        p_case = sigmoid(intercept + x_adapt @ beta)
        residual = (p_case - y_adapt) * weights
        grad_beta = (x_adapt.T @ residual) / weight_sum + float(l2) * (beta - beta_prior)
        grad_intercept = float(np.sum(residual) / weight_sum)
        beta -= float(learning_rate) * grad_beta
        intercept -= float(learning_rate) * grad_intercept

        if len(y_cal) > 0:
            score = mean_logp_correct(y_cal, sigmoid(intercept + x_cal @ beta))
        else:
            score = mean_logp_correct(y_adapt, sigmoid(intercept + x_adapt @ beta))
        if score > best_score:
            best_score = score
            best_iter = iteration

    return best_iter, float(best_score)


def stratified_train_cal_split(
    labels: np.ndarray,
    *,
    cal_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    indices = np.arange(len(labels))
    counts = np.bincount(labels, minlength=2)
    if len(labels) < 4 or np.unique(labels).size < 2 or int(np.min(counts[counts > 0])) < 2:
        return indices, np.array([], dtype=int)

    test_size = max(2, int(round(float(cal_fraction) * len(labels))))
    test_size = min(test_size, len(labels) - 2)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    adapt_idx, cal_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return indices[adapt_idx], indices[cal_idx]


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    threshold = float(max(threshold, 0.0))
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def estimate_tlpr_step_size(
    feature_matrix: np.ndarray,
    sample_weights: np.ndarray,
    *,
    lambda_value: float,
    alpha: float,
    requested_learning_rate: float,
) -> float:
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    weight_sum = float(np.sum(sample_weights))
    if feature_matrix.size == 0 or weight_sum <= 0:
        return float(requested_learning_rate)

    weighted_features = feature_matrix * np.sqrt(sample_weights / weight_sum)[:, None]
    spectral_sq = float(np.linalg.norm(weighted_features, ord=2) ** 2)
    l2_strength = float(lambda_value) * max(0.0, 1.0 - float(alpha))
    lipschitz = 0.25 * spectral_sq + l2_strength
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        return float(requested_learning_rate)

    stable_step = 0.95 / lipschitz
    if requested_learning_rate is None or float(requested_learning_rate) <= 0:
        return float(stable_step)
    return float(min(float(requested_learning_rate), stable_step))


def fit_binary_tlpr_elastic_net(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    beta_prior: np.ndarray,
    intercept_prior: float,
    alpha: float,
    lambda_value: float,
    n_iter: int,
    learning_rate: float,
    class_weight: Optional[str] = None,
    beta_init: Optional[np.ndarray] = None,
    intercept_init: Optional[float] = None,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float, int]:
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    labels_float = labels.astype(np.float64)
    beta_prior = np.asarray(beta_prior, dtype=np.float64)
    beta_current = (
        beta_prior.copy()
        if beta_init is None
        else np.asarray(beta_init, dtype=np.float64).copy()
    )
    intercept_current = float(intercept_prior) if intercept_init is None else float(intercept_init)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    lambda_value = float(max(lambda_value, 0.0))
    l1_strength = lambda_value * alpha
    l2_strength = lambda_value * (1.0 - alpha)

    sample_weights = make_sample_weight(labels, class_weight)
    weight_sum = float(np.sum(sample_weights))
    if weight_sum <= 0:
        raise ValueError("TL-PR sample weights sum to zero.")

    step_size = estimate_tlpr_step_size(
        feature_matrix,
        sample_weights,
        lambda_value=lambda_value,
        alpha=alpha,
        requested_learning_rate=learning_rate,
    )

    iterations_run = 0
    for iteration_index in range(int(n_iter)):
        beta_before = beta_current.copy()
        intercept_before = float(intercept_current)

        probabilities = sigmoid(intercept_current + feature_matrix @ beta_current)
        residual = (probabilities - labels_float) * sample_weights
        beta_gradient = (feature_matrix.T @ residual) / weight_sum
        if l2_strength > 0:
            beta_gradient = beta_gradient + l2_strength * (beta_current - beta_prior)
        intercept_gradient = float(np.sum(residual) / weight_sum)

        beta_proposed = beta_current - step_size * beta_gradient
        if l1_strength > 0:
            beta_current = beta_prior + soft_threshold(
                beta_proposed - beta_prior,
                step_size * l1_strength,
            )
        else:
            beta_current = beta_proposed
        intercept_current -= step_size * intercept_gradient

        iterations_run = iteration_index + 1
        max_beta_change = (
            float(np.max(np.abs(beta_current - beta_before)))
            if len(beta_current)
            else 0.0
        )
        max_change = max(max_beta_change, abs(float(intercept_current) - intercept_before))
        if float(tol) > 0 and max_change < float(tol):
            break

    return beta_current, float(intercept_current), int(iterations_run)


def make_tlpr_lambda_grid(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    beta_prior: np.ndarray,
    intercept_prior: float,
    alpha: float,
    n_lambdas: int,
    lambda_min_ratio: float,
    ridge_grid_max: float,
    ridge_grid_min: float,
    class_weight: Optional[str],
) -> np.ndarray:
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    n_lambdas = max(1, int(n_lambdas))

    if alpha > 0:
        sample_weights = make_sample_weight(labels, class_weight)
        weight_sum = float(np.sum(sample_weights))
        if weight_sum <= 0:
            raise ValueError("TL-PR sample weights sum to zero while making lambda grid.")
        probabilities_at_prior = sigmoid(float(intercept_prior) + feature_matrix @ beta_prior)
        residual_at_prior = (probabilities_at_prior - labels.astype(np.float64)) * sample_weights
        gradient_at_prior = (feature_matrix.T @ residual_at_prior) / weight_sum
        lambda_max = float(np.max(np.abs(gradient_at_prior)) + 1e-8) / alpha
        lambda_min = max(lambda_max * float(lambda_min_ratio), 1e-8)
        return np.geomspace(lambda_max, lambda_min, n_lambdas)

    per_feature_second_moment = np.mean(feature_matrix * feature_matrix, axis=0)
    ridge_scale = float(np.nanmean(per_feature_second_moment))
    if not np.isfinite(ridge_scale) or ridge_scale <= 0:
        ridge_scale = 1.0
    lambda_high = ridge_scale * float(ridge_grid_max)
    lambda_low = ridge_scale * float(ridge_grid_min)
    lambda_low = max(lambda_low, 1e-8)
    lambda_high = max(lambda_high, lambda_low)
    return np.geomspace(lambda_high, lambda_low, n_lambdas)


def make_tlpr_cv_splits(
    labels: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    positive_counts = counts[counts > 0]
    if len(labels) < 4 or len(positive_counts) < 2:
        return []
    n_splits = min(int(n_folds), int(np.min(positive_counts)))
    if n_splits < 2:
        return []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (fit_idx.astype(int), valid_idx.astype(int))
        for fit_idx, valid_idx in splitter.split(np.zeros(len(labels)), labels)
    ]


def select_tlpr_hyperparameters(
    *,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    beta_prior: np.ndarray,
    intercept_prior: float,
    alpha_grid: Sequence[float],
    n_lambdas: int,
    lambda_min_ratio: float,
    ridge_grid_max: float,
    ridge_grid_min: float,
    cv_folds: int,
    n_iter: int,
    learning_rate: float,
    class_weight: Optional[str],
    seed: int,
    tol: float,
) -> tuple[float, float, float]:
    cv_splits = make_tlpr_cv_splits(labels, n_folds=cv_folds, seed=seed)
    if not cv_splits:
        raise ValueError("TL-PR needs at least two stratified folds for alpha/lambda tuning.")

    best_alpha = np.nan
    best_lambda = np.nan
    best_cv_logp = -np.inf

    for alpha_value in alpha_grid:
        alpha_value = float(np.clip(alpha_value, 0.0, 1.0))
        lambda_grid = make_tlpr_lambda_grid(
            feature_matrix,
            labels,
            beta_prior=beta_prior,
            intercept_prior=intercept_prior,
            alpha=alpha_value,
            n_lambdas=n_lambdas,
            lambda_min_ratio=lambda_min_ratio,
            ridge_grid_max=ridge_grid_max,
            ridge_grid_min=ridge_grid_min,
            class_weight=class_weight,
        )
        for lambda_value in lambda_grid:
            fold_scores: list[float] = []
            for fit_idx, valid_idx in cv_splits:
                beta_fold, intercept_fold, _ = fit_binary_tlpr_elastic_net(
                    feature_matrix[fit_idx],
                    labels[fit_idx],
                    beta_prior=beta_prior,
                    intercept_prior=intercept_prior,
                    alpha=alpha_value,
                    lambda_value=float(lambda_value),
                    n_iter=n_iter,
                    learning_rate=learning_rate,
                    class_weight=class_weight,
                    tol=tol,
                )
                valid_probabilities = sigmoid(intercept_fold + feature_matrix[valid_idx] @ beta_fold)
                fold_scores.append(mean_logp_correct(labels[valid_idx], valid_probabilities))
            mean_cv_logp = float(np.mean(fold_scores))
            is_better = mean_cv_logp > best_cv_logp + 1e-12
            is_tie_with_more_shrinkage = (
                np.isfinite(best_cv_logp)
                and abs(mean_cv_logp - best_cv_logp) <= 1e-12
                and float(lambda_value) > float(best_lambda)
            )
            if is_better or is_tie_with_more_shrinkage:
                best_alpha = float(alpha_value)
                best_lambda = float(lambda_value)
                best_cv_logp = mean_cv_logp

    return float(best_alpha), float(best_lambda), float(best_cv_logp)