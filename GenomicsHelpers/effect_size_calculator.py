"""Estimate local marginal variant effects from ancestry-nearby training samples.

The default estimator is a single-variant logistic regression for a binary trait.
Within the ancestry ball centered at ``a`` with radius ``r``, the reported effect
size is the coefficient on the target variant dosage. Optional non-genetic
covariates can be included in the regression, while other genetic variants are
left out so the result remains a marginal effect estimate for the target
variant.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

import numpy as np

from GenomicsHelpers.oracle_data_adapter import iter_training_records
from GenomicsHelpers.oracle_data_adapter import read_ancestry_coordinate
from GenomicsHelpers.oracle_data_adapter import read_label
from GenomicsHelpers.oracle_data_adapter import read_optional_covariates
from GenomicsHelpers.oracle_data_adapter import read_variant_dosage


@dataclass(frozen=True)
class LocalVariantData:
    """Arrays extracted from the ancestry-local neighborhood for one variant."""

    labels: np.ndarray
    genotype: np.ndarray
    covariates: np.ndarray | None
    sample_count: int


def effect_size_calculator(
    training_data: Any,
    ancestry_coordinate: Sequence[float],
    target_variant: Any,
    radius: float,
    *,
    min_samples: int = 25,
    fallback_effect: float = 0.0,
    l2_penalty: float = 1e-6,
    max_iter: int = 50,
    tolerance: float = 1e-8,
    logger: logging.Logger | None = None,
) -> float:
    """Estimate ``\hat b_j(a)`` from the closed ancestry ball around ``a``.

    Args:
        training_data: Loaded dataset container understood by
            ``GenomicsHelpers.oracle_data_adapter``.
        ancestry_coordinate: The target ancestry location ``a``.
        target_variant: Variant key or index understood by the adapter. For the
            MEC DataFrame defaults, both ``dosage__...`` column names and the
            unprefixed variant names they correspond to are accepted.
        radius: Closed-ball radius around ``a`` used to choose local samples.
        min_samples: Minimum number of local samples required before attempting
            regression.
        fallback_effect: Value returned when the local neighborhood is too small
            or does not contain enough variation to identify the effect.
        l2_penalty: Small ridge penalty used to stabilize the Newton updates.
        max_iter: Maximum number of Newton iterations.
        tolerance: Convergence tolerance for the coefficient updates.
        logger: Optional standard-library logger used to record why the
            fallback effect is returned.

    Returns:
        The local marginal effect estimate for the target variant.
    """

    local_data = prepare_local_variant_data(
        training_data=training_data,
        ancestry_coordinate=ancestry_coordinate,
        target_variant=target_variant,
        radius=radius,
    )
    failure_reason = get_nonidentifiable_local_effect_reason(
        local_data,
        min_samples=min_samples,
    )
    if failure_reason is not None:
        if logger is not None:
            logger.info(
                "Returning fallback effect %s for variant %r at radius %s: %s",
                fallback_effect,
                target_variant,
                radius,
                failure_reason,
            )
        return fallback_effect

    return estimate_marginal_logistic_effect(
        labels=local_data.labels,
        genotype=local_data.genotype,
        covariates=local_data.covariates,
        l2_penalty=l2_penalty,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def prepare_local_variant_data(
    training_data: Any,
    ancestry_coordinate: Sequence[float],
    target_variant: Any,
    radius: float,
) -> LocalVariantData:
    """Extract labels, genotype dosage, and covariates inside the closed ball.

    Missing target dosages are mean-imputed within the local neighborhood so the
    downstream logistic fit never sees NaNs from sparse genotype gaps.
    """

    center = np.asarray(ancestry_coordinate, dtype=float)
    labels: list[float] = []
    genotype: list[float] = []
    covariates: list[np.ndarray | None] = []

    for record in iter_training_records(training_data):
        record_ancestry = read_ancestry_coordinate(record)
        if not is_inside_closed_ball(record_ancestry, center, radius):
            continue

        labels.append(read_label(record))
        genotype.append(read_variant_dosage(record, target_variant))

        covariates.append(read_optional_covariates(record))

    covariate_matrix = None
    if any(row is not None for row in covariates):
        covariate_matrix = align_covariate_rows(covariates, sample_count=len(labels))

    genotype_array = np.asarray(genotype, dtype=float)
    genotype_array = impute_missing_genotype_values(genotype_array)

    return LocalVariantData(
        labels=np.asarray(labels, dtype=float),
        genotype=genotype_array,
        covariates=covariate_matrix,
        sample_count=len(labels),
    )


def is_inside_closed_ball(
    point: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> bool:
    """Return whether ``point`` lies in the closed Euclidean ball."""

    return float(np.linalg.norm(point - center)) <= radius


def has_identifiable_local_effect(
    local_data: LocalVariantData,
    *,
    min_samples: int,
) -> bool:
    """Check the minimum conditions needed for a local logistic effect estimate."""

    return get_nonidentifiable_local_effect_reason(
        local_data,
        min_samples=min_samples,
    ) is None


def get_nonidentifiable_local_effect_reason(
    local_data: LocalVariantData,
    *,
    min_samples: int,
) -> str | None:
    """Return the reason the local effect is not estimable, if any."""

    if local_data.sample_count == 0:
        return "no local samples were found inside the ancestry ball"

    if local_data.sample_count < min_samples:
        return (
            f"only {local_data.sample_count} local samples were found, "
            f"below min_samples={min_samples}"
        )

    unique_labels = np.unique(local_data.labels)
    if unique_labels.size < 2:
        return f"local labels contain only one class: {unique_labels.tolist()}"

    if not np.isfinite(local_data.genotype).any():
        return "local genotype is entirely missing"

    if np.allclose(local_data.genotype, local_data.genotype[0]):
        return (
            "local genotype has no variation; all target dosages are "
            f"approximately {float(local_data.genotype[0])}"
        )

    return None


def estimate_marginal_logistic_effect(
    labels: np.ndarray,
    genotype: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    l2_penalty: float = 1e-6,
    max_iter: int = 50,
    tolerance: float = 1e-8,
) -> float:
    """Fit a single-variant logistic regression and return the dosage coefficient."""

    design_matrix = build_design_matrix(genotype, covariates)
    coefficients = fit_logistic_regression_newton(
        design_matrix=design_matrix,
        labels=labels,
        l2_penalty=l2_penalty,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    return float(coefficients[1])


def build_design_matrix(
    genotype: np.ndarray,
    covariates: np.ndarray | None,
) -> np.ndarray:
    """Build an intercept-first design matrix for local logistic regression."""

    intercept = np.ones((genotype.shape[0], 1), dtype=float)
    dosage_column = genotype.reshape(-1, 1)
    if covariates is None:
        return np.hstack([intercept, dosage_column])
    return np.hstack([intercept, dosage_column, covariates])


def fit_logistic_regression_newton(
    design_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    l2_penalty: float,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    """Solve a penalized logistic regression with Newton updates."""

    coefficients = np.zeros(design_matrix.shape[1], dtype=float)
    penalty = np.eye(design_matrix.shape[1], dtype=float) * l2_penalty
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        linear_predictor = design_matrix @ coefficients
        probabilities = sigmoid(linear_predictor)
        weights = probabilities * (1.0 - probabilities)

        weighted_design = design_matrix * weights[:, None]
        hessian = design_matrix.T @ weighted_design + penalty
        gradient = design_matrix.T @ (labels - probabilities) - penalty @ coefficients

        step = np.linalg.solve(hessian, gradient)
        updated = coefficients + step
        if np.linalg.norm(updated - coefficients) <= tolerance:
            coefficients = updated
            break
        coefficients = updated

    return coefficients


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic link."""

    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def impute_missing_genotype_values(genotype: np.ndarray) -> np.ndarray:
    """Fill sparse missing dosage values with the local mean dosage."""

    if genotype.size == 0:
        return genotype

    finite_mask = np.isfinite(genotype)
    if finite_mask.all() or not finite_mask.any():
        return genotype

    imputed = genotype.copy()
    imputed[~finite_mask] = float(np.mean(imputed[finite_mask]))
    return imputed


def align_covariate_rows(
    covariate_rows: Sequence[np.ndarray | None],
    *,
    sample_count: int,
) -> np.ndarray:
    """Stack covariates and fill missing rows with zeros when needed."""

    if not covariate_rows:
        return np.zeros((sample_count, 0), dtype=float)

    first_present_row = next(row for row in covariate_rows if row is not None)
    width = first_present_row.shape[0]
    normalized_rows = []
    for row in covariate_rows:
        if row is None:
            normalized_rows.append(np.zeros(width, dtype=float))
            continue
        if row.shape[0] != width:
            raise ValueError("All local covariate vectors must have the same length.")
        normalized_rows.append(row)
    return np.vstack(normalized_rows)

