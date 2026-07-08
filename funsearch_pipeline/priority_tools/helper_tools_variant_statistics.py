# helper_tools
from __future__ import annotations

import math
from numbers import Integral
from numbers import Real

import numpy as np

from GenomicsHelpers.effect_size_calculator import build_design_matrix
from GenomicsHelpers.effect_size_calculator import estimate_marginal_logistic_effect
from GenomicsHelpers.effect_size_calculator import fit_logistic_regression_newton
from GenomicsHelpers.effect_size_calculator import impute_missing_genotype_values
from GenomicsHelpers.effect_size_calculator import sigmoid

from .contracts import PriorityAncestryCoordinate
from .contracts import PriorityTargetVariant
from .contracts import PriorityTrainingData
from .contracts import PriorityTrainingRecord
from .helper_tools_ancestry_distance import _equal_count_shell_upper_bounds
from .helper_tools_ancestry_distance import _radius_for_exact_prefix_count
from .helper_tools_ancestry_distance import _validated_sample_count
from .helper_tools_ancestry_distance import _validate_ancestry_shapes

_DEFAULT_EFFECT_MIN_SAMPLES = 25
_DEFAULT_L2_PENALTY = 1e-6
_DEFAULT_MAX_ITER = 50
_DEFAULT_TOLERANCE = 1e-8


def _validated_radius(radius: Real) -> float:
    """Validate a non-negative finite ancestry radius.

    Input:
        radius: Candidate closed-ball radius in ancestry distance.

    Output:
        Validated floating-point radius.

    Raises:
        TypeError: if `radius` is not a real number.
        ValueError: if `radius` is negative or non-finite.
    """

    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be a real number.")
    numeric_radius = float(radius)
    if not math.isfinite(numeric_radius):
        raise ValueError("radius must be finite.")
    if numeric_radius < 0.0:
        raise ValueError("radius must be non-negative.")
    return numeric_radius


def _validated_min_samples(min_samples: int) -> int:
    """Validate a positive integer minimum-sample request."""

    if isinstance(min_samples, bool) or not isinstance(min_samples, Integral):
        raise TypeError("min_samples must be an integer.")
    numeric_min_samples = int(min_samples)
    if numeric_min_samples < 1:
        raise ValueError("min_samples must be positive.")
    return numeric_min_samples


def _validated_interval_count(n: int) -> int:
    """Validate the requested number of equal-count regions."""

    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer.")
    numeric_n = int(n)
    if numeric_n < 1 or numeric_n > 20:
        raise ValueError("n must lie in the inclusive range [1, 20].")
    return numeric_n


def _sorted_records_with_distances(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
) -> tuple[tuple[float, PriorityTrainingRecord], ...]:
    """Return records paired with sorted ancestry distances to the target."""

    _validated_sample_count(training_data)
    _validate_ancestry_shapes(training_data, ancestry_coordinate)
    target_values = ancestry_coordinate.values
    record_distances: list[tuple[float, PriorityTrainingRecord]] = []
    for record_index, record in enumerate(training_data.records):
        distance = math.dist(record.ancestry_coordinate, target_values)
        if not math.isfinite(distance):
            raise ValueError(
                "Encountered a non-finite ancestry distance at record index "
                f"{record_index}."
            )
        record_distances.append((float(distance), record))
    record_distances.sort(key=lambda item: item[0])
    return tuple(record_distances)


def _target_variant_dosage(
    record: PriorityTrainingRecord,
    target_variant: PriorityTargetVariant,
) -> float:
    """Read the target dosage from the normalized record payload."""

    if target_variant.name in record.variant_dosages:
        return float(record.variant_dosages[target_variant.name])
    if target_variant.dosage_field in record.variant_dosages:
        return float(record.variant_dosages[target_variant.dosage_field])
    raise ValueError(
        "PriorityTrainingRecord.variant_dosages does not contain the target "
        f"variant {target_variant.name!r}."
    )


def _finite_dosages_for_records(
    records: tuple[PriorityTrainingRecord, ...],
    target_variant: PriorityTargetVariant,
) -> np.ndarray:
    """Return all finite target dosages for the provided records.

    Prepared evaluator artifacts already mean-impute dosage columns before the
    priority-function contract is built, so missing values are not expected in
    normal pipeline execution. Any residual non-finite dosage is skipped here as
    a defensive fallback for synthetic tests or malformed external inputs.
    """

    finite_dosages: list[float] = []
    for record in records:
        dosage = _target_variant_dosage(record, target_variant)
        if math.isfinite(dosage):
            finite_dosages.append(float(dosage))
    return np.asarray(finite_dosages, dtype=float)


def _equal_count_interval_records(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    n: int,
) -> tuple[tuple[tuple[float, float], tuple[PriorityTrainingRecord, ...]], ...]:
    """Return equal-count interval boundaries paired with their records."""

    numeric_n = _validated_interval_count(n)
    record_distances = _sorted_records_with_distances(training_data, ancestry_coordinate)
    sorted_distances = tuple(distance for distance, _ in record_distances)
    upper_bounds = _equal_count_shell_upper_bounds(sorted_distances, numeric_n)

    interval_records: list[tuple[tuple[float, float], tuple[PriorityTrainingRecord, ...]]] = []
    lower_bound = 0.0
    distance_index = 0
    for upper_bound in upper_bounds:
        records_in_interval: list[PriorityTrainingRecord] = []
        while distance_index < len(record_distances):
            distance, record = record_distances[distance_index]
            if distance < lower_bound:
                distance_index += 1
                continue
            if distance >= upper_bound:
                break
            records_in_interval.append(record)
            distance_index += 1
        interval_records.append(((lower_bound, upper_bound), tuple(records_in_interval)))
        lower_bound = upper_bound
    return tuple(interval_records)


def _cumulative_radius_records(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    n: int,
) -> tuple[tuple[float, tuple[PriorityTrainingRecord, ...]], ...]:
    """Return cumulative radii paired with all records inside each radius."""

    numeric_n = _validated_interval_count(n)
    record_distances = _sorted_records_with_distances(training_data, ancestry_coordinate)
    sorted_distances = tuple(distance for distance, _ in record_distances)
    upper_bounds = _equal_count_shell_upper_bounds(sorted_distances, numeric_n)

    cumulative_records: list[tuple[float, tuple[PriorityTrainingRecord, ...]]] = []
    distance_index = 0
    included_records: list[PriorityTrainingRecord] = []
    for upper_bound in upper_bounds:
        while distance_index < len(record_distances):
            distance, record = record_distances[distance_index]
            if distance > upper_bound:
                break
            included_records.append(record)
            distance_index += 1
        cumulative_records.append((upper_bound, tuple(included_records)))
    return tuple(cumulative_records)


def _records_within_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    radius: float,
) -> tuple[PriorityTrainingRecord, ...]:
    """Return all records inside the closed ancestry ball of radius `radius`."""

    numeric_radius = _validated_radius(radius)
    record_distances = _sorted_records_with_distances(training_data, ancestry_coordinate)
    return tuple(
        record
        for distance, record in record_distances
        if distance <= numeric_radius
    )


def _allele_frequency_within_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    radius: float,
) -> float:
    """Return local target-variant allele frequency inside a radius.

    Prepared evaluator artifacts mean-impute dosage columns before these strict
    priority-function contracts are created, so missing target dosages are not
    expected during normal evaluation runs.
    """

    records = _records_within_radius(training_data, ancestry_coordinate, radius)
    dosages = _finite_dosages_for_records(records, target_variant)
    if dosages.size == 0:
        raise ValueError("No finite target dosages were found inside the requested radius.")
    return float(np.mean(dosages) / 2.0)


def _dosage_entropy_for_records(
    records: tuple[PriorityTrainingRecord, ...],
    target_variant: PriorityTargetVariant,
) -> float:
    """Return Shannon entropy of the target-dosage distribution.

    Dosages are binned into the three genotype-centered ranges `(-inf, 0.5)`,
    `[0.5, 1.5)`, and `[1.5, inf)` so the score remains stable even if a
    defensively imputed dosage is fractional. Higher entropy means the local
    dosage distribution uses more of the 0/1/2 genotype support.
    """

    dosages = _finite_dosages_for_records(records, target_variant)
    if dosages.size == 0:
        return 0.0

    counts, _ = np.histogram(dosages, bins=(-math.inf, 0.5, 1.5, math.inf))
    total = int(np.sum(counts))
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def _effect_size_and_standard_error_for_records(
    records: tuple[PriorityTrainingRecord, ...],
    target_variant: PriorityTargetVariant,
    *,
    min_samples: int = _DEFAULT_EFFECT_MIN_SAMPLES,
    l2_penalty: float = _DEFAULT_L2_PENALTY,
    max_iter: int = _DEFAULT_MAX_ITER,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> tuple[float, float]:
    """Return marginal effect size and standard error for one record subset.

    The effect-size estimate is always computed with
    `GenomicsHelpers.effect_size_calculator.estimate_marginal_logistic_effect`
    so these helper tools stay aligned with the oracle's single-variant logistic
    estimator. Prepared evaluator artifacts already mean-impute dosage columns,
    but a final defensive local mean-imputation is still applied to any
    non-finite residual genotype values before the estimator is called.
    """

    numeric_min_samples = _validated_min_samples(min_samples)
    if not records:
        return 0.0, math.inf

    labels = np.asarray([record.label for record in records], dtype=float)
    genotype = np.asarray(
        [_target_variant_dosage(record, target_variant) for record in records],
        dtype=float,
    )
    genotype = impute_missing_genotype_values(genotype)

    if labels.shape[0] < numeric_min_samples:
        return 0.0, math.inf
    if np.unique(labels).size < 2:
        return 0.0, math.inf
    if not np.isfinite(genotype).any():
        return 0.0, math.inf
    if np.allclose(genotype, genotype[0]):
        return 0.0, math.inf

    effect_size = estimate_marginal_logistic_effect(
        labels=labels,
        genotype=genotype,
        covariates=None,
        l2_penalty=l2_penalty,
        max_iter=max_iter,
        tolerance=tolerance,
    )

    design_matrix = build_design_matrix(genotype, None)
    coefficients = fit_logistic_regression_newton(
        design_matrix=design_matrix,
        labels=labels,
        l2_penalty=l2_penalty,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    probabilities = sigmoid(design_matrix @ coefficients)
    weights = probabilities * (1.0 - probabilities)
    penalty = np.eye(design_matrix.shape[1], dtype=float) * l2_penalty
    penalty[0, 0] = 0.0
    weighted_design = design_matrix * weights[:, None]
    hessian = design_matrix.T @ weighted_design + penalty
    covariance = np.linalg.pinv(hessian)
    variance = float(covariance[1, 1])
    if not math.isfinite(variance) or variance < 0.0:
        return float(effect_size), math.inf
    return float(effect_size), math.sqrt(variance)


def _effect_size_within_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    radius: float,
    *,
    min_samples: int = _DEFAULT_EFFECT_MIN_SAMPLES,
) -> float:
    """Return local marginal effect size inside a radius.

    This helper is intentionally not exposed to evolved priority functions so
    candidate programs cannot probe arbitrary radii one-by-one. It returns `0.0`
    when the closed ball contains no data or the marginal effect is not
    identifiable.
    """

    records = _records_within_radius(training_data, ancestry_coordinate, radius)
    effect_size, _ = _effect_size_and_standard_error_for_records(
        records,
        target_variant,
        min_samples=min_samples,
    )
    return effect_size


def _effect_size_standard_error_within_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    radius: float,
    *,
    min_samples: int = _DEFAULT_EFFECT_MIN_SAMPLES,
) -> tuple[float, float]:
    """Return local marginal effect size and standard error inside a radius.

    This helper is intentionally not exposed to evolved priority functions so
    candidate programs cannot probe arbitrary radii one-by-one. It returns
    `(0.0, math.inf)` when the closed ball contains no data or the marginal
    effect is not identifiable.
    """

    records = _records_within_radius(training_data, ancestry_coordinate, radius)
    return _effect_size_and_standard_error_for_records(
        records,
        target_variant,
        min_samples=min_samples,
    )


def _minimum_radius_for_sample_count(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    min_samples: int,
) -> tuple[float, int]:
    """Return the smallest radius enclosing a usable-sample target count.

    If the requested `min_samples` exceeds the number of records with a finite
    target dosage, the count is clamped to the usable total and the returned
    tuple reports that effective sample target. If no usable target dosage is
    available at all, the method returns `(0.0, 0)` instead of raising.
    """

    numeric_min_samples = _validated_min_samples(min_samples)
    usable_distances: list[float] = []
    for distance, record in _sorted_records_with_distances(training_data, ancestry_coordinate):
        dosage = _target_variant_dosage(record, target_variant)
        if math.isfinite(dosage):
            usable_distances.append(distance)

    usable_count = len(usable_distances)
    if usable_count == 0:
        return 0.0, 0

    effective_min_samples = min(numeric_min_samples, usable_count)
    radius = _radius_for_exact_prefix_count(
        tuple(usable_distances),
        effective_min_samples,
    )
    return radius, effective_min_samples


__all__ = [
    "_allele_frequency_within_radius",
    "_cumulative_radius_records",
    "_dosage_entropy_for_records",
    "_effect_size_and_standard_error_for_records",
    "_effect_size_standard_error_within_radius",
    "_effect_size_within_radius",
    "_equal_count_interval_records",
    "_minimum_radius_for_sample_count",
]