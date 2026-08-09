# helper_tools
from __future__ import annotations

import math
import random
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
from .helper_tools_ancestry_distance import _exact_count_from_percentage
from .helper_tools_ancestry_distance import _equal_count_shell_upper_bounds
from .helper_tools_ancestry_distance import _radius_for_exact_prefix_count
from .helper_tools_ancestry_distance import _validated_sample_count
from .helper_tools_ancestry_distance import _validate_ancestry_shapes

_DEFAULT_EFFECT_MIN_SAMPLES = 25
_DEFAULT_L2_PENALTY = 1e-6
_DEFAULT_MAX_ITER = 50
_DEFAULT_TOLERANCE = 1e-8
_DEFAULT_NOVELTY_PERCENTAGE = 10.0
_DEFAULT_NOVELTY_BASELINE_SAMPLE_SIZE = 100
_NOVELTY_RANDOM_SEED = 0


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
    if numeric_n < 1 or numeric_n > 30:
        raise ValueError("n must lie in the inclusive range [1, 30].")
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


def _sorted_distances_to_values(
    training_data: PriorityTrainingData,
    target_values: tuple[float, ...],
    *,
    skip_record_index: int | None = None,
) -> tuple[float, ...]:
    """Return sorted ancestry distances to an arbitrary ancestry point."""

    _validated_sample_count(training_data)
    target_dimension = len(target_values)
    if training_data.ancestry_dimension != target_dimension:
        raise ValueError(
            "training_data.ancestry_dimension must match the target ancestry dimension."
        )

    distances: list[float] = []
    for record_index, record in enumerate(training_data.records):
        if skip_record_index is not None and record_index == skip_record_index:
            continue
        distance = math.dist(record.ancestry_coordinate, target_values)
        if not math.isfinite(distance):
            raise ValueError(
                "Encountered a non-finite ancestry distance at record index "
                f"{record_index}."
            )
        distances.append(float(distance))
    distances.sort()
    return tuple(distances)


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


def _label_entropy_for_records(
    records: tuple[PriorityTrainingRecord, ...],
) -> float:
    """Return Shannon entropy of the label distribution."""

    if not records:
        return 0.0

    labels = np.asarray([record.label for record in records], dtype=float)
    _, counts = np.unique(labels, return_counts=True)
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


def _variant_dosage_matrix(
    records: tuple[PriorityTrainingRecord, ...],
    variant_names: tuple[str, ...],
) -> np.ndarray:
    """Return a records-by-variants dosage matrix in training-data order."""

    if not records:
        return np.zeros((0, len(variant_names)), dtype=float)

    matrix = np.asarray(
        [
            [float(record.variant_dosages[variant_name]) for variant_name in variant_names]
            for record in records
        ],
        dtype=float,
    )
    if np.isfinite(matrix).all():
        return matrix

    imputed = matrix.copy()
    for column_index in range(imputed.shape[1]):
        imputed[:, column_index] = impute_missing_genotype_values(imputed[:, column_index])
    return imputed


def _target_variant_column_index(
    training_data: PriorityTrainingData,
    target_variant: PriorityTargetVariant,
) -> int:
    """Resolve the aligned target-variant column index."""

    if 0 <= target_variant.column_index < len(training_data.variant_names):
        if training_data.variant_names[target_variant.column_index] == target_variant.name:
            return int(target_variant.column_index)
    try:
        return training_data.variant_names.index(target_variant.name)
    except ValueError as error:
        raise ValueError(
            f"Target variant {target_variant.name!r} is not aligned to training_data.variant_names."
        ) from error


def _correlation_profile(
    dosage_matrix: np.ndarray,
    target_column_index: int,
) -> np.ndarray:
    """Return target-vs-other dosage correlations for one dosage matrix."""

    variant_count = dosage_matrix.shape[1]
    if dosage_matrix.shape[0] < 2 or variant_count <= 1:
        return np.zeros((0,), dtype=float)

    target_values = dosage_matrix[:, target_column_index]
    target_centered = target_values - np.mean(target_values)
    target_scale = float(np.linalg.norm(target_centered))
    if target_scale == 0.0:
        return np.full((variant_count - 1,), np.nan, dtype=float)

    profile: list[float] = []
    for column_index in range(variant_count):
        if column_index == target_column_index:
            continue
        other_values = dosage_matrix[:, column_index]
        other_centered = other_values - np.mean(other_values)
        other_scale = float(np.linalg.norm(other_centered))
        if other_scale == 0.0:
            profile.append(math.nan)
            continue
        correlation = float(np.dot(target_centered, other_centered) / (target_scale * other_scale))
        profile.append(max(-1.0, min(1.0, correlation)))
    return np.asarray(profile, dtype=float)


def _ld_profile_similarity_for_records(
    local_records: tuple[PriorityTrainingRecord, ...],
    global_records: tuple[PriorityTrainingRecord, ...],
    training_data: PriorityTrainingData,
    target_variant: PriorityTargetVariant,
) -> float:
    """Compare local and global target-variant dosage-correlation profiles."""

    if not local_records or len(training_data.variant_names) <= 1:
        return 0.0

    target_column_index = _target_variant_column_index(training_data, target_variant)
    global_profile = _correlation_profile(
        _variant_dosage_matrix(global_records, training_data.variant_names),
        target_column_index,
    )
    local_profile = _correlation_profile(
        _variant_dosage_matrix(local_records, training_data.variant_names),
        target_column_index,
    )
    if global_profile.size == 0 or local_profile.size == 0:
        return 0.0

    finite_mask = np.isfinite(global_profile) & np.isfinite(local_profile)
    if not np.any(finite_mask):
        return 0.0
    mean_absolute_difference = float(np.mean(np.abs(local_profile[finite_mask] - global_profile[finite_mask])))
    similarity = 1.0 - mean_absolute_difference / 2.0
    return max(0.0, min(1.0, similarity))


def _standardized_effect_change_for_interval_records(
    interval_records: tuple[tuple[tuple[float, float], tuple[PriorityTrainingRecord, ...]], ...],
    target_variant: PriorityTargetVariant,
    *,
    min_samples: int = _DEFAULT_EFFECT_MIN_SAMPLES,
) -> list[float]:
    """Return uncertainty-adjusted adjacent effect changes."""

    interval_results = [
        _effect_size_and_standard_error_for_records(
            records,
            target_variant,
            min_samples=min_samples,
        )
        for _, records in interval_records
    ]
    standardized_changes: list[float] = []
    for left_result, right_result in zip(interval_results[:-1], interval_results[1:]):
        left_effect, left_standard_error = left_result
        right_effect, right_standard_error = right_result
        if not math.isfinite(left_standard_error) or not math.isfinite(right_standard_error):
            standardized_changes.append(0.0)
            continue
        denominator = math.hypot(left_standard_error, right_standard_error)
        if denominator <= 0.0:
            standardized_changes.append(0.0)
            continue
        standardized_changes.append(abs(left_effect - right_effect) / denominator)
    return standardized_changes


def _novelty_reference_radius(
    training_data: PriorityTrainingData,
    target_values: tuple[float, ...],
    *,
    percentage: float = _DEFAULT_NOVELTY_PERCENTAGE,
) -> float:
    """Return the radius enclosing a fixed small training percentage."""

    sample_count = _validated_sample_count(training_data)
    target_count = max(1, _exact_count_from_percentage(sample_count, percentage))
    sorted_distances = _sorted_distances_to_values(training_data, target_values)
    return _radius_for_exact_prefix_count(
        sorted_distances,
        min(target_count, len(sorted_distances)),
    )


def _ancestry_novelty_score(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    *,
    percentage: float = _DEFAULT_NOVELTY_PERCENTAGE,
    baseline_sample_size: int = _DEFAULT_NOVELTY_BASELINE_SAMPLE_SIZE,
) -> float:
    """Return normalized ancestry-support sparsity relative to training baseline."""

    _validated_sample_count(training_data)
    _validate_ancestry_shapes(training_data, ancestry_coordinate)
    numeric_baseline_sample_size = _validated_min_samples(baseline_sample_size)

    target_radius = _novelty_reference_radius(
        training_data,
        ancestry_coordinate.values,
        percentage=percentage,
    )

    sample_count = len(training_data.records)
    sampled_indices = list(range(sample_count))
    if sample_count > numeric_baseline_sample_size:
        sampled_indices = random.Random(_NOVELTY_RANDOM_SEED).sample(
            sampled_indices,
            k=numeric_baseline_sample_size,
        )

    target_count = max(1, _exact_count_from_percentage(sample_count, percentage))
    baseline_radii: list[float] = []
    for record_index in sampled_indices:
        record = training_data.records[record_index]
        sorted_distances = _sorted_distances_to_values(
            training_data,
            record.ancestry_coordinate,
            skip_record_index=record_index,
        )
        if not sorted_distances:
            baseline_radii.append(0.0)
            continue
        baseline_radii.append(
            _radius_for_exact_prefix_count(
                sorted_distances,
                min(target_count, len(sorted_distances)),
            )
        )

    if not baseline_radii:
        return 1.0 if target_radius == 0.0 else float(target_radius)
    baseline_median = float(np.median(np.asarray(baseline_radii, dtype=float)))
    if baseline_median <= 0.0:
        return 1.0 if target_radius <= 0.0 else float(target_radius)
    return float(target_radius / baseline_median)


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


def _minimum_radius_for_training_percentage(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    percentage: float,
) -> tuple[float, float]:
    """Return the smallest radius for a requested training-data percentage.

    The requested percentage is interpreted against the full training-set size,
    then converted to an integer target count with floor rounding. The returned
    effective percentage reports the actual usable-sample fraction enclosed
    after rounding and after clamping to the count of records with finite target
    dosages. If no usable target dosage is available at all, the method returns
    `(0.0, 0.0)` instead of raising.
    """

    sorted_records = _sorted_records_with_distances(
        training_data,
        ancestry_coordinate,
    )
    sample_count = len(sorted_records)
    requested_min_samples = _exact_count_from_percentage(sample_count, percentage)

    usable_distances: list[float] = []
    for distance, record in sorted_records:
        dosage = _target_variant_dosage(record, target_variant)
        if math.isfinite(dosage):
            usable_distances.append(distance)

    usable_count = len(usable_distances)
    if usable_count == 0:
        return 0.0, 0.0

    effective_min_samples = min(requested_min_samples, usable_count)
    radius = _radius_for_exact_prefix_count(
        tuple(usable_distances),
        effective_min_samples,
    )
    effective_percentage = 100.0 * effective_min_samples / sample_count
    return radius, effective_percentage


__all__ = [
    "_ancestry_novelty_score",
    "_allele_frequency_within_radius",
    "_cumulative_radius_records",
    "_dosage_entropy_for_records",
    "_effect_size_and_standard_error_for_records",
    "_effect_size_standard_error_within_radius",
    "_effect_size_within_radius",
    "_equal_count_interval_records",
    "_label_entropy_for_records",
    "_ld_profile_similarity_for_records",
    "_minimum_radius_for_training_percentage",
    "_minimum_radius_for_sample_count",
    "_standardized_effect_change_for_interval_records",
]