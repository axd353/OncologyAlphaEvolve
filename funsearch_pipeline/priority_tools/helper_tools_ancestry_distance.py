# helper_tools
from __future__ import annotations

import math
from numbers import Integral
from numbers import Real

from .contracts import PriorityAncestryCoordinate
from .contracts import PriorityTrainingData


def _validated_sample_count(training_data: PriorityTrainingData) -> int:
    """Validate sample-count metadata before any ancestry-distance computation.

    Input:
        training_data: Normalized oracle-train payload exposed to priority
            functions.

    Output:
        Number of training records, equal to `len(training_data.records)`.

    Raises:
        ValueError: if the dataset is empty or `sample_count` disagrees with the
            actual record count.
    """

    observed_sample_count = len(training_data.records)
    if training_data.sample_count != observed_sample_count:
        raise ValueError(
            "PriorityTrainingData.sample_count must match len(training_data.records)."
        )
    if observed_sample_count == 0:
        raise ValueError("At least one training sample is required.")
    return observed_sample_count


def _validate_ancestry_shapes(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
) -> None:
    """Validate ancestry dimensionality for the target point and all records.

    Input:
        training_data: Normalized oracle-train payload.
        ancestry_coordinate: Target ancestry point around which distances will be
            measured.

    Output:
        No return value. The function only validates contract consistency.

    Raises:
        ValueError: if the target ancestry dimension or any training-record
            ancestry vector has the wrong length.
    """

    target_dimension = len(ancestry_coordinate.values)
    if ancestry_coordinate.dimension != target_dimension:
        raise ValueError(
            "PriorityAncestryCoordinate.dimension must match len(values)."
        )
    if training_data.ancestry_dimension != target_dimension:
        raise ValueError(
            "training_data.ancestry_dimension must match the target ancestry dimension."
        )
    for record_index, record in enumerate(training_data.records):
        record_dimension = len(record.ancestry_coordinate)
        if record_dimension != training_data.ancestry_dimension:
            raise ValueError(
                "Training record ancestry dimension mismatch at index "
                f"{record_index}."
            )


def _sorted_ancestry_distances(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
) -> tuple[float, ...]:
    """Compute sorted Euclidean ancestry distances to a target point.

    Input:
        training_data: Normalized oracle-train payload.
        ancestry_coordinate: Target ancestry point around which distances will be
            measured.

    Output:
        Tuple of finite Euclidean distances, sorted in ascending order.

    Raises:
        ValueError: if the dataset is empty, contract shapes are inconsistent, or
            a non-finite distance is encountered.
    """

    _validated_sample_count(training_data)
    _validate_ancestry_shapes(training_data, ancestry_coordinate)
    target_values = ancestry_coordinate.values
    distances: list[float] = []
    for record_index, record in enumerate(training_data.records):
        distance = math.dist(record.ancestry_coordinate, target_values)
        if not math.isfinite(distance):
            raise ValueError(
                "Encountered a non-finite ancestry distance at record index "
                f"{record_index}."
            )
        distances.append(float(distance))
    distances.sort()
    return tuple(distances)


def _exact_count_from_percentage(sample_count: int, percentage: Real) -> int:
    """Convert a percentage request into an empirical sample count.

    Input:
        sample_count: Number of training samples available.
        percentage: Requested percentage in the inclusive range `[0.0, 100.0]`.

    Output:
        Integer number of samples that must be enclosed by a radius. When the
        requested percentage does not land exactly on an integer sample count,
        the value is rounded down with `floor(...)`.

    Raises:
        TypeError: if `percentage` is not a real number.
        ValueError: if the percentage is out of range.
    """

    if isinstance(percentage, bool) or not isinstance(percentage, Real):
        raise TypeError("p must be a real number.")
    numeric_percentage = float(percentage)
    if not math.isfinite(numeric_percentage):
        raise ValueError("p must be finite.")
    if numeric_percentage < 0.0 or numeric_percentage > 100.0:
        raise ValueError("p must lie in the inclusive range [0.0, 100.0].")

    exact_count = sample_count * numeric_percentage / 100.0
    return math.floor(exact_count)


def _radius_for_exact_prefix_count(
    sorted_distances: tuple[float, ...],
    prefix_count: int,
) -> float:
    """Return the smallest radius whose strict interior covers a prefix target.

    Input:
        sorted_distances: Distances sorted in ascending order.
        prefix_count: Number of smallest distances that must satisfy
            `distance < radius` at minimum.

    Output:
        Smallest floating-point radius whose strict interior contains at least
        `prefix_count` sorted distances under the strict inequality
        `distance < radius`. If multiple samples share the boundary distance,
        all boundary-tied samples are assigned to the inner region.

    Raises:
        TypeError: if `prefix_count` is not an integer.
        ValueError: if `prefix_count` is out of range, the distance sequence is
            empty.
    """

    if isinstance(prefix_count, bool) or not isinstance(prefix_count, Integral):
        raise TypeError("prefix_count must be an integer.")
    sample_count = len(sorted_distances)
    if sample_count == 0:
        raise ValueError("At least one sorted distance is required.")

    numeric_prefix_count = int(prefix_count)
    if numeric_prefix_count < 0 or numeric_prefix_count > sample_count:
        raise ValueError(
            f"prefix_count must lie in the inclusive range [0, {sample_count}]."
        )
    if numeric_prefix_count == 0:
        return 0.0

    lower_distance = sorted_distances[numeric_prefix_count - 1]
    if numeric_prefix_count == sample_count:
        return math.nextafter(lower_distance, math.inf)

    return math.nextafter(lower_distance, math.inf)


def _equal_count_shell_upper_bounds(
    sorted_distances: tuple[float, ...],
    shell_count: int,
) -> tuple[float, ...]:
    """Build ancestry-distance shell boundaries from near-equal count targets.

    Input:
        sorted_distances: Distances sorted in ascending order.
        shell_count: Number of shells to create. Must be an integer in
            `[1, 30]`.

    Output:
        Tuple `(r1, r2, ..., r_n)` of shell upper bounds. Each corresponding
        half-open shell `[r_{i-1}, r_i)` is built from a target partition of the
        sorted distances where `r_0 = 0`. When the sample count is not divisible
        by `shell_count`, the earlier shells receive one extra target sample
        until the remainder is exhausted. If multiple samples share a boundary
        distance, all of those boundary-tied samples are assigned to the earlier
        shell, so actual shell sizes can become unequal.

    Raises:
        TypeError: if `shell_count` is not an integer.
        ValueError: if the shell count is outside `[1, 30]`.
    """

    if isinstance(shell_count, bool) or not isinstance(shell_count, Integral):
        raise TypeError("n must be an integer.")
    numeric_shell_count = int(shell_count)
    if numeric_shell_count < 1 or numeric_shell_count > 30:
        raise ValueError("n must lie in the inclusive range [1, 30].")

    sample_count = len(sorted_distances)
    base_shell_size = sample_count // numeric_shell_count
    remainder = sample_count % numeric_shell_count
    prefix_count = 0
    upper_bounds: list[float] = []
    for shell_index in range(numeric_shell_count):
        shell_size = base_shell_size + (1 if shell_index < remainder else 0)
        prefix_count += shell_size
        upper_bounds.append(
            _radius_for_exact_prefix_count(sorted_distances, prefix_count)
        )
    return tuple(upper_bounds)


def _radial_shell_volume(
    lower_radius: float,
    upper_radius: float,
    dimension: int,
) -> float:
    """Return the Euclidean volume of a radial shell.

    Input:
        lower_radius: Inner shell radius. Must be finite and non-negative.
        upper_radius: Outer shell radius. Must be finite and at least as large
            as `lower_radius`.
        dimension: Ambient ancestry-space dimension. Must be a positive integer.

    Output:
        Volume of the region `{x : lower_radius <= ||x|| < upper_radius}` in the
        given Euclidean dimension.

    Raises:
        TypeError: if `dimension` is not an integer.
        ValueError: if either radius is non-finite, negative, ordered
            incorrectly, or if `dimension` is not positive.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, Integral):
        raise TypeError("dimension must be an integer.")
    numeric_dimension = int(dimension)
    if numeric_dimension < 1:
        raise ValueError("dimension must be positive.")
    if not math.isfinite(lower_radius) or not math.isfinite(upper_radius):
        raise ValueError("Shell radii must be finite.")
    if lower_radius < 0.0 or upper_radius < 0.0:
        raise ValueError("Shell radii must be non-negative.")
    if upper_radius < lower_radius:
        raise ValueError("upper_radius must be at least lower_radius.")

    unit_ball_volume = math.pi ** (numeric_dimension / 2.0) / math.gamma(
        numeric_dimension / 2.0 + 1.0
    )
    return unit_ball_volume * (
        upper_radius ** numeric_dimension - lower_radius ** numeric_dimension
    )