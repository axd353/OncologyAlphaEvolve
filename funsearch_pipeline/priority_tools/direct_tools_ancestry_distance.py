# direct_tools
from __future__ import annotations

from .helper_tools_ancestry_distance import _equal_count_shell_upper_bounds
from .helper_tools_ancestry_distance import _exact_count_from_percentage
from .helper_tools_ancestry_distance import _radial_shell_volume
from .helper_tools_ancestry_distance import _radius_for_exact_prefix_count
from .helper_tools_ancestry_distance import _sorted_ancestry_distances
from .contracts import PriorityAncestryCoordinate
from .contracts import PriorityTrainingData


def radius_for_percentage(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    p: float,
) -> float:
    """Return an ancestry-distance radius for a requested sample percentage.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: Target ancestry point `a` around which the radius is
            measured. This is the exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        p: Requested percentage of training samples to include. The value is in
            `[0.0, 100.0]`, so `p = 25.0` means 25 percent. If `p` does not map
            to an integer sample count, the target count is rounded down.

    Output:
        Radius `r1` centered at `ancestry_coordinate` that converts the
        percentage request into a distance cutoff over the training samples. At
        least `p` percent of training samples satisfy
        `euclidean_distance(record.ancestry_coordinate, ancestry_coordinate.values) < r1`.
        Edge cases: if `p` does not land on an integer sample count, the target
        count is rounded down; if multiple samples share the cutoff distance,
        all of those tied samples are kept inside the radius; if `p == 100.0`,
        the radius is pushed just above the largest observed distance.

    Raises:
        TypeError: if `p` is not a real number.
        ValueError: if the dataset is empty, the contract shapes are inconsistent,
            or `p` is out of range.
    """

    sorted_distances = _sorted_ancestry_distances(training_data, ancestry_coordinate)
    prefix_count = _exact_count_from_percentage(len(sorted_distances), p)
    return _radius_for_exact_prefix_count(sorted_distances, prefix_count)


def equal_count_intervals(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    n: int,
) -> list[tuple[float, float]]:
    """Partition ancestry distance into `n` half-open intervals around the target.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: Target ancestry point `a` around which distances are
            measured. This is the exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        n: Number of intervals to produce. Must be an integer in `[1, 20]`.

    Output:
        List of `n` interval pairs `[(0.0, r1), (r1, r2), ..., (r_{n-1}, r_n)]`.
        Each pair represents the half-open interval `[lower, upper)` in
        ancestry-distance from `ancestry_coordinate`, and every training-sample
        distance `d` satisfies `d < r_n`. The target behavior is to split the
        samples as evenly as possible across the `n` intervals.
        Edge cases: if the sample count is not divisible by `n`, earlier
        intervals get one extra target sample until the remainder is exhausted;
        if multiple samples share a boundary distance, all of those tied samples
        are assigned to the earlier interval.

    Raises:
        TypeError: if `n` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are inconsistent,
            or `n` falls outside `[1, 20]`.
    """

    sorted_distances = _sorted_ancestry_distances(training_data, ancestry_coordinate)
    upper_bounds = _equal_count_shell_upper_bounds(sorted_distances, n)
    intervals: list[tuple[float, float]] = []
    lower_bound = 0.0
    for upper_bound in upper_bounds:
        intervals.append((lower_bound, upper_bound))
        lower_bound = upper_bound
    return intervals


def equal_count_interval_densities(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    n: int,
) -> list[float]:
    """Return sample densities for the intervals produced by `equal_count_intervals`.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        n: Number of intervals to produce. Must be an integer in `[1, 20]`.

    Output:
        List of `n` densities, ordered from the closest ancestry region to the
        farthest. Density `i` corresponds to the radial region
        `[r_{i-1}, r_i)` with `r_0 = 0`, where the radii are chosen to split
        the training samples as evenly as possible around
        `ancestry_coordinate`. Each density equals
        `number_of_training_samples_in_region / region_volume`, where
        `region_volume` is the Euclidean shell volume between the two radii in
        `training_data.ancestry_dimension` dimensions.

    Raises:
        TypeError: if `n` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are inconsistent,
            `n` falls outside `[1, 20]`, or a returned interval has zero
            Euclidean region volume while still containing samples.
    """

    intervals = equal_count_intervals(training_data, ancestry_coordinate, n)
    sorted_distances = _sorted_ancestry_distances(training_data, ancestry_coordinate)
    densities: list[float] = []
    distance_index = 0
    for lower_bound, upper_bound in intervals:
        interval_count = 0
        while distance_index < len(sorted_distances):
            distance = sorted_distances[distance_index]
            if distance < lower_bound:
                distance_index += 1
                continue
            if distance >= upper_bound:
                break
            interval_count += 1
            distance_index += 1

        interval_volume = _radial_shell_volume(
            lower_bound,
            upper_bound,
            training_data.ancestry_dimension,
        )
        if interval_volume <= 0.0:
            if interval_count == 0:
                densities.append(0.0)
                continue
            raise ValueError(
                "Density is undefined because an interval returned by equal_count_intervals has zero Euclidean region volume but still contains samples."
            )
        densities.append(interval_count / interval_volume)
    return densities


__all__ = [
    "equal_count_interval_densities",
    "equal_count_intervals",
    "radius_for_percentage",
]