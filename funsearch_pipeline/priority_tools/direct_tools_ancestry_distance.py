# direct_tools
from __future__ import annotations

from .helper_tools_ancestry_distance import _equal_count_shell_upper_bounds
from .helper_tools_ancestry_distance import _exact_count_from_percentage
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


__all__ = ["equal_count_intervals", "radius_for_percentage"]