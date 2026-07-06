from __future__ import annotations

import math

import pytest

from funsearch_pipeline.priority_tools import equal_count_interval_densities
from funsearch_pipeline.priority_tools import equal_count_intervals
from funsearch_pipeline.priority_tools import radius_for_percentage
from funsearch_pipeline.priority_tools.contracts import PriorityAncestryCoordinate
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingData
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingRecord


def _make_training_data(*coordinates: float) -> PriorityTrainingData:
    records = tuple(
        PriorityTrainingRecord(
            label=float(index % 2),
            ancestry_coordinate=(float(coordinate),),
            variant_dosages={},
            covariates=None,
        )
        for index, coordinate in enumerate(coordinates)
    )
    return PriorityTrainingData(
        records=records,
        variant_names=(),
        variant_dosage_fields=(),
        covariate_names=(),
        sample_count=len(records),
        ancestry_dimension=1,
        has_additional_covariates=False,
    )


def _distances(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
) -> list[float]:
    return [
        math.dist(record.ancestry_coordinate, ancestry_coordinate.values)
        for record in training_data.records
    ]


def test_radius_for_percentage_returns_exact_empirical_cutoff() -> None:
    training_data = _make_training_data(-5.0, -1.0, 2.0, 4.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius = radius_for_percentage(training_data, ancestry_coordinate, 50.0)

    distances = _distances(training_data, ancestry_coordinate)
    assert sum(distance < radius for distance in distances) == 2
    assert 2.0 < radius < 4.0


def test_radius_for_percentage_rounds_down_non_exact_percentage() -> None:
    training_data = _make_training_data(-5.0, -1.0, 2.0, 4.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius = radius_for_percentage(training_data, ancestry_coordinate, 30.0)

    distances = _distances(training_data, ancestry_coordinate)
    assert sum(distance < radius for distance in distances) == 1
    assert 1.0 < radius < 2.0


def test_radius_for_percentage_assigns_tied_boundary_distance_to_inner_region() -> None:
    training_data = _make_training_data(-1.0, 1.0, 3.0, 4.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius = radius_for_percentage(training_data, ancestry_coordinate, 25.0)

    distances = _distances(training_data, ancestry_coordinate)
    assert radius > 1.0
    assert sum(distance < radius for distance in distances) == 2


def test_equal_count_intervals_partition_samples_evenly() -> None:
    training_data = _make_training_data(-1.0, 2.0, -3.0, 4.0, -5.0, 6.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    intervals = equal_count_intervals(training_data, ancestry_coordinate, 3)

    distances = _distances(training_data, ancestry_coordinate)
    counts = [
        sum(lower <= distance < upper for distance in distances)
        for lower, upper in intervals
    ]
    assert len(intervals) == 3
    assert counts == [2, 2, 2]
    assert intervals[0][0] == 0.0
    assert intervals[0][1] == intervals[1][0]
    assert intervals[1][1] == intervals[2][0]
    assert all(distance < intervals[-1][1] for distance in distances)


def test_equal_count_intervals_distributes_non_divisible_sample_count() -> None:
    training_data = _make_training_data(-1.0, 2.0, -3.0, 4.0, -5.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    intervals = equal_count_intervals(training_data, ancestry_coordinate, 3)

    distances = _distances(training_data, ancestry_coordinate)
    counts = [
        sum(lower <= distance < upper for distance in distances)
        for lower, upper in intervals
    ]
    assert counts == [2, 2, 1]


def test_equal_count_intervals_assigns_tied_boundary_distance_to_earlier_interval() -> None:
    training_data = _make_training_data(-1.0, 1.0, -1.0, 2.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    intervals = equal_count_intervals(training_data, ancestry_coordinate, 2)

    distances = _distances(training_data, ancestry_coordinate)
    counts = [
        sum(lower <= distance < upper for distance in distances)
        for lower, upper in intervals
    ]
    assert counts == [3, 1]


def test_equal_count_interval_densities_match_interval_counts_over_1d_shell_length() -> None:
    training_data = _make_training_data(-1.0, 2.0, -3.0, 4.0, -5.0, 6.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    densities = equal_count_interval_densities(training_data, ancestry_coordinate, 3)

    assert len(densities) == 3
    assert math.isclose(densities[0], 0.5)
    assert math.isclose(densities[1], 0.5)
    assert math.isclose(densities[2], 0.5)


def test_equal_count_interval_densities_return_zero_for_empty_zero_volume_intervals() -> None:
    training_data = _make_training_data(-1.0, 3.0)
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    densities = equal_count_interval_densities(training_data, ancestry_coordinate, 4)

    assert math.isclose(densities[0], 0.5)
    assert math.isclose(densities[1], 0.25)
    assert densities[2:] == [0.0, 0.0]
