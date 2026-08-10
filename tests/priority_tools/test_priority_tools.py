from __future__ import annotations

import math

import funsearch_pipeline.priority_tools.helper_tools_variant_statistics as variant_helpers
import pytest

from GenomicsHelpers.effect_size_calculator import estimate_marginal_logistic_effect as real_estimate_marginal_logistic_effect
from funsearch_pipeline.priority_tools import ancestry_novelty_score
from funsearch_pipeline.priority_tools import dosage_entropy_by_cumulative_radius
from funsearch_pipeline.priority_tools import dosage_entropy_by_interval
from funsearch_pipeline.priority_tools import equal_count_interval_densities
from funsearch_pipeline.priority_tools import equal_count_intervals
from funsearch_pipeline.priority_tools import effect_size_by_cumulative_radius
from funsearch_pipeline.priority_tools import effect_size_by_interval
from funsearch_pipeline.priority_tools import effect_size_standard_error_by_cumulative_radius
from funsearch_pipeline.priority_tools import effect_size_standard_error_by_interval
from funsearch_pipeline.priority_tools import label_entropy_by_cumulative_radius
from funsearch_pipeline.priority_tools import minimum_radius_for_training_percentage
from funsearch_pipeline.priority_tools import radius_for_percentage
from funsearch_pipeline.priority_tools import standardized_effect_change_by_interval
from funsearch_pipeline.priority_tools import target_ld_similarity_by_cumulative_radius
from funsearch_pipeline.priority_tools.contracts import PriorityAncestryCoordinate
from funsearch_pipeline.priority_tools.contracts import PriorityTargetVariant
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingData
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingRecord
from funsearch_pipeline.priority_tools.direct_tools_variant_statistics import minimum_radius_for_sample_count


def _make_training_data(
    coordinates: tuple[float, ...],
    *,
    labels: tuple[float, ...] | None = None,
    dosages: tuple[float, ...] | None = None,
) -> PriorityTrainingData:
    if labels is None:
        labels = tuple(float(index % 2) for index in range(len(coordinates)))
    if dosages is None:
        dosages = tuple(float(index % 3) for index in range(len(coordinates)))
    if len(labels) != len(coordinates):
        raise ValueError("labels must align with coordinates")
    if len(dosages) != len(coordinates):
        raise ValueError("dosages must align with coordinates")

    records = tuple(
        PriorityTrainingRecord(
            label=float(labels[index]),
            ancestry_coordinate=(float(coordinate),),
            variant_dosages={"rs1": float(dosages[index])},
            covariates=None,
        )
        for index, coordinate in enumerate(coordinates)
    )
    return PriorityTrainingData(
        records=records,
	    variant_names=("rs1",),
	    variant_dosage_fields=("dosage__rs1",),
        covariate_names=(),
        sample_count=len(records),
        ancestry_dimension=1,
        has_additional_covariates=False,
    )


def _target_variant() -> PriorityTargetVariant:
    return PriorityTargetVariant(name="rs1", dosage_field="dosage__rs1", column_index=0)


def _make_multivariant_training_data(
    coordinates: tuple[float, ...],
    *,
    labels: tuple[float, ...],
    dosage_rows: tuple[tuple[float, ...], ...],
) -> PriorityTrainingData:
    variant_count = len(dosage_rows[0])
    variant_names = tuple(f"rs{index + 1}" for index in range(variant_count))
    records = tuple(
        PriorityTrainingRecord(
            label=float(labels[index]),
            ancestry_coordinate=(float(coordinate),),
            variant_dosages={
                variant_name: float(dosage_rows[index][variant_index])
                for variant_index, variant_name in enumerate(variant_names)
            },
            covariates=None,
        )
        for index, coordinate in enumerate(coordinates)
    )
    return PriorityTrainingData(
        records=records,
	    variant_names=variant_names,
	    variant_dosage_fields=tuple(f"dosage__{variant_name}" for variant_name in variant_names),
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
    training_data = _make_training_data((-5.0, -1.0, 2.0, 4.0))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius = radius_for_percentage(training_data, ancestry_coordinate, 50.0)

    distances = _distances(training_data, ancestry_coordinate)
    assert sum(distance < radius for distance in distances) == 2
    assert 2.0 < radius < 4.0


def test_radius_for_percentage_rounds_down_non_exact_percentage() -> None:
    training_data = _make_training_data((-5.0, -1.0, 2.0, 4.0))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius = radius_for_percentage(training_data, ancestry_coordinate, 30.0)

    distances = _distances(training_data, ancestry_coordinate)
    assert sum(distance < radius for distance in distances) == 1
    assert 1.0 < radius < 2.0


def test_radius_for_percentage_assigns_tied_boundary_distance_to_inner_region() -> None:
    training_data = _make_training_data((-1.0, 1.0, 3.0, 4.0))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius = radius_for_percentage(training_data, ancestry_coordinate, 25.0)

    distances = _distances(training_data, ancestry_coordinate)
    assert radius > 1.0
    assert sum(distance < radius for distance in distances) == 2


def test_equal_count_intervals_partition_samples_evenly() -> None:
    training_data = _make_training_data((-1.0, 2.0, -3.0, 4.0, -5.0, 6.0))
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
    training_data = _make_training_data((-1.0, 2.0, -3.0, 4.0, -5.0))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    intervals = equal_count_intervals(training_data, ancestry_coordinate, 3)

    distances = _distances(training_data, ancestry_coordinate)
    counts = [
        sum(lower <= distance < upper for distance in distances)
        for lower, upper in intervals
    ]
    assert counts == [2, 2, 1]


def test_equal_count_intervals_assigns_tied_boundary_distance_to_earlier_interval() -> None:
    training_data = _make_training_data((-1.0, 1.0, -1.0, 2.0))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    intervals = equal_count_intervals(training_data, ancestry_coordinate, 2)

    distances = _distances(training_data, ancestry_coordinate)
    counts = [
        sum(lower <= distance < upper for distance in distances)
        for lower, upper in intervals
    ]
    assert counts == [3, 1]


def test_equal_count_intervals_accept_upper_bound_of_thirty() -> None:
    training_data = _make_training_data(tuple(float(index) for index in range(30)))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    intervals = equal_count_intervals(training_data, ancestry_coordinate, 30)

    assert len(intervals) == 30


def test_equal_count_intervals_reject_more_than_thirty() -> None:
    training_data = _make_training_data(tuple(float(index) for index in range(31)))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    with pytest.raises(ValueError, match=r"\[1, 30\]"):
        equal_count_intervals(training_data, ancestry_coordinate, 31)


def test_equal_count_interval_densities_match_interval_counts_over_1d_shell_length() -> None:
    training_data = _make_training_data((-1.0, 2.0, -3.0, 4.0, -5.0, 6.0))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    densities = equal_count_interval_densities(training_data, ancestry_coordinate, 3)

    assert len(densities) == 3
    assert math.isclose(densities[0], 0.5)
    assert math.isclose(densities[1], 0.5)
    assert math.isclose(densities[2], 0.5)


def test_equal_count_interval_densities_return_zero_for_empty_zero_volume_intervals() -> None:
    training_data = _make_training_data((-1.0, 3.0))
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    densities = equal_count_interval_densities(training_data, ancestry_coordinate, 4)

    assert math.isclose(densities[0], 0.5)
    assert math.isclose(densities[1], 0.25)
    assert densities[2:] == [0.0, 0.0]


def test_minimum_radius_for_sample_count_clamps_to_usable_total() -> None:
    training_data = _make_training_data(
        (-4.0, -1.0, 2.0, 5.0),
        dosages=(0.0, float("nan"), 1.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius, effective_min_samples = minimum_radius_for_sample_count(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        10,
    )

    distances = _distances(training_data, ancestry_coordinate)
    usable_distances = [
        distance
        for distance, record in zip(distances, training_data.records)
        if math.isfinite(record.variant_dosages["rs1"])
    ]
    assert effective_min_samples == 3
    assert sum(distance < radius for distance in usable_distances) == 3


def test_minimum_radius_for_training_percentage_rounds_down_and_reports_effective_percentage() -> None:
    training_data = _make_training_data(
        (-4.0, -1.0, 2.0, 5.0),
        dosages=(0.0, float("nan"), 1.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius, effective_percentage = minimum_radius_for_training_percentage(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        30.0,
    )

    distances = _distances(training_data, ancestry_coordinate)
    usable_distances = [
        distance
        for distance, record in zip(distances, training_data.records)
        if math.isfinite(record.variant_dosages["rs1"])
    ]
    assert effective_percentage == pytest.approx(25.0)
    assert sum(distance < radius for distance in usable_distances) == 1


def test_minimum_radius_for_training_percentage_clamps_to_usable_total() -> None:
    training_data = _make_training_data(
        (-4.0, -1.0, 2.0, 5.0),
        dosages=(0.0, float("nan"), 1.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radius, effective_percentage = minimum_radius_for_training_percentage(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        100.0,
    )

    distances = _distances(training_data, ancestry_coordinate)
    usable_distances = [
        distance
        for distance, record in zip(distances, training_data.records)
        if math.isfinite(record.variant_dosages["rs1"])
    ]
    assert effective_percentage == pytest.approx(75.0)
    assert sum(distance < radius for distance in usable_distances) == 3


def test_dosage_entropy_by_interval_tracks_per_ring_dosage_diversity() -> None:
    training_data = _make_training_data(
        (-1.0, 1.0, -3.0, 3.0),
        dosages=(0.0, 0.0, 0.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    entropies = dosage_entropy_by_interval(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        2,
    )

    assert entropies[0] == pytest.approx(0.0)
    assert entropies[1] == pytest.approx(1.0)


def test_dosage_entropy_by_cumulative_radius_returns_radii_with_entropy() -> None:
    training_data = _make_training_data(
        (-1.0, 1.0, -3.0, 3.0),
        dosages=(0.0, 0.0, 0.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radii_and_entropy = dosage_entropy_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        2,
    )

    assert len(radii_and_entropy) == 2
    assert radii_and_entropy[0][0] < radii_and_entropy[1][0]
    assert radii_and_entropy[0][1] == pytest.approx(0.0)
    assert radii_and_entropy[1][1] == pytest.approx(0.8112781244591328)


def test_label_entropy_by_cumulative_radius_tracks_case_control_balance() -> None:
    training_data = _make_training_data(
        (-1.0, 1.0, -3.0, 3.0),
        labels=(0.0, 0.0, 0.0, 1.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radii_and_entropy = label_entropy_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        2,
    )

    assert len(radii_and_entropy) == 2
    assert radii_and_entropy[0][1] == pytest.approx(0.0)
    assert radii_and_entropy[1][1] == pytest.approx(0.8112781244591328)


def test_ancestry_novelty_score_is_larger_for_farther_target() -> None:
    coordinates = tuple(float(index) / 20.0 for index in range(200))
    training_data = _make_training_data(coordinates)
    near_coordinate = PriorityAncestryCoordinate(values=(0.5,), dimension=1)
    far_coordinate = PriorityAncestryCoordinate(values=(20.0,), dimension=1)

    near_score = ancestry_novelty_score(training_data, near_coordinate)
    far_score = ancestry_novelty_score(training_data, far_coordinate)

    assert near_score > 0.0
    assert far_score > near_score


def test_effect_size_by_interval_uses_shared_logistic_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    training_data = _make_training_data(
        (-1.0, 1.0, -3.0, 3.0),
        labels=(0.0, 1.0, 0.0, 1.0),
        dosages=(0.0, 1.0, 1.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)
    call_counter = {"count": 0}

    def counting_estimator(*args: object, **kwargs: object) -> float:
        call_counter["count"] += 1
        return real_estimate_marginal_logistic_effect(*args, **kwargs)

    monkeypatch.setattr(
        variant_helpers,
        "estimate_marginal_logistic_effect",
        counting_estimator,
    )

    effect_sizes = effect_size_by_interval(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        2,
        min_samples=2,
    )

    assert len(effect_sizes) == 2
    assert call_counter["count"] == 2
    assert all(math.isfinite(value) for value in effect_sizes)


def test_effect_size_standard_error_by_interval_returns_infinity_for_unidentifiable_region() -> None:
    training_data = _make_training_data(
        (-1.0, 3.0),
        labels=(0.0, 1.0),
        dosages=(0.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    standard_errors = effect_size_standard_error_by_interval(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        4,
        min_samples=1,
    )

    assert standard_errors == [math.inf, math.inf, math.inf, math.inf]


def test_standardized_effect_change_by_interval_returns_adjacent_scores() -> None:
    training_data = _make_training_data(
        (-1.0, 1.0, -3.0, 3.0),
        labels=(0.0, 1.0, 0.0, 1.0),
        dosages=(0.0, 1.0, 1.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    standardized_changes = standardized_effect_change_by_interval(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        2,
        min_samples=2,
    )

    assert len(standardized_changes) == 1
    assert math.isfinite(standardized_changes[0])
    assert standardized_changes[0] >= 0.0


def test_effect_size_by_cumulative_radius_accepts_upper_bound_of_thirty() -> None:
    coordinates = tuple(float(index) for index in range(30))
    labels = tuple(float(index % 2) for index in range(30))
    dosages = tuple(float(index % 3) for index in range(30))
    training_data = _make_training_data(
        coordinates,
        labels=labels,
        dosages=dosages,
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radii_and_effects = effect_size_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        30,
        min_samples=1,
    )

    assert len(radii_and_effects) == 30


def test_effect_size_by_cumulative_radius_rejects_more_than_thirty() -> None:
    coordinates = tuple(float(index) for index in range(31))
    labels = tuple(float(index % 2) for index in range(31))
    dosages = tuple(float(index % 3) for index in range(31))
    training_data = _make_training_data(
        coordinates,
        labels=labels,
        dosages=dosages,
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    with pytest.raises(ValueError, match=r"\[1, 30\]"):
        effect_size_by_cumulative_radius(
            training_data,
            ancestry_coordinate,
            _target_variant(),
            31,
            min_samples=1,
        )


def test_effect_size_by_cumulative_radius_returns_radius_effect_pairs() -> None:
    training_data = _make_training_data(
        (-1.0, 1.0, -3.0, 3.0),
        labels=(0.0, 1.0, 0.0, 1.0),
        dosages=(0.0, 1.0, 1.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radii_and_effects = effect_size_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        2,
        min_samples=2,
    )

    assert len(radii_and_effects) == 2
    assert radii_and_effects[0][0] < radii_and_effects[1][0]
    assert all(math.isfinite(effect_size) for _, effect_size in radii_and_effects)


def test_effect_size_standard_error_by_cumulative_radius_returns_radius_error_pairs() -> None:
    training_data = _make_training_data(
        (-1.0, 1.0, -3.0, 3.0),
        labels=(0.0, 1.0, 0.0, 1.0),
        dosages=(0.0, 1.0, 1.0, 2.0),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    radii_and_errors = effect_size_standard_error_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        2,
        min_samples=2,
    )

    assert len(radii_and_errors) == 2
    assert radii_and_errors[0][0] < radii_and_errors[1][0]
    assert all(math.isfinite(standard_error) for _, standard_error in radii_and_errors)


def test_target_ld_similarity_by_cumulative_radius_returns_one_for_matching_profiles() -> None:
    training_data = _make_multivariant_training_data(
        coordinates=(-1.0, 1.0, -3.0, 3.0),
        labels=(0.0, 1.0, 0.0, 1.0),
        dosage_rows=(
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 1.0),
        ),
    )
    ancestry_coordinate = PriorityAncestryCoordinate(values=(0.0,), dimension=1)

    similarities = target_ld_similarity_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        _target_variant(),
        2,
    )

    assert len(similarities) == 2
    assert similarities[0][1] == pytest.approx(1.0)
    assert similarities[1][1] == pytest.approx(1.0)
