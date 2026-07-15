# direct_tools
from __future__ import annotations

from .contracts import PriorityAncestryCoordinate
from .contracts import PriorityTargetVariant
from .contracts import PriorityTrainingData
from .helper_tools_variant_statistics import _cumulative_radius_records
from .helper_tools_variant_statistics import _dosage_entropy_for_records
from .helper_tools_variant_statistics import _effect_size_and_standard_error_for_records
from .helper_tools_variant_statistics import _equal_count_interval_records
from .helper_tools_variant_statistics import _minimum_radius_for_sample_count


def dosage_entropy_by_interval(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
) -> list[float]:
    """Return target-dosage entropy in each equal-count ancestry interval.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of equal-count ancestry-distance intervals. Must be an integer
            in `[1, 30]`. The interval boundaries are the same as those returned
            by `equal_count_intervals(training_data, ancestry_coordinate, n)`.

    Output:
        List of `n` Shannon entropies, ordered from the closest ancestry region
        to the farthest and aligned one-to-one with
        `equal_count_intervals(training_data, ancestry_coordinate, n)`. Each
        value is computed from a three-bin target-dosage histogram centered on
        the genotype-dosage support `{0, 1, 2}`. Higher entropy means the region
        uses more of the dosage support and is less concentrated on a single
        dosage state.

    Raises:
        TypeError: if `n` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, or `n` falls outside `[1, 30]`.
    """

    return [
        _dosage_entropy_for_records(records, target_variant)
        for _, records in _equal_count_interval_records(
            training_data,
            ancestry_coordinate,
            n,
        )
    ]


def dosage_entropy_by_cumulative_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
) -> list[tuple[float, float]]:
    """Return target-dosage entropy as the ancestry ball grows outward.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of cumulative radii. Must be an integer in `[1, 30]`. The
            radii are the cumulative upper bounds of the equal-count intervals.

    Output:
        Let the equal-count interval boundaries be

            [0, r_1), [r_1, r_2), ..., [r_{n-1}, r_n)

        as returned by `equal_count_intervals(training_data, ancestry_coordinate, n)`.
        This method then forms the `n` cumulative closed balls

            ||x - a|| <= r_1, ||x - a|| <= r_2, ..., ||x - a|| <= r_n

        around the target ancestry point `a = ancestry_coordinate.values`.
        It returns the corresponding `n` `(radius, entropy)` pairs with
        strictly increasing radii. Each entropy is computed from all records in
        its cumulative closed ball, using the same three-bin target-dosage
        histogram as `dosage_entropy_by_interval(...)`.

    Raises:
        TypeError: if `n` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, or `n` falls outside `[1, 30]`.
    """

    return [
        (radius, _dosage_entropy_for_records(records, target_variant))
        for radius, records in _cumulative_radius_records(
            training_data,
            ancestry_coordinate,
            n,
        )
    ]


def effect_size_by_interval(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
    min_samples: int = 25,
) -> list[float]:
    """Return marginal effect sizes in equal-count ancestry intervals.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of equal-count ancestry-distance intervals. Must be an integer
            in `[1, 30]`.
        min_samples: Minimum number of records required in one interval before a
            marginal effect is attempted. Must be a positive integer.

    Output:
        List of `n` marginal effect-size estimates, ordered from the closest
        ancestry region to the farthest and aligned one-to-one with
        `equal_count_intervals(training_data, ancestry_coordinate, n)`. Each
        effect is the single-variant logistic-regression coefficient on the
        target dosage using only that interval's records and no covariates.
        Concretely, for one interval the method fits

            logit(P(label = 1)) = intercept + effect_size * target_dosage

        on the records in that interval, and returns the fitted
        `effect_size` coefficient. If an interval has no data, too few records,
        only one label class, or no dosage variation, the returned value for
        that interval is `0.0`.

    Raises:
        TypeError: if `n` or `min_samples` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, `n` falls outside `[1, 30]`, or `min_samples` is not
            positive.
    """

    return [
        _effect_size_and_standard_error_for_records(
            records,
            target_variant,
            min_samples=min_samples,
        )[0]
        for _, records in _equal_count_interval_records(
            training_data,
            ancestry_coordinate,
            n,
        )
    ]


def effect_size_standard_error_by_interval(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
    min_samples: int = 25,
) -> list[float]:
    """Return marginal-effect standard errors in equal-count ancestry intervals.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of equal-count ancestry-distance intervals. Must be an integer
            in `[1, 30]`.
        min_samples: Minimum number of records required in one interval before a
            marginal effect is attempted. Must be a positive integer.

    Output:
        List of `n` standard errors, ordered from the closest ancestry region to
        the farthest and aligned one-to-one with
        `equal_count_intervals(training_data, ancestry_coordinate, n)`. The
        underlying point estimate in each interval comes from fitting

            logit(P(label = 1)) = intercept + effect_size * target_dosage

        on that interval's records with no covariates. The reported standard
        error is the standard error of the fitted `effect_size` coefficient
        from that same logistic model. If an interval has no data, too few
        records, only one label class, or no dosage variation, the returned
        standard error for that interval is `math.inf`.

    Raises:
        TypeError: if `n` or `min_samples` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, `n` falls outside `[1, 30]`, or `min_samples` is not
            positive.
    """

    return [
        _effect_size_and_standard_error_for_records(
            records,
            target_variant,
            min_samples=min_samples,
        )[1]
        for _, records in _equal_count_interval_records(
            training_data,
            ancestry_coordinate,
            n,
        )
    ]


def effect_size_by_cumulative_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
    min_samples: int = 25,
) -> list[tuple[float, float]]:
    """Return marginal effect sizes as the ancestry ball grows outward.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of cumulative radii. Must be an integer in `[1, 30]`.
        min_samples: Minimum number of records required inside one cumulative
            ball before a marginal effect is attempted. Must be a positive
            integer.

    Output:
        Let the equal-count interval boundaries be

            [0, r_1), [r_1, r_2), ..., [r_{n-1}, r_n)

        as returned by `equal_count_intervals(training_data, ancestry_coordinate, n)`.
        This method then forms the `n` cumulative closed balls

            ||x - a|| <= r_1, ||x - a|| <= r_2, ..., ||x - a|| <= r_n

        around the target ancestry point `a = ancestry_coordinate.values`.
        It returns the corresponding `n` `(radius, effect_size)` pairs with
        strictly increasing radii. Each effect is the single-variant
        logistic-regression coefficient on target dosage using all records
        inside the corresponding cumulative closed ball and no covariates.
        Concretely, for one cumulative radius the method fits

            logit(P(label = 1)) = intercept + effect_size * target_dosage

        on all records whose ancestry distance is at most that radius, and
        returns the fitted `effect_size` coefficient. If a cumulative ball has
        no data, too few records, only one label class, or no dosage variation,
        the returned effect at that radius is `0.0`.

    Raises:
        TypeError: if `n` or `min_samples` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, `n` falls outside `[1, 30]`, or `min_samples` is not
            positive.
    """

    return [
        (
            radius,
            _effect_size_and_standard_error_for_records(
                records,
                target_variant,
                min_samples=min_samples,
            )[0],
        )
        for radius, records in _cumulative_radius_records(
            training_data,
            ancestry_coordinate,
            n,
        )
    ]


def effect_size_standard_error_by_cumulative_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
    min_samples: int = 25,
) -> list[tuple[float, float]]:
    """Return marginal-effect standard errors as the ancestry ball grows outward.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of cumulative radii. Must be an integer in `[1, 30]`.
        min_samples: Minimum number of records required inside one cumulative
            ball before a marginal effect is attempted. Must be a positive
            integer.

    Output:
        Let the equal-count interval boundaries be

            [0, r_1), [r_1, r_2), ..., [r_{n-1}, r_n)

        as returned by `equal_count_intervals(training_data, ancestry_coordinate, n)`.
        This method then forms the `n` cumulative closed balls

            ||x - a|| <= r_1, ||x - a|| <= r_2, ..., ||x - a|| <= r_n

        around the target ancestry point `a = ancestry_coordinate.values`.
        It returns the corresponding `n` `(radius, standard_error)` pairs with
        strictly increasing radii. At each radius the method fits

            logit(P(label = 1)) = intercept + effect_size * target_dosage

        on all records inside that cumulative radius with no covariates. The
        reported standard error is the standard error of the fitted
        `effect_size` coefficient from that same logistic model. If a cumulative
        ball has no data, too few records, only one label class, or no dosage
        variation, the returned standard error at that radius is `math.inf`.

    Raises:
        TypeError: if `n` or `min_samples` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, `n` falls outside `[1, 30]`, or `min_samples` is not
            positive.
    """

    return [
        (
            radius,
            _effect_size_and_standard_error_for_records(
                records,
                target_variant,
                min_samples=min_samples,
            )[1],
        )
        for radius, records in _cumulative_radius_records(
            training_data,
            ancestry_coordinate,
            n,
        )
    ]


def minimum_radius_for_sample_count(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    min_samples: int,
) -> tuple[float, int]:
    """Return the smallest radius enclosing a usable target-dosage sample count.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        min_samples: Requested minimum number of usable samples. Must be a
            positive integer. A sample is usable here when its target dosage is
            finite.

    Output:
        Pair `(radius, effective_min_samples)`. `radius` is the smallest closed
        ancestry-ball radius whose interior covers at least
        `effective_min_samples` usable samples for the target variant.
        `effective_min_samples` equals the requested `min_samples` unless the
        dataset has fewer usable target dosages, in which case it is clamped to
        that usable total and the returned radius encloses all usable samples.
        If no usable target dosage exists at all, the method returns `(0.0, 0)`
        instead of raising.

    Raises:
        TypeError: if `min_samples` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, or `min_samples` is not positive.
    """

    return _minimum_radius_for_sample_count(
        training_data,
        ancestry_coordinate,
        target_variant,
        min_samples,
    )


__all__ = [
    "dosage_entropy_by_cumulative_radius",
    "dosage_entropy_by_interval",
    "effect_size_by_cumulative_radius",
    "effect_size_by_interval",
    "effect_size_standard_error_by_cumulative_radius",
    "effect_size_standard_error_by_interval",
    "minimum_radius_for_sample_count",
]