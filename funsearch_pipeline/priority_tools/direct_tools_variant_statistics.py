# direct_tools
from __future__ import annotations

from .contracts import PriorityAncestryCoordinate
from .contracts import PriorityTargetVariant
from .contracts import PriorityTrainingData
from .helper_tools_variant_statistics import _ancestry_novelty_score
from .helper_tools_variant_statistics import _cumulative_radius_records
from .helper_tools_variant_statistics import _dosage_entropy_for_records
from .helper_tools_variant_statistics import _effect_size_and_standard_error_for_records
from .helper_tools_variant_statistics import _equal_count_interval_records
from .helper_tools_variant_statistics import _label_entropy_for_records
from .helper_tools_variant_statistics import _ld_profile_similarity_for_records
from .helper_tools_variant_statistics import _minimum_radius_for_training_percentage
from .helper_tools_variant_statistics import _minimum_radius_for_sample_count
from .helper_tools_variant_statistics import _standardized_effect_change_for_interval_records


def ancestry_novelty_score(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
) -> float:
    """Return how novel the target ancestry point is relative to training support.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.

    Output:
        A non-negative normalized support score. First, the method measures the
        radius needed to enclose 10 percent of the training set around the
        target ancestry point `a = ancestry_coordinate.values`, with floor
        rounding and a minimum of one enclosed sample. It then measures the
        same 10-percent radius around up to 100 deterministically sampled
        training subjects, excluding each sampled subject itself from its own
        baseline radius calculation. The returned value is

            target_radius / median(baseline_radii)

        so a value near `1.0` means the target lies in a region with typical
        ancestry support, values above `1.0` indicate sparser-than-typical
        local support, and values below `1.0` indicate denser-than-typical
        support.

    Raises:
        ValueError: if the dataset is empty or the contract shapes are
            inconsistent.
    """

    return _ancestry_novelty_score(training_data, ancestry_coordinate)


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


def label_entropy_by_cumulative_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    n: int,
) -> list[tuple[float, float]]:
    """Return label entropy as the ancestry ball grows outward.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        n: Number of cumulative radii. Must be an integer in `[1, 30]`. The
            radii are the cumulative upper bounds of the equal-count intervals.

    Output:
        Let the equal-count interval boundaries be

            [0, r_1), [r_1, r_2), ..., [r_{n-1}, r_n)

        as returned by `equal_count_intervals(training_data, ancestry_coordinate, n)`.
        This method then forms the `n` cumulative closed balls

            ||x - a|| <= r_1, ||x - a|| <= r_2, ..., ||x - a|| <= r_n

        around the target ancestry point `a = ancestry_coordinate.values`.
        It returns the corresponding `n` `(radius, label_entropy)` pairs with
        strictly increasing radii. Each label entropy is the Shannon entropy of
        the phenotype labels inside that cumulative closed ball. For a binary
        label, values lie in `[0.0, 1.0]`, where `0.0` means all labels in the
        ball are the same and `1.0` means the labels are evenly balanced.

    Raises:
        TypeError: if `n` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, or `n` falls outside `[1, 30]`.
    """

    return [
        (radius, _label_entropy_for_records(records))
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


def standardized_effect_change_by_interval(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
    min_samples: int = 25,
) -> list[float]:
    """Return uncertainty-adjusted effect changes between adjacent intervals.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of equal-count ancestry-distance intervals. Must be an
            integer in `[1, 30]`.
        min_samples: Minimum number of records required in one interval before
            a marginal effect is attempted. Must be a positive integer.

    Output:
        List of `n - 1` standardized adjacent changes. Let `effect_i` and
        `standard_error_i` be the effect-size estimate and its standard error
        in interval `i` from the corresponding interval-based helper tools. For
        each adjacent pair of intervals, this method returns

            abs(effect_i - effect_{i+1})
            / sqrt(standard_error_i^2 + standard_error_{i+1}^2)

        in order from the closest interval pair to the farthest. Larger values
        indicate stronger evidence that the effect differs between neighboring
        ancestry regions. If either adjacent interval is not identifiable, the
        returned standardized change for that pair is `0.0`.

    Raises:
        TypeError: if `n` or `min_samples` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, `n` falls outside `[1, 30]`, or `min_samples` is not
            positive.
    """

    return _standardized_effect_change_for_interval_records(
        _equal_count_interval_records(
            training_data,
            ancestry_coordinate,
            n,
        ),
        target_variant,
        min_samples=min_samples,
    )


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


def target_ld_similarity_by_cumulative_radius(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    n: int,
) -> list[tuple[float, float]]:
    """Return local-vs-global LD-profile similarity as the ancestry ball grows outward.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        n: Number of cumulative radii. Must be an integer in `[1, 30]`.

    Output:
        Let the equal-count interval boundaries be

            [0, r_1), [r_1, r_2), ..., [r_{n-1}, r_n)

        as returned by `equal_count_intervals(training_data, ancestry_coordinate, n)`.
        This method then forms the `n` cumulative closed balls

            ||x - a|| <= r_1, ||x - a|| <= r_2, ..., ||x - a|| <= r_n

        around the target ancestry point `a = ancestry_coordinate.values`.
        It returns the corresponding `n` `(radius, similarity)` pairs with
        strictly increasing radii. For each cumulative ball, the method
        compares the target variant's empirical dosage-correlation profile with
        all other available variants inside that ball against the same profile
        computed on the full training data. The returned similarity lies in
        `[0.0, 1.0]`, where `1.0` means the local and global empirical LD
        profiles are identical and smaller values mean the target variant tags
        the rest of the variant set differently in the local ancestry region.

    Raises:
        TypeError: if `n` is not an integer.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, or `n` falls outside `[1, 30]`.
    """

    return [
        (
            radius,
            _ld_profile_similarity_for_records(
                records,
                training_data.records,
                training_data,
                target_variant,
            ),
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


def minimum_radius_for_training_percentage(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    percentage: float,
) -> tuple[float, float]:
    """Return the smallest radius enclosing a requested training-data percentage.

    Input:
        training_data: The exact same `PriorityTrainingData` object that the
            priority function receives as its `training_data` argument.
        ancestry_coordinate: The exact same `PriorityAncestryCoordinate` object
            that the priority function receives as its `ancestry_coordinate`
            argument.
        target_variant: The exact same `PriorityTargetVariant` object that the
            priority function receives as its `target_variant` argument.
        percentage: Requested percentage of total training samples to cover.
            The value is in `[0.0, 100.0]`, so `percentage = 25.0` means 25
            percent. If `percentage` does not map to an integer sample count,
            the target count is rounded down. A sample is usable here when its
            target dosage is finite.

    Output:
        Pair `(radius, effective_percentage)`. `radius` is the smallest closed
        ancestry-ball radius whose interior covers at least the usable sample
        count implied by `percentage` after floor rounding.
        `effective_percentage` is the actual enclosed usable-sample percentage
        measured against `training_data.sample_count`, after rounding and after
        clamping to the usable target-dosage total. If no usable target dosage
        exists at all, the method returns `(0.0, 0.0)` instead of raising.

    Raises:
        TypeError: if `percentage` is not a real number.
        ValueError: if the dataset is empty, the contract shapes are
            inconsistent, or `percentage` is out of range.
    """

    return _minimum_radius_for_training_percentage(
        training_data,
        ancestry_coordinate,
        target_variant,
        percentage,
    )


__all__ = [
    "ancestry_novelty_score",
    "dosage_entropy_by_cumulative_radius",
    "dosage_entropy_by_interval",
    "effect_size_by_cumulative_radius",
    "effect_size_by_interval",
    "effect_size_standard_error_by_cumulative_radius",
    "effect_size_standard_error_by_interval",
    "label_entropy_by_cumulative_radius",
    "minimum_radius_for_training_percentage",
    "minimum_radius_for_sample_count",
    "standardized_effect_change_by_interval",
    "target_ld_similarity_by_cumulative_radius",
]