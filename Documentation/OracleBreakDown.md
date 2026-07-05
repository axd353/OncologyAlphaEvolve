# Oracle Breakdown

The oracle is split into two parts.

## Part 1: Priority Function

This is the part intended to be evolved with the FunSearch-style search loop.

Input signature:

```python
priority_function(training_data, ancestry_coordinate, target_variant) -> float
```

The evolved function must keep the exact three-argument signature below. The
names matter because the evaluator validates the callable before it runs any
oracle work:

```python
def priority(training_data, ancestry_coordinate, target_variant) -> float:
        ...
```

The oracle is responsible for converting raw pickled data into strict objects
before it calls the priority function. The priority function should not import
or call [GenomicsHelpers/oracle_data_adapter.py](GenomicsHelpers/oracle_data_adapter.py).

Exact runtime argument shapes:

```python
PriorityTrainingData(
        records=tuple[PriorityTrainingRecord, ...],
        variant_names=tuple[str, ...],
        variant_dosage_fields=tuple[str, ...],
        covariate_names=tuple[str, ...],
        sample_count=int,
        ancestry_dimension=int,
        has_additional_covariates=bool,
)

PriorityTrainingRecord(
        label=float,
        ancestry_coordinate=tuple[float, ...],
        variant_dosages=dict[str, float],
        covariates=dict[str, float] | None,
)

PriorityAncestryCoordinate(
        values=tuple[float, ...],
        dimension=int,
)

PriorityTargetVariant(
        name=str,
        dosage_field=str,
        column_index=int,
)
```

Inputs:

- `training_data`: a `PriorityTrainingData` object. `records` is the Oracle-Train
    sample set normalized into strict per-subject records. `variant_names` contains
    logical variant identifiers with any `dosage__` prefix stripped. `variant_dosage_fields`
    contains the exact dosage-column names aligned to `variant_names`.
- `ancestry_coordinate`: a `PriorityAncestryCoordinate` object for the current
    target subject.
- `target_variant`: a `PriorityTargetVariant` object. `name` is the logical
    variant identifier, `dosage_field` is the exact dosage column name used by the
    oracle internals, and `column_index` is its aligned position in
    `training_data.variant_names`.

Output:

- `radius`: a finite, non-negative radius `r` that defines the closed ball around
    the ancestry point `a`

Its job is only to choose the radius.

The priority function should inspect only these normalized objects. It should
not estimate the effect size itself; it only returns the radius that the
effect-size calculator will use.

## Part 2: Effect Size Calculator

This part operationalizes the oracle once a radius has been chosen.

Input signature:

```python
effect_size_calculator(training_data, ancestry_coordinate, target_variant, radius) -> float
```

Inputs:

- `training_data`: the labelled training dataset
- `ancestry_coordinate`: the target subject ancestry vector `a`
- `target_variant`: the target variant `j`
- `radius`: the radius selected by the priority function

Output:

- `effect_size`: the estimated marginal effect size `\hat b_j(a)`

Repository location:

- [GenomicsHelpers/effect_size_calculator.py](GenomicsHelpers/effect_size_calculator.py)

## Handoff

The two parts fit together as follows:

1. The priority function receives `(training_data, ancestry_coordinate, target_variant)`.
2. It returns a radius `r`.
3. That radius is passed into `effect_size_calculator(...)`.
4. The effect size calculator uses that radius to operationalize the oracle and return `\hat b_j(a)`.

So the overall oracle behavior is:

```python
radius = priority_function(training_data, ancestry_coordinate, target_variant)
effect_size = effect_size_calculator(
    training_data,
    ancestry_coordinate,
    target_variant,
    radius,
)
```

This document only records the interface split. Further details about estimation and implementation can be documented later.