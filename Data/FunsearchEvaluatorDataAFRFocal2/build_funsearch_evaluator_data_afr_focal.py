from __future__ import annotations

"""Build AFR focal evaluator-ready datasets from prepared OncoArray pickles.

The source pickles in Data/FunsearchEvaluatorDataOncoArray are already
standardized condition-wide. This script pools the prepared train/test/heldout
pickles per condition, resamples them into AFR-focused splits with explicit
case/control quotas, and preserves the existing standardized ancestry
coordinates.

- heldout: 200 African_Ancestry cases + 200 African_Ancestry controls
- test: 200 African_Ancestry cases + 200 African_Ancestry controls +
  30 European cases + 30 European controls + 10 Asian cases + 10 Asian controls
- train: 55 African_Ancestry cases + 55 African_Ancestry controls +
    1400 European cases + 1400 European controls + 50 Asian cases + 50 Asian controls

Heldout AFR selections are shared across `no_covariates` and
`with_covariates`, while test/train selections are drawn independently per
condition and remain disjoint within each condition.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from build_funsearch_evaluator_data import configure_logger
from build_funsearch_evaluator_data import format_source_counts


DEFAULT_INPUT_DIR = DATA_DIR / "FunsearchEvaluatorDataOncoArray"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILENAME = "build_funsearch_evaluator_data.log"
DEFAULT_TRANSFORMATIONS_FILENAME = "transformations.txt"
DEFAULT_TRACKING_FILENAME = "output_row_tracking.pkl"
SOURCE_TRACKING_FILENAME = "output_row_tracking.pkl"
OUTPUT_SPLITS = ("heldout", "test", "train")

HELDOUT_AFR_CASE_COUNT = 200
HELDOUT_AFR_CONTROL_COUNT = 200

TEST_AFR_CASE_COUNT = 200
TEST_AFR_CONTROL_COUNT = 200
TEST_EUROPEAN_CASE_COUNT = 30
TEST_EUROPEAN_CONTROL_COUNT = 30
TEST_ASIAN_CASE_COUNT = 10
TEST_ASIAN_CONTROL_COUNT = 10

TRAIN_AFR_CASE_COUNT = 55
TRAIN_AFR_CONTROL_COUNT = 55
TRAIN_EUROPEAN_CASE_COUNT = 1400
TRAIN_EUROPEAN_CONTROL_COUNT = 1400
TRAIN_ASIAN_CASE_COUNT = 50
TRAIN_ASIAN_CONTROL_COUNT = 50

PHENOTYPE_CONTROL = 0
PHENOTYPE_CASE = 1
PHENOTYPE_ORDER = (PHENOTYPE_CONTROL, PHENOTYPE_CASE)

ANCESTRY_AFR = "African_Ancestry"
ANCESTRY_ASIAN = "Asian"
ANCESTRY_EUROPEAN = "European"
ANCESTRY_ORDER = (ANCESTRY_AFR, ANCESTRY_ASIAN, ANCESTRY_EUROPEAN)


@dataclass(frozen=True)
class ConditionSpec:
    name: str


@dataclass(frozen=True)
class PooledConditionData:
    frame: pd.DataFrame
    tracking: pd.DataFrame


@dataclass(frozen=True)
class ConditionKeyPlan:
    heldout_keys: set[tuple[str, int]]
    test_keys: set[tuple[str, int]]
    train_keys: set[tuple[str, int]]


CONDITIONS = (
    ConditionSpec(name="no_covariates"),
    ConditionSpec(name="with_covariates"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AFR focal evaluator-ready OncoArray datasets in "
            "Data/FunsearchEvaluatorDataAFRFocal2/."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the prepared OncoArray evaluator pickles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the AFR focal evaluator pickles are written.",
    )
    parser.add_argument(
        "--heldout-afr-case-count",
        type=int,
        default=HELDOUT_AFR_CASE_COUNT,
        help="Number of shared African_Ancestry cases in heldout.",
    )
    parser.add_argument(
        "--heldout-afr-control-count",
        type=int,
        default=HELDOUT_AFR_CONTROL_COUNT,
        help="Number of shared African_Ancestry controls in heldout.",
    )
    parser.add_argument(
        "--test-afr-case-count",
        type=int,
        default=TEST_AFR_CASE_COUNT,
        help="Number of African_Ancestry cases in test.",
    )
    parser.add_argument(
        "--test-afr-control-count",
        type=int,
        default=TEST_AFR_CONTROL_COUNT,
        help="Number of African_Ancestry controls in test.",
    )
    parser.add_argument(
        "--test-european-case-count",
        type=int,
        default=TEST_EUROPEAN_CASE_COUNT,
        help="Number of European cases in test.",
    )
    parser.add_argument(
        "--test-european-control-count",
        type=int,
        default=TEST_EUROPEAN_CONTROL_COUNT,
        help="Number of European controls in test.",
    )
    parser.add_argument(
        "--test-asian-case-count",
        type=int,
        default=TEST_ASIAN_CASE_COUNT,
        help="Number of Asian cases in test.",
    )
    parser.add_argument(
        "--test-asian-control-count",
        type=int,
        default=TEST_ASIAN_CONTROL_COUNT,
        help="Number of Asian controls in test.",
    )
    parser.add_argument(
        "--train-afr-case-count",
        type=int,
        default=TRAIN_AFR_CASE_COUNT,
        help="Number of African_Ancestry cases in train.",
    )
    parser.add_argument(
        "--train-afr-control-count",
        type=int,
        default=TRAIN_AFR_CONTROL_COUNT,
        help="Number of African_Ancestry controls in train.",
    )
    parser.add_argument(
        "--train-european-case-count",
        type=int,
        default=TRAIN_EUROPEAN_CASE_COUNT,
        help="Number of European cases in train.",
    )
    parser.add_argument(
        "--train-european-control-count",
        type=int,
        default=TRAIN_EUROPEAN_CONTROL_COUNT,
        help="Number of European controls in train.",
    )
    parser.add_argument(
        "--train-asian-case-count",
        type=int,
        default=TRAIN_ASIAN_CASE_COUNT,
        help="Number of Asian cases in train.",
    )
    parser.add_argument(
        "--train-asian-control-count",
        type=int,
        default=TRAIN_ASIAN_CONTROL_COUNT,
        help="Number of Asian controls in train.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for field_name in (
        "heldout_afr_case_count",
        "heldout_afr_control_count",
        "test_afr_case_count",
        "test_afr_control_count",
        "test_european_case_count",
        "test_european_control_count",
        "test_asian_case_count",
        "test_asian_control_count",
        "train_afr_case_count",
        "train_afr_control_count",
        "train_european_case_count",
        "train_european_control_count",
        "train_asian_case_count",
        "train_asian_control_count",
    ):
        if getattr(args, field_name) < 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be non-negative.")

    heldout_quota_counts = build_heldout_quota_counts(args)
    test_quota_counts = build_test_quota_counts(args)
    train_quota_counts = build_train_quota_counts(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / DEFAULT_LOG_FILENAME
    transformations_path = output_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    tracking_path = output_dir / DEFAULT_TRACKING_FILENAME
    logger = configure_logger(log_path)
    logger.info(
        (
            "Starting AFR focal OncoArray evaluator data build input_dir=%s "
            "output_dir=%s heldout=%s test=%s train=%s random_seed=%d"
        ),
        args.input_dir,
        output_dir,
        format_group_counts(heldout_quota_counts),
        format_group_counts(test_quota_counts),
        format_group_counts(train_quota_counts),
        args.random_seed,
    )

    seed_sequence = np.random.SeedSequence(args.random_seed)
    child_sequences = seed_sequence.spawn(1 + len(CONDITIONS))
    pooled_conditions = {
        condition.name: load_pooled_condition_data(condition, args.input_dir)
        for condition in CONDITIONS
    }
    shared_rng = np.random.default_rng(child_sequences[0])
    shared_heldout_keys = build_shared_heldout_keys(
        pooled_data=pooled_conditions[CONDITIONS[0].name],
        heldout_quota_counts=heldout_quota_counts,
        rng=shared_rng,
        logger=logger,
    )

    condition_plans = {
        condition.name: build_condition_key_plan(
            pooled_data=pooled_conditions[condition.name],
            shared_heldout_keys=shared_heldout_keys,
            test_quota_counts=test_quota_counts,
            train_quota_counts=train_quota_counts,
            rng=np.random.default_rng(child_sequence),
            context=condition.name,
        )
        for condition, child_sequence in zip(CONDITIONS, child_sequences[1:])
    }
    log_plan_overlap(
        no_cov_plan=condition_plans["no_covariates"],
        with_cov_plan=condition_plans["with_covariates"],
        logger=logger,
    )

    tracking_frames: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        condition_tracking = build_condition_datasets(
            spec=condition,
            pooled_data=pooled_conditions[condition.name],
            output_dir=output_dir,
            plan=condition_plans[condition.name],
            heldout_quota_counts=heldout_quota_counts,
            test_quota_counts=test_quota_counts,
            train_quota_counts=train_quota_counts,
            logger=logger,
        )
        tracking_frames.append(condition_tracking)

    copy_transformations_file(args.input_dir, transformations_path)
    logger.info("Copied ancestry transformations to %s", transformations_path)

    tracking_frame = pd.concat(tracking_frames, ignore_index=True)
    tracking_frame.to_pickle(tracking_path)
    logger.info("Wrote output row tracking to %s rows=%d", tracking_path, len(tracking_frame))
    logger.info("Log file path=%s", log_path)
    logger.info("Finished AFR focal OncoArray evaluator data build")


def build_heldout_quota_counts(args: argparse.Namespace) -> dict[tuple[str, int], int]:
    return {
        (ANCESTRY_AFR, PHENOTYPE_CASE): int(args.heldout_afr_case_count),
        (ANCESTRY_AFR, PHENOTYPE_CONTROL): int(args.heldout_afr_control_count),
    }


def build_test_quota_counts(args: argparse.Namespace) -> dict[tuple[str, int], int]:
    return {
        (ANCESTRY_AFR, PHENOTYPE_CASE): int(args.test_afr_case_count),
        (ANCESTRY_AFR, PHENOTYPE_CONTROL): int(args.test_afr_control_count),
        (ANCESTRY_EUROPEAN, PHENOTYPE_CASE): int(args.test_european_case_count),
        (ANCESTRY_EUROPEAN, PHENOTYPE_CONTROL): int(args.test_european_control_count),
        (ANCESTRY_ASIAN, PHENOTYPE_CASE): int(args.test_asian_case_count),
        (ANCESTRY_ASIAN, PHENOTYPE_CONTROL): int(args.test_asian_control_count),
    }


def build_train_quota_counts(args: argparse.Namespace) -> dict[tuple[str, int], int]:
    return {
        (ANCESTRY_AFR, PHENOTYPE_CASE): int(args.train_afr_case_count),
        (ANCESTRY_AFR, PHENOTYPE_CONTROL): int(args.train_afr_control_count),
        (ANCESTRY_EUROPEAN, PHENOTYPE_CASE): int(args.train_european_case_count),
        (ANCESTRY_EUROPEAN, PHENOTYPE_CONTROL): int(args.train_european_control_count),
        (ANCESTRY_ASIAN, PHENOTYPE_CASE): int(args.train_asian_case_count),
        (ANCESTRY_ASIAN, PHENOTYPE_CONTROL): int(args.train_asian_control_count),
    }


def build_shared_heldout_keys(
    *,
    pooled_data: PooledConditionData,
    heldout_quota_counts: dict[tuple[str, int], int],
    rng: np.random.Generator,
    logger: logging.Logger,
) -> set[tuple[str, int]]:
    available_by_group = build_group_key_pool(pooled_data.tracking)
    selected_groups: list[set[tuple[str, int]]] = []
    for group_key, count in heldout_quota_counts.items():
        ancestry_group, phenotype = group_key
        selected = draw_keys_from_pool(
            rng,
            available_by_group[group_key],
            count,
            f"shared heldout {ancestry_group} {phenotype_name(phenotype)}",
        )
        selected_groups.append(selected)

    selected_keys = merge_key_sets(*selected_groups)
    source_counts: dict[str, int] = {}
    for logical_source_name, _ in selected_keys:
        source_counts[logical_source_name] = source_counts.get(logical_source_name, 0) + 1
    logger.info(
        "Shared heldout key plan composition=%s sources=%s total=%d",
        format_group_counts(heldout_quota_counts),
        format_source_counts(dict(sorted(source_counts.items()))),
        len(selected_keys),
    )
    return selected_keys


def build_condition_key_plan(
    *,
    pooled_data: PooledConditionData,
    shared_heldout_keys: set[tuple[str, int]],
    test_quota_counts: dict[tuple[str, int], int],
    train_quota_counts: dict[tuple[str, int], int],
    rng: np.random.Generator,
    context: str,
) -> ConditionKeyPlan:
    available_by_group = build_group_key_pool(pooled_data.tracking)
    consume_selected_keys_from_available(
        available_by_group,
        shared_heldout_keys,
        f"{context} shared heldout",
    )
    test_keys = draw_grouped_keys(
        rng=rng,
        available_by_group=available_by_group,
        quota_counts=test_quota_counts,
        context=f"{context} test",
    )
    train_keys = draw_grouped_keys(
        rng=rng,
        available_by_group=available_by_group,
        quota_counts=train_quota_counts,
        context=f"{context} train",
    )
    return ConditionKeyPlan(
        heldout_keys=set(shared_heldout_keys),
        test_keys=test_keys,
        train_keys=train_keys,
    )


def build_group_key_pool(
    tracking: pd.DataFrame,
) -> dict[tuple[str, int], set[tuple[str, int]]]:
    return {
        (ancestry_group, phenotype): group_keys(tracking, ancestry_group, phenotype)
        for ancestry_group in ANCESTRY_ORDER
        for phenotype in PHENOTYPE_ORDER
    }


def draw_grouped_keys(
    *,
    rng: np.random.Generator,
    available_by_group: dict[tuple[str, int], set[tuple[str, int]]],
    quota_counts: dict[tuple[str, int], int],
    context: str,
) -> set[tuple[str, int]]:
    selected_groups: list[set[tuple[str, int]]] = []
    for group_key, count in quota_counts.items():
        ancestry_group, phenotype = group_key
        selected = draw_keys_from_pool(
            rng,
            available_by_group[group_key],
            count,
            f"{context} {ancestry_group} {phenotype_name(phenotype)}",
        )
        consume_key_set(
            available_by_group[group_key],
            selected,
            f"{context} {ancestry_group} {phenotype_name(phenotype)}",
        )
        selected_groups.append(selected)
    return merge_key_sets(*selected_groups)


def consume_selected_keys_from_available(
    available_by_group: dict[tuple[str, int], set[tuple[str, int]]],
    selected_keys: set[tuple[str, int]],
    context: str,
) -> None:
    consumed_total = 0
    for group_key, available_keys in available_by_group.items():
        selected_for_group = available_keys & selected_keys
        if not selected_for_group:
            continue
        consume_key_set(
            available_keys,
            selected_for_group,
            f"{context} {group_key[0]} {phenotype_name(group_key[1])}",
        )
        consumed_total += len(selected_for_group)
    if consumed_total != len(selected_keys):
        raise ValueError(
            f"Selection mismatch for {context}: consumed {consumed_total} keys, expected {len(selected_keys)}."
        )


def build_condition_datasets(
    *,
    spec: ConditionSpec,
    pooled_data: PooledConditionData,
    output_dir: Path,
    plan: ConditionKeyPlan,
    heldout_quota_counts: dict[tuple[str, int], int],
    test_quota_counts: dict[tuple[str, int], int],
    train_quota_counts: dict[tuple[str, int], int],
    logger: logging.Logger,
) -> pd.DataFrame:
    heldout_positions = positions_for_keys(pooled_data.tracking, plan.heldout_keys)
    test_positions = positions_for_keys(pooled_data.tracking, plan.test_keys)
    train_positions = positions_for_keys(pooled_data.tracking, plan.train_keys)

    heldout_output_name = f"{spec.name}_heldout.pkl"
    heldout_dataset, heldout_source_counts, heldout_tracking = assemble_output(
        pooled_data,
        heldout_positions,
        output_name=heldout_output_name,
    )

    test_output_name = f"{spec.name}_test.pkl"
    test_dataset, test_source_counts, test_tracking = assemble_output(
        pooled_data,
        test_positions,
        output_name=test_output_name,
    )

    train_output_name = f"{spec.name}_train.pkl"
    train_dataset, train_source_counts, train_tracking = assemble_output(
        pooled_data,
        train_positions,
        output_name=train_output_name,
    )

    validate_positions_count(heldout_positions, sum(heldout_quota_counts.values()), f"{spec.name} heldout")
    validate_positions_count(test_positions, sum(test_quota_counts.values()), f"{spec.name} test")
    validate_positions_count(train_positions, sum(train_quota_counts.values()), f"{spec.name} train")
    validate_dataset_size(len(heldout_dataset), sum(heldout_quota_counts.values()), heldout_output_name)
    validate_dataset_size(len(test_dataset), sum(test_quota_counts.values()), test_output_name)
    validate_dataset_size(len(train_dataset), sum(train_quota_counts.values()), train_output_name)
    validate_output_group_counts(heldout_dataset, heldout_tracking, heldout_quota_counts, heldout_output_name)
    validate_output_group_counts(test_dataset, test_tracking, test_quota_counts, test_output_name)
    validate_output_group_counts(train_dataset, train_tracking, train_quota_counts, train_output_name)

    condition_outputs = {
        heldout_output_name: (heldout_dataset, heldout_source_counts, heldout_tracking),
        test_output_name: (test_dataset, test_source_counts, test_tracking),
        train_output_name: (train_dataset, train_source_counts, train_tracking),
    }
    for output_name, (dataset, source_counts, tracking) in condition_outputs.items():
        output_path = output_dir / output_name
        dataset.to_pickle(output_path)
        logger.info(
            "Wrote %s rows=%d sources=%s composition=%s",
            output_path,
            len(dataset),
            format_source_counts(source_counts),
            format_group_counts(compute_output_group_counts(dataset, tracking)),
        )

    logger.info(
        "Condition %s reused standardized ancestry coordinates from prepared OncoArray pickles",
        spec.name,
    )

    condition_tracking = pd.concat(
        [heldout_tracking, test_tracking, train_tracking],
        ignore_index=True,
    )
    return condition_tracking


def load_pooled_condition_data(spec: ConditionSpec, input_dir: Path) -> PooledConditionData:
    tracking_path = input_dir / SOURCE_TRACKING_FILENAME
    if not tracking_path.exists():
        raise FileNotFoundError(f"Missing source tracking file: {tracking_path}")

    source_tracking = pd.read_pickle(tracking_path)
    frames: list[pd.DataFrame] = []
    tracking_parts: list[pd.DataFrame] = []
    for split_name in OUTPUT_SPLITS:
        output_pickle_name = f"{spec.name}_{split_name}.pkl"
        dataset_path = input_dir / output_pickle_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing prepared dataset pickle: {dataset_path}")
        dataset = pd.read_pickle(dataset_path).reset_index(drop=True)
        if "phenotype" not in dataset.columns:
            raise KeyError(f"Missing phenotype column in prepared dataset: {dataset_path}")
        split_tracking = source_tracking.loc[
            source_tracking["output_pickle_name"] == output_pickle_name
        ].reset_index(drop=True)
        if len(dataset) != len(split_tracking):
            raise ValueError(
                f"Tracking length mismatch for {output_pickle_name}: "
                f"dataset has {len(dataset)} rows, tracking has {len(split_tracking)} rows."
            )
        frames.append(dataset)
        tracking_parts.append(split_tracking)

    pooled_frame = pd.concat(frames, ignore_index=True)
    phenotypes = pooled_frame["phenotype"].astype(int)
    unexpected_phenotypes = sorted(set(phenotypes.unique()) - {PHENOTYPE_CONTROL, PHENOTYPE_CASE})
    if unexpected_phenotypes:
        raise ValueError(
            f"Prepared dataset contains unsupported phenotype values: {unexpected_phenotypes}."
        )

    pooled_tracking = pd.concat(tracking_parts, ignore_index=True)
    pooled_tracking = pooled_tracking.assign(
        logical_source_name=pooled_tracking["source_pickle_name"].map(logical_source_key),
        ancestry_group=pooled_tracking["source_pickle_name"].map(source_name_to_ancestry_group),
        phenotype=phenotypes.to_numpy(dtype=int, copy=False),
    )
    return PooledConditionData(frame=pooled_frame, tracking=pooled_tracking)


def logical_source_key(source_name: str) -> str:
    stem = Path(source_name).stem
    if stem.endswith("_add_covs"):
        stem = stem[: -len("_add_covs")]
    return stem


def draw_keys_from_pool(
    rng: np.random.Generator,
    available_keys: set[tuple[str, int]],
    count: int,
    context: str,
) -> set[tuple[str, int]]:
    if count < 0:
        raise ValueError(f"Sample count must be non-negative for {context}.")
    available = np.array(sorted(available_keys), dtype=object)
    if count > len(available):
        raise ValueError(
            f"Requested {count} samples for {context}, but only {len(available)} are available."
        )
    if count == 0:
        return set()
    selected = rng.choice(available, size=count, replace=False)
    return {tuple(item) for item in np.asarray(selected, dtype=object).tolist()}


def group_keys(
    tracking: pd.DataFrame,
    ancestry_group: str,
    phenotype: int,
) -> set[tuple[str, int]]:
    group_tracking = tracking.loc[
        (tracking["ancestry_group"] == ancestry_group) & (tracking["phenotype"] == phenotype),
        ["logical_source_name", "source_row_number"],
    ]
    return {
        (str(row.logical_source_name), int(row.source_row_number))
        for row in group_tracking.itertuples(index=False)
    }


def positions_for_keys(
    tracking: pd.DataFrame,
    selected_keys: set[tuple[str, int]],
) -> np.ndarray:
    key_to_position = {
        (str(row.logical_source_name), int(row.source_row_number)): int(position)
        for position, row in enumerate(
            tracking.loc[:, ["logical_source_name", "source_row_number"]].itertuples(index=False)
        )
    }
    missing_keys = selected_keys - set(key_to_position)
    if missing_keys:
        first_missing_key = sorted(missing_keys)[0]
        raise ValueError(
            f"Missing selected key in condition {first_missing_key[0]} row {first_missing_key[1]}."
        )
    return np.array(sorted(key_to_position[key] for key in selected_keys), dtype=int)


def consume_key_set(
    available_keys: set[tuple[str, int]],
    selected_keys: set[tuple[str, int]],
    context: str,
) -> None:
    if not selected_keys.issubset(available_keys):
        raise ValueError(f"Overlapping selections detected for {context}.")
    available_keys.difference_update(selected_keys)


def merge_key_sets(*key_sets: set[tuple[str, int]]) -> set[tuple[str, int]]:
    merged: set[tuple[str, int]] = set()
    for key_set in key_sets:
        if merged & key_set:
            raise ValueError("Merged key sets contain duplicate selections.")
        merged.update(key_set)
    return merged


def log_plan_overlap(
    *,
    no_cov_plan: ConditionKeyPlan,
    with_cov_plan: ConditionKeyPlan,
    logger: logging.Logger,
) -> None:
    no_cov_discovery = no_cov_plan.test_keys | no_cov_plan.train_keys
    with_cov_discovery = with_cov_plan.test_keys | with_cov_plan.train_keys
    logger.info(
        (
            "Cross-condition key overlap heldout=%d discovery=%d "
            "same_split_test=%d same_split_train=%d nc_train_vs_wc_test=%d nc_test_vs_wc_train=%d"
        ),
        len(no_cov_plan.heldout_keys & with_cov_plan.heldout_keys),
        len(no_cov_discovery & with_cov_discovery),
        len(no_cov_plan.test_keys & with_cov_plan.test_keys),
        len(no_cov_plan.train_keys & with_cov_plan.train_keys),
        len(no_cov_plan.train_keys & with_cov_plan.test_keys),
        len(no_cov_plan.test_keys & with_cov_plan.train_keys),
    )


def validate_positions_count(positions: np.ndarray, expected_total: int, context: str) -> None:
    actual_total = int(len(np.asarray(positions, dtype=int)))
    if actual_total != expected_total:
        raise ValueError(
            f"Selection size mismatch for {context}: expected {expected_total}, got {actual_total}."
        )


def validate_dataset_size(actual_size: int, expected_size: int, output_name: str) -> None:
    if actual_size != expected_size:
        raise ValueError(
            f"Dataset size mismatch for {output_name}: expected {expected_size}, got {actual_size}."
        )


def validate_output_group_counts(
    dataset: pd.DataFrame,
    tracking: pd.DataFrame,
    expected_counts: dict[tuple[str, int], int],
    output_name: str,
) -> None:
    actual_counts = compute_output_group_counts(dataset, tracking)
    if actual_counts != expected_counts:
        raise ValueError(
            f"Composition mismatch for {output_name}: expected {format_group_counts(expected_counts)}, "
            f"got {format_group_counts(actual_counts)}."
        )


def assemble_output(
    pooled_data: PooledConditionData,
    positions: np.ndarray,
    *,
    output_name: str,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    positions = np.asarray(positions, dtype=int)
    dataset = pooled_data.frame.iloc[positions].reset_index(drop=True).copy()
    tracking = pooled_data.tracking.iloc[positions].reset_index(drop=True).copy()
    tracking = tracking.assign(
        output_pickle_name=output_name,
        output_row_number=np.arange(len(tracking), dtype=int),
    )
    tracking = tracking.loc[
        :,
        [
            "output_pickle_name",
            "output_row_number",
            "source_pickle_name",
            "source_pickle_path",
            "source_row_number",
        ],
    ]
    source_counts = tracking["source_pickle_name"].value_counts(sort=False).to_dict()
    return dataset, {str(key): int(value) for key, value in source_counts.items()}, tracking


def compute_output_group_counts(
    dataset: pd.DataFrame,
    tracking: pd.DataFrame,
) -> dict[tuple[str, int], int]:
    if len(dataset) != len(tracking):
        raise ValueError("Dataset/tracking length mismatch while computing output composition.")
    source_names = tracking["source_pickle_name"].tolist()
    phenotypes = dataset["phenotype"].astype(int).tolist()
    group_counts: dict[tuple[str, int], int] = {}
    for source_name, phenotype in zip(source_names, phenotypes):
        group_key = (source_name_to_ancestry_group(str(source_name)), int(phenotype))
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
    return {group_key: count for group_key, count in group_counts.items() if count > 0}


def source_name_to_ancestry_group(source_name: str) -> str:
    if ANCESTRY_AFR in source_name:
        return ANCESTRY_AFR
    if ANCESTRY_ASIAN in source_name:
        return ANCESTRY_ASIAN
    if ANCESTRY_EUROPEAN in source_name:
        return ANCESTRY_EUROPEAN
    raise ValueError(f"Could not infer ancestry group from source name: {source_name}")


def phenotype_name(phenotype: int) -> str:
    if phenotype == PHENOTYPE_CASE:
        return "cases"
    if phenotype == PHENOTYPE_CONTROL:
        return "controls"
    raise ValueError(f"Unsupported phenotype value: {phenotype}")


def format_group_counts(group_counts: dict[tuple[str, int], int]) -> str:
    pieces: list[str] = []
    for ancestry_group in ANCESTRY_ORDER:
        controls = int(group_counts.get((ancestry_group, PHENOTYPE_CONTROL), 0))
        cases = int(group_counts.get((ancestry_group, PHENOTYPE_CASE), 0))
        if controls == 0 and cases == 0:
            continue
        pieces.append(f"{ancestry_group}:controls={controls},cases={cases}")
    return "; ".join(pieces) if pieces else "none"


def copy_transformations_file(input_dir: Path, output_path: Path) -> None:
    source_path = input_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source transformations file: {source_path}")
    output_path.write_text(source_path.read_text())


if __name__ == "__main__":
    main()