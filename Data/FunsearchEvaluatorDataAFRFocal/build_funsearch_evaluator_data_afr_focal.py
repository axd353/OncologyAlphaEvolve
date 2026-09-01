from __future__ import annotations

"""Build AFR focal evaluator-ready datasets from prepared OncoArray pickles.

The source pickles in Data/FunsearchEvaluatorDataOncoArray are already
standardized condition-wide. This script pools the prepared train/test/heldout
pickles per condition, resamples them into AFR-focused discovery splits, and
preserves the existing standardized ancestry coordinates.

- heldout: 500 African_Ancestry subjects
- test: 375 African_Ancestry subjects + 165 random non-African_Ancestry subjects
- train: 400 African_Ancestry subjects + 200 Asian subjects + 2400 European subjects

Heldout AFR selections are shared across `no_covariates` and
`with_covariates`, while train/test selections are drawn independently per
condition.
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

HELDOUT_AFR_COUNT = 500
TEST_AFR_COUNT = 375
TEST_RANDOM_COUNT = 165
TRAIN_AFR_COUNT = 400
TRAIN_ASIAN_COUNT = 200
TRAIN_RANDOM_COUNT = 2400

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
            "Data/FunsearchEvaluatorDataAFRFocal/."
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
        "--heldout-afr-count",
        type=int,
        default=HELDOUT_AFR_COUNT,
        help="Number of shared AFR subjects in heldout.",
    )
    parser.add_argument(
        "--test-afr-count",
        type=int,
        default=TEST_AFR_COUNT,
        help="Number of AFR subjects in test.",
    )
    parser.add_argument(
        "--test-random-count",
        type=int,
        default=TEST_RANDOM_COUNT,
        help="Number of random non-AFR subjects in test.",
    )
    parser.add_argument(
        "--train-afr-count",
        type=int,
        default=TRAIN_AFR_COUNT,
        help="Number of AFR subjects in train.",
    )
    parser.add_argument(
        "--train-asian-count",
        type=int,
        default=TRAIN_ASIAN_COUNT,
        help="Number of Asian subjects in train.",
    )
    parser.add_argument(
        "--train-random-count",
        type=int,
        default=TRAIN_RANDOM_COUNT,
        help="Number of random European subjects in train.",
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
        "heldout_afr_count",
        "test_afr_count",
        "test_random_count",
        "train_afr_count",
        "train_asian_count",
        "train_random_count",
    ):
        if getattr(args, field_name) < 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be non-negative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / DEFAULT_LOG_FILENAME
    transformations_path = output_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    tracking_path = output_dir / DEFAULT_TRACKING_FILENAME
    logger = configure_logger(log_path)
    logger.info(
        (
            "Starting AFR focal OncoArray evaluator data build input_dir=%s "
            "output_dir=%s heldout_afr=%d test_afr=%d test_random=%d "
            "train_afr=%d train_asian=%d train_random=%d random_seed=%d"
        ),
        args.input_dir,
        output_dir,
        args.heldout_afr_count,
        args.test_afr_count,
        args.test_random_count,
        args.train_afr_count,
        args.train_asian_count,
        args.train_random_count,
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
        heldout_afr_count=int(args.heldout_afr_count),
        rng=shared_rng,
        logger=logger,
    )

    condition_plans = build_condition_key_plans(
        pooled_data=pooled_conditions[CONDITIONS[0].name],
        shared_heldout_keys=shared_heldout_keys,
        test_afr_count=int(args.test_afr_count),
        test_random_count=int(args.test_random_count),
        train_afr_count=int(args.train_afr_count),
        train_asian_count=int(args.train_asian_count),
        train_random_count=int(args.train_random_count),
        no_covariates_rng=np.random.default_rng(child_sequences[1]),
        with_covariates_rng=np.random.default_rng(child_sequences[2]),
        logger=logger,
    )

    tracking_frames: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        condition_tracking = build_condition_datasets(
            spec=condition,
            pooled_data=pooled_conditions[condition.name],
            output_dir=output_dir,
            plan=condition_plans[condition.name],
            heldout_afr_count=int(args.heldout_afr_count),
            test_afr_count=int(args.test_afr_count),
            test_random_count=int(args.test_random_count),
            train_afr_count=int(args.train_afr_count),
            train_asian_count=int(args.train_asian_count),
            train_random_count=int(args.train_random_count),
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


def build_shared_heldout_keys(
    *,
    pooled_data: PooledConditionData,
    heldout_afr_count: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> set[tuple[str, int]]:
    afr_tracking = pooled_data.tracking.loc[
        pooled_data.tracking["ancestry_group"] == ANCESTRY_AFR,
        ["logical_source_name", "source_row_number"],
    ]
    available_keys = [
        (str(row.logical_source_name), int(row.source_row_number))
        for row in afr_tracking.itertuples(index=False)
    ]
    if heldout_afr_count > len(available_keys):
        raise ValueError(
            f"Requested {heldout_afr_count} shared heldout AFR rows, but only {len(available_keys)} are available."
        )
    chosen_indices = rng.choice(len(available_keys), size=heldout_afr_count, replace=False)
    selected_keys = {available_keys[int(index)] for index in np.asarray(chosen_indices, dtype=int)}
    source_counts: dict[str, int] = {}
    for logical_source_name, _ in selected_keys:
        source_counts[logical_source_name] = source_counts.get(logical_source_name, 0) + 1
    logger.info(
        "Shared heldout AFR row plan counts=%s total=%d",
        format_source_counts(dict(sorted(source_counts.items()))),
        len(selected_keys),
    )
    return selected_keys


def build_condition_datasets(
    *,
    spec: ConditionSpec,
    pooled_data: PooledConditionData,
    output_dir: Path,
    plan: ConditionKeyPlan,
    heldout_afr_count: int,
    test_afr_count: int,
    test_random_count: int,
    train_afr_count: int,
    train_asian_count: int,
    train_random_count: int,
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

    validate_positions_count(heldout_positions, heldout_afr_count, f"{spec.name} heldout")
    validate_dataset_size(len(heldout_dataset), heldout_afr_count, heldout_output_name)
    validate_positions_count(test_positions, test_afr_count + test_random_count, f"{spec.name} test")
    validate_dataset_size(len(test_dataset), test_afr_count + test_random_count, test_output_name)
    validate_positions_count(
        train_positions,
        train_afr_count + train_asian_count + train_random_count,
        f"{spec.name} train",
    )
    validate_dataset_size(
        len(train_dataset),
        train_afr_count + train_asian_count + train_random_count,
        train_output_name,
    )

    condition_outputs = {
        heldout_output_name: (heldout_dataset, heldout_source_counts),
        test_output_name: (test_dataset, test_source_counts),
        train_output_name: (train_dataset, train_source_counts),
    }
    for output_name, (dataset, source_counts) in condition_outputs.items():
        output_path = output_dir / output_name
        dataset.to_pickle(output_path)
        logger.info(
            "Wrote %s rows=%d sources=%s ancestry_counts=%s",
            output_path,
            len(dataset),
            format_source_counts(source_counts),
            format_ancestry_counts(source_counts),
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
    pooled_tracking = pd.concat(tracking_parts, ignore_index=True)
    pooled_tracking = pooled_tracking.assign(
        logical_source_name=pooled_tracking["source_pickle_name"].map(logical_source_key),
        ancestry_group=pooled_tracking["source_pickle_name"].map(source_name_to_ancestry_group),
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


def ancestry_keys(tracking: pd.DataFrame, ancestry_group: str) -> set[tuple[str, int]]:
    ancestry_tracking = tracking.loc[
        tracking["ancestry_group"] == ancestry_group,
        ["logical_source_name", "source_row_number"],
    ]
    return {
        (str(row.logical_source_name), int(row.source_row_number))
        for row in ancestry_tracking.itertuples(index=False)
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


def build_condition_key_plans(
    *,
    pooled_data: PooledConditionData,
    shared_heldout_keys: set[tuple[str, int]],
    test_afr_count: int,
    test_random_count: int,
    train_afr_count: int,
    train_asian_count: int,
    train_random_count: int,
    no_covariates_rng: np.random.Generator,
    with_covariates_rng: np.random.Generator,
    logger: logging.Logger,
) -> dict[str, ConditionKeyPlan]:
    available = {
        ANCESTRY_AFR: ancestry_keys(pooled_data.tracking, ANCESTRY_AFR),
        ANCESTRY_ASIAN: ancestry_keys(pooled_data.tracking, ANCESTRY_ASIAN),
        ANCESTRY_EUROPEAN: ancestry_keys(pooled_data.tracking, ANCESTRY_EUROPEAN),
    }
    consume_key_set(available[ANCESTRY_AFR], shared_heldout_keys, "shared heldout AFR")

    no_cov_test_afr = draw_keys_from_pool(
        no_covariates_rng,
        available[ANCESTRY_AFR],
        test_afr_count,
        "no_covariates test AFR",
    )
    consume_key_set(available[ANCESTRY_AFR], no_cov_test_afr, "no_covariates test AFR")
    no_cov_train_afr = draw_keys_from_pool(
        no_covariates_rng,
        available[ANCESTRY_AFR],
        train_afr_count,
        "no_covariates train AFR",
    )
    consume_key_set(available[ANCESTRY_AFR], no_cov_train_afr, "no_covariates train AFR")

    no_cov_test_random = draw_keys_from_pool(
        no_covariates_rng,
        available[ANCESTRY_EUROPEAN],
        test_random_count,
        "no_covariates test random European",
    )
    consume_key_set(available[ANCESTRY_EUROPEAN], no_cov_test_random, "no_covariates test random European")
    no_cov_train_asian = draw_keys_from_pool(
        no_covariates_rng,
        available[ANCESTRY_ASIAN],
        train_asian_count,
        "no_covariates train Asian",
    )
    consume_key_set(available[ANCESTRY_ASIAN], no_cov_train_asian, "no_covariates train Asian")
    no_cov_train_random = draw_keys_from_pool(
        no_covariates_rng,
        available[ANCESTRY_EUROPEAN],
        train_random_count,
        "no_covariates train random European",
    )
    consume_key_set(available[ANCESTRY_EUROPEAN], no_cov_train_random, "no_covariates train random European")

    with_cov_test_afr = draw_keys_with_fallback(
        rng=with_covariates_rng,
        unique_available=available[ANCESTRY_AFR],
        count=test_afr_count,
        context="with_covariates test AFR",
        fallback_pools=[no_cov_train_afr, no_cov_test_afr],
    )
    consume_key_set(available[ANCESTRY_AFR], with_cov_test_afr & available[ANCESTRY_AFR], "with_covariates test AFR unique")

    with_cov_train_afr = draw_keys_with_fallback(
        rng=with_covariates_rng,
        unique_available=available[ANCESTRY_AFR],
        count=train_afr_count,
        context="with_covariates train AFR",
        fallback_pools=[no_cov_test_afr, no_cov_train_afr - with_cov_test_afr],
    )
    consume_key_set(available[ANCESTRY_AFR], with_cov_train_afr & available[ANCESTRY_AFR], "with_covariates train AFR unique")

    with_cov_test_random = draw_keys_from_pool(
        with_covariates_rng,
        available[ANCESTRY_EUROPEAN],
        test_random_count,
        "with_covariates test random European",
    )
    consume_key_set(available[ANCESTRY_EUROPEAN], with_cov_test_random, "with_covariates test random European")
    with_cov_train_asian = draw_keys_from_pool(
        with_covariates_rng,
        available[ANCESTRY_ASIAN],
        train_asian_count,
        "with_covariates train Asian",
    )
    consume_key_set(available[ANCESTRY_ASIAN], with_cov_train_asian, "with_covariates train Asian")
    with_cov_train_random = draw_keys_from_pool(
        with_covariates_rng,
        available[ANCESTRY_EUROPEAN],
        train_random_count,
        "with_covariates train random European",
    )
    consume_key_set(available[ANCESTRY_EUROPEAN], with_cov_train_random, "with_covariates train random European")

    no_cov_plan = ConditionKeyPlan(
        heldout_keys=shared_heldout_keys,
        test_keys=merge_key_sets(no_cov_test_afr, no_cov_test_random),
        train_keys=merge_key_sets(no_cov_train_afr, no_cov_train_asian, no_cov_train_random),
    )
    with_cov_plan = ConditionKeyPlan(
        heldout_keys=shared_heldout_keys,
        test_keys=merge_key_sets(with_cov_test_afr, with_cov_test_random),
        train_keys=merge_key_sets(with_cov_train_afr, with_cov_train_asian, with_cov_train_random),
    )

    log_plan_overlap(
        no_cov_plan=no_cov_plan,
        with_cov_plan=with_cov_plan,
        logger=logger,
    )
    return {
        "no_covariates": no_cov_plan,
        "with_covariates": with_cov_plan,
    }


def draw_keys_with_fallback(
    *,
    rng: np.random.Generator,
    unique_available: set[tuple[str, int]],
    count: int,
    context: str,
    fallback_pools: list[set[tuple[str, int]]],
) -> set[tuple[str, int]]:
    unique_count = min(count, len(unique_available))
    selected = draw_keys_from_pool(rng, unique_available, unique_count, context)
    shortfall = count - len(selected)
    if shortfall == 0:
        return selected

    fallback_selected: set[tuple[str, int]] = set()
    for pool in fallback_pools:
        candidates = set(pool) - selected - fallback_selected
        if not candidates:
            continue
        take_count = min(shortfall, len(candidates))
        chosen = draw_keys_from_pool(rng, candidates, take_count, context)
        fallback_selected.update(chosen)
        shortfall -= len(chosen)
        if shortfall == 0:
            break
    if shortfall != 0:
        raise ValueError(f"Requested {count} samples for {context}, but fallback pools were insufficient.")
    return selected | fallback_selected


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


def source_name_to_ancestry_group(source_name: str) -> str:
    if ANCESTRY_AFR in source_name:
        return ANCESTRY_AFR
    if ANCESTRY_ASIAN in source_name:
        return ANCESTRY_ASIAN
    if ANCESTRY_EUROPEAN in source_name:
        return ANCESTRY_EUROPEAN
    raise ValueError(f"Could not infer ancestry group from source name: {source_name}")


def format_ancestry_counts(source_counts: dict[str, int]) -> str:
    ancestry_counts: dict[str, int] = {}
    for source_name, count in source_counts.items():
        if count <= 0:
            continue
        ancestry_group = source_name_to_ancestry_group(source_name)
        ancestry_counts[ancestry_group] = ancestry_counts.get(ancestry_group, 0) + int(count)
    if not ancestry_counts:
        return "none"
    return ", ".join(
        f"{ancestry_group}:{ancestry_counts[ancestry_group]}"
        for ancestry_group in ANCESTRY_ORDER
        if ancestry_group in ancestry_counts
    )


def copy_transformations_file(input_dir: Path, output_path: Path) -> None:
    source_path = input_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source transformations file: {source_path}")
    output_path.write_text(source_path.read_text())


if __name__ == "__main__":
    main()