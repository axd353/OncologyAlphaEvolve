from __future__ import annotations

"""Build evaluator-ready OncoArray datasets without modifying raw pickles.

This mirrors the MEC builder where possible, but uses the OncoArray shard names
and a simpler split policy:

- heldout = all rows from the raw test shards
- test = a calibration split sampled from the raw train shards
- train = the remaining raw train rows

The output pickles remain plain pandas DataFrames. Missing-value imputation is
still deferred to the evaluator, matching the MEC pipeline contract.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from build_funsearch_evaluator_data import assemble_dataset
from build_funsearch_evaluator_data import compute_standardization_transform
from build_funsearch_evaluator_data import ConditionSpec
from build_funsearch_evaluator_data import configure_logger
from build_funsearch_evaluator_data import draw_positions
from build_funsearch_evaluator_data import format_float
from build_funsearch_evaluator_data import format_source_counts
from build_funsearch_evaluator_data import load_source_frames
from build_funsearch_evaluator_data import StandardizationTransform
from build_funsearch_evaluator_data import standardize_ancestry_columns


TRAIN_POPULATIONS = ("African_Ancestry", "Asian", "European")
DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parent / "RawDataOncoArray"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "FunsearchEvaluatorDataOncoArray"
DEFAULT_LOG_FILENAME = "build_funsearch_evaluator_data.log"
DEFAULT_TRANSFORMATIONS_FILENAME = "transformations.txt"
DEFAULT_TRACKING_FILENAME = "output_row_tracking.pkl"
DEFAULT_CALIBRATION_FRACTION = 0.20


CONDITIONS = (
    ConditionSpec(
        name="no_covariates",
        train_files=(
            "train_African_Ancestry.pkl",
            "train_Asian.pkl",
            "train_European.pkl",
        ),
        test_files=(
            "test_African_Ancestry.pkl",
            "test_Asian.pkl",
            "test_European.pkl",
        ),
    ),
    ConditionSpec(
        name="with_covariates",
        train_files=(
            "train_African_Ancestry_add_covs.pkl",
            "train_Asian_add_covs.pkl",
            "train_European_add_covs.pkl",
        ),
        test_files=(
            "test_African_Ancestry_add_covs.pkl",
            "test_Asian_add_covs.pkl",
            "test_European_add_covs.pkl",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build evaluator-ready OncoArray datasets in "
            "Data/FunsearchEvaluatorDataOncoArray/ without modifying "
            "Data/RawDataOncoArray/."
        )
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing the six raw OncoArray pickle shards for each condition.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the evaluator pickles and transformations.txt are written.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help="Fraction of each raw train shard sampled into the calibration/test output.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Seed for reproducible train/calibration splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_fraction < 0.0 or args.calibration_fraction >= 1.0:
        raise ValueError("--calibration-fraction must be in the interval [0, 1).")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / DEFAULT_LOG_FILENAME
    transformations_path = output_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    tracking_path = output_dir / DEFAULT_TRACKING_FILENAME
    logger = configure_logger(log_path)
    logger.info(
        "Starting OncoArray evaluator data build raw_data_dir=%s output_dir=%s calibration_fraction=%s random_seed=%d",
        args.raw_data_dir,
        output_dir,
        args.calibration_fraction,
        args.random_seed,
    )

    seed_sequence = np.random.SeedSequence(args.random_seed)
    child_sequences = seed_sequence.spawn(1 + len(CONDITIONS))
    shared_split_rng = np.random.default_rng(child_sequences[0])
    shared_calibration_positions = build_shared_calibration_positions(
        raw_data_dir=args.raw_data_dir,
        calibration_fraction=float(args.calibration_fraction),
        rng=shared_split_rng,
        logger=logger,
    )

    transforms: dict[str, StandardizationTransform] = {}
    tracking_frames: list[pd.DataFrame] = []
    for condition, child_sequence in zip(CONDITIONS, child_sequences[1:]):
        condition_rng = np.random.default_rng(child_sequence)
        transform, condition_tracking = build_condition_datasets(
            spec=condition,
            raw_data_dir=args.raw_data_dir,
            output_dir=output_dir,
            calibration_fraction=float(args.calibration_fraction),
            rng=condition_rng,
            shared_calibration_positions=shared_calibration_positions,
            logger=logger,
        )
        transforms[condition.name] = transform
        tracking_frames.append(condition_tracking)

    write_transformations_file(transformations_path, transforms)
    logger.info("Wrote ancestry transformations to %s", transformations_path)

    tracking_frame = pd.concat(tracking_frames, ignore_index=True)
    tracking_frame.to_pickle(tracking_path)
    logger.info("Wrote output row tracking to %s rows=%d", tracking_path, len(tracking_frame))
    logger.info("Log file path=%s", log_path)
    logger.info("Finished OncoArray evaluator data build")


def build_shared_calibration_positions(
    *,
    raw_data_dir: Path,
    calibration_fraction: float,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    reference_spec = CONDITIONS[0]
    reference_frames = load_source_frames(reference_spec, raw_data_dir)
    train_lengths = {
        population: len(reference_frames[source_name])
        for population, source_name in zip(TRAIN_POPULATIONS, reference_spec.train_files)
    }
    calibration_positions = {
        population: draw_positions(
            rng,
            set(range(train_lengths[population])),
            int(train_lengths[population] * calibration_fraction),
            f"shared calibration from {population}",
        )
        for population in TRAIN_POPULATIONS
    }
    logger.info(
        "Shared calibration row plan counts=%s",
        ", ".join(
            f"{population}:{len(calibration_positions[population])}"
            for population in TRAIN_POPULATIONS
        ),
    )
    return calibration_positions


def build_condition_datasets(
    *,
    spec: ConditionSpec,
    raw_data_dir: Path,
    output_dir: Path,
    calibration_fraction: float,
    rng: np.random.Generator,
    shared_calibration_positions: dict[str, np.ndarray],
    logger: logging.Logger,
) -> tuple[StandardizationTransform, pd.DataFrame]:
    del calibration_fraction
    del rng

    source_frames = load_source_frames(spec, raw_data_dir)
    source_paths = {
        source_name: (raw_data_dir / source_name).resolve()
        for source_name in (*spec.train_files, *spec.test_files)
    }
    # Fit one ancestry transform across all six raw shards in the condition,
    # matching the MEC builder's pooled standardization rule.
    transform = compute_standardization_transform(source_frames.values())
    standardized_frames = {
        source_name: standardize_ancestry_columns(frame, transform)
        for source_name, frame in source_frames.items()
    }

    calibration_positions: dict[str, np.ndarray] = {}
    remaining_positions: dict[str, np.ndarray] = {}
    for population, source_name in zip(TRAIN_POPULATIONS, spec.train_files):
        positions = np.array(shared_calibration_positions[population], dtype=int, copy=True)
        expected_count = len(positions)
        if len(positions) and int(positions.max()) >= len(source_frames[source_name]):
            raise ValueError(
                f"Shared calibration row index out of bounds for {spec.name} {source_name}."
            )
        calibration_positions[source_name] = np.sort(positions)

        selected_set = set(calibration_positions[source_name].tolist())
        if len(selected_set) != expected_count:
            raise ValueError(f"Duplicate calibration selections detected for {source_name}.")

        all_positions = np.arange(len(source_frames[source_name]), dtype=int)
        train_positions = np.array(
            [position for position in all_positions if position not in selected_set],
            dtype=int,
        )
        if len(train_positions) + len(calibration_positions[source_name]) != len(source_frames[source_name]):
            raise ValueError(f"Partitioning mismatch for {source_name}.")
        remaining_positions[source_name] = train_positions

    heldout_output_name = f"{spec.name}_heldout.pkl"
    heldout_dataset, heldout_source_counts, heldout_tracking = assemble_dataset(
        standardized_frames,
        source_paths,
        [
            (source_name, np.arange(len(standardized_frames[source_name]), dtype=int))
            for source_name in spec.test_files
        ],
        output_name=heldout_output_name,
    )
    test_output_name = f"{spec.name}_test.pkl"
    test_dataset, test_source_counts, test_tracking = assemble_dataset(
        standardized_frames,
        source_paths,
        [
            (source_name, calibration_positions[source_name])
            for source_name in spec.train_files
        ],
        output_name=test_output_name,
    )
    train_output_name = f"{spec.name}_train.pkl"
    train_dataset, train_source_counts, train_tracking = assemble_dataset(
        standardized_frames,
        source_paths,
        [
            (source_name, remaining_positions[source_name])
            for source_name in spec.train_files
        ],
        output_name=train_output_name,
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
        "Condition %s standardization sample_count=%d radius=%s center=%s",
        spec.name,
        transform.sample_count,
        format_float(transform.radius),
        ",".join(format_float(value) for value in transform.center),
    )

    condition_tracking = pd.concat(
        [heldout_tracking, test_tracking, train_tracking],
        ignore_index=True,
    )
    return transform, condition_tracking


def write_transformations_file(
    transformations_path: Path,
    transforms: dict[str, StandardizationTransform],
) -> None:
    transformations_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for condition in CONDITIONS:
        transform = transforms[condition.name]
        lines.extend(
            [
                f"[{condition.name}]",
                f"sample_count={transform.sample_count}",
                "ancestry_center=" + ",".join(format_float(value) for value in transform.center),
                f"radius={format_float(transform.radius)}",
                "",
            ]
        )
    transformations_path.write_text("\n".join(lines).rstrip() + "\n")


def source_name_to_ancestry_group(source_name: str) -> str:
    stem = Path(source_name).stem
    if stem.endswith("_add_covs"):
        stem = stem[: -len("_add_covs")]
    return stem.split("_", 1)[1]


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
        for ancestry_group in TRAIN_POPULATIONS
        if ancestry_group in ancestry_counts
    )


if __name__ == "__main__":
    main()