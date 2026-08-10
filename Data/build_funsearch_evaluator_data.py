from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ANCESTRY_COLUMNS = tuple(f"PC{index}" for index in range(1, 17))
TRAIN_POPULATIONS = ("AA", "JA", "LA")
DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parent / "RawData"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "FunsearchEvaluatorData"
DEFAULT_LOG_FILENAME = "build_funsearch_evaluator_data.log"
DEFAULT_TRANSFORMATIONS_FILENAME = "transformations.txt"
DEFAULT_TRACKING_FILENAME = "output_row_tracking.pkl"


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    train_files: tuple[str, str, str]
    test_files: tuple[str, str, str]


@dataclass(frozen=True)
class StandardizationTransform:
    center: tuple[float, ...]
    radius: float
    sample_count: int


CONDITIONS = (
    ConditionSpec(
        name="no_covariates",
        train_files=("train_AA.pkl", "train_JA.pkl", "train_LA.pkl"),
        test_files=("test_AA.pkl", "test_JA.pkl", "test_LA.pkl"),
    ),
    ConditionSpec(
        name="with_covariates",
        train_files=(
            "train_AA_add_covs.pkl",
            "train_JA_add_covs.pkl",
            "train_LA_add_covs.pkl",
        ),
        test_files=(
            "test_AA_add_covs.pkl",
            "test_JA_add_covs.pkl",
            "test_LA_add_covs.pkl",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build evaluator-ready MEC datasets in Data/FunsearchEvaluatorData/ "
            "without modifying Data/RawData/."
        )
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing the six raw pickle shards for each condition.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the evaluator pickles and transformations.txt are written.",
    )
    parser.add_argument(
        "--p-add",
        type=float,
        default=0.0,
        help="Percent of remaining JA train samples to add to the test set, after heldout sampling.",
    )
    parser.add_argument(
        "--m-ho",
        type=int,
        default=3,
        help="Multiplier for matched AA and LA heldout sampling relative to the JA heldout count.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Seed for reproducible sampling.",
    )
    return parser.parse_args()


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_funsearch_evaluator_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def main() -> None:
    args = parse_args()
    if args.m_ho < 0:
        raise ValueError("--m-ho must be non-negative.")
    if args.p_add < 0 or args.p_add > 100:
        raise ValueError("--p-add must be between 0 and 100.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / DEFAULT_LOG_FILENAME
    transformations_path = output_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    tracking_path = output_dir / DEFAULT_TRACKING_FILENAME
    logger = configure_logger(log_path)
    logger.info(
        "Starting evaluator data build raw_data_dir=%s output_dir=%s p_add=%s m_ho=%d random_seed=%d",
        args.raw_data_dir,
        output_dir,
        args.p_add,
        args.m_ho,
        args.random_seed,
    )

    seed_sequence = np.random.SeedSequence(args.random_seed)
    child_sequences = seed_sequence.spawn(1 + len(CONDITIONS))
    shared_heldout_rng = np.random.default_rng(child_sequences[0])
    shared_heldout_positions = build_shared_heldout_positions(
        raw_data_dir=args.raw_data_dir,
        m_ho=int(args.m_ho),
        rng=shared_heldout_rng,
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
            p_add=float(args.p_add),
            m_ho=int(args.m_ho),
            rng=condition_rng,
            shared_heldout_positions=shared_heldout_positions,
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
    logger.info("Finished evaluator data build")


def build_shared_heldout_positions(
    *,
    raw_data_dir: Path,
    m_ho: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    reference_spec = CONDITIONS[0]
    reference_frames = load_source_frames(reference_spec, raw_data_dir)
    train_lengths = {
        population: len(reference_frames[source_name])
        for population, source_name in zip(TRAIN_POPULATIONS, reference_spec.train_files)
    }
    heldout_ja_count = int(math.floor(0.20 * train_lengths["JA"]))
    heldout_counts = {
        "AA": m_ho * heldout_ja_count,
        "JA": heldout_ja_count,
        "LA": m_ho * heldout_ja_count,
    }
    shared_positions = {
        population: draw_positions(
            rng,
            set(range(train_lengths[population])),
            heldout_counts[population],
            f"shared heldout from {population}",
        )
        for population in TRAIN_POPULATIONS
    }
    logger.info(
        "Shared heldout row plan counts=%s",
        ", ".join(
            f"{population}:{len(shared_positions[population])}"
            for population in TRAIN_POPULATIONS
        ),
    )
    return shared_positions


def build_condition_datasets(
    *,
    spec: ConditionSpec,
    raw_data_dir: Path,
    output_dir: Path,
    p_add: float,
    m_ho: int,
    rng: np.random.Generator,
    shared_heldout_positions: dict[str, np.ndarray],
    logger: logging.Logger,
) -> tuple[StandardizationTransform, pd.DataFrame]:
    source_frames = load_source_frames(spec, raw_data_dir)
    source_paths = {
        source_name: (raw_data_dir / source_name).resolve()
        for source_name in (*spec.train_files, *spec.test_files)
    }
    transform = compute_standardization_transform(source_frames.values())
    standardized_frames = {
        source_name: standardize_ancestry_columns(frame, transform)
        for source_name, frame in source_frames.items()
    }

    train_aa, train_ja, train_la = spec.train_files
    test_aa, test_ja, test_la = spec.test_files

    remaining_positions = {
        source_name: set(range(len(source_frames[source_name])))
        for source_name in spec.train_files
    }

    heldout_ja_count = int(math.floor(0.20 * len(source_frames[train_ja])))
    heldout_counts = {
        train_aa: m_ho * heldout_ja_count,
        train_ja: heldout_ja_count,
        train_la: m_ho * heldout_ja_count,
    }
    heldout_positions = {}
    for population, source_name in zip(TRAIN_POPULATIONS, spec.train_files):
        positions = np.array(shared_heldout_positions[population], dtype=int, copy=True)
        expected_count = heldout_counts[source_name]
        if len(positions) != expected_count:
            raise ValueError(
                f"Shared heldout row count mismatch for {spec.name} {source_name}: "
                f"expected {expected_count}, got {len(positions)}."
            )
        if len(positions) and int(positions.max()) >= len(source_frames[source_name]):
            raise ValueError(
                f"Shared heldout row index out of bounds for {spec.name} {source_name}."
            )
        heldout_positions[source_name] = np.sort(positions)
    for source_name, positions in heldout_positions.items():
        remaining_positions[source_name].difference_update(positions.tolist())

    test_added_ja_count = int(math.floor((p_add / 100.0) * len(remaining_positions[train_ja])))
    test_train_counts = {
        train_aa: test_added_ja_count,
        train_ja: test_added_ja_count,
        train_la: test_added_ja_count,
    }
    test_train_positions = {
        source_name: draw_positions(
            rng,
            remaining_positions[source_name],
            test_train_counts[source_name],
            f"{spec.name} test augmentation from {source_name}",
        )
        for source_name in spec.train_files
    }
    for source_name, positions in test_train_positions.items():
        remaining_positions[source_name].difference_update(positions.tolist())

    validate_train_partitioning(
        spec=spec,
        source_frames=source_frames,
        heldout_positions=heldout_positions,
        test_train_positions=test_train_positions,
        remaining_positions=remaining_positions,
    )

    heldout_output_name = f"{spec.name}_heldout.pkl"
    heldout_dataset, heldout_source_counts, heldout_tracking = assemble_dataset(
        standardized_frames,
        source_paths,
        [
            (train_aa, heldout_positions[train_aa]),
            (train_ja, heldout_positions[train_ja]),
            (train_la, heldout_positions[train_la]),
        ],
        output_name=heldout_output_name,
    )
    test_output_name = f"{spec.name}_test.pkl"
    test_dataset, test_source_counts, test_tracking = assemble_dataset(
        standardized_frames,
        source_paths,
        [
            (test_aa, np.arange(len(standardized_frames[test_aa]), dtype=int)),
            (test_ja, np.arange(len(standardized_frames[test_ja]), dtype=int)),
            (test_la, np.arange(len(standardized_frames[test_la]), dtype=int)),
            (train_aa, test_train_positions[train_aa]),
            (train_ja, test_train_positions[train_ja]),
            (train_la, test_train_positions[train_la]),
        ],
        output_name=test_output_name,
    )
    train_output_name = f"{spec.name}_train.pkl"
    train_dataset, train_source_counts, train_tracking = assemble_dataset(
        standardized_frames,
        source_paths,
        [
            (train_aa, np.array(sorted(remaining_positions[train_aa]), dtype=int)),
            (train_ja, np.array(sorted(remaining_positions[train_ja]), dtype=int)),
            (train_la, np.array(sorted(remaining_positions[train_la]), dtype=int)),
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
            "Wrote %s rows=%d sources=%s",
            output_path,
            len(dataset),
            format_source_counts(source_counts),
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


def load_source_frames(spec: ConditionSpec, raw_data_dir: Path) -> dict[str, pd.DataFrame]:
    source_frames: dict[str, pd.DataFrame] = {}
    for source_name in (*spec.train_files, *spec.test_files):
        source_path = raw_data_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw source pickle: {source_path}")
        frame = pd.read_pickle(source_path)
        missing_ancestry_columns = [column for column in ANCESTRY_COLUMNS if column not in frame.columns]
        if missing_ancestry_columns:
            raise ValueError(
                f"Source pickle {source_path} is missing ancestry columns: {missing_ancestry_columns}"
            )
        source_frames[source_name] = frame.reset_index(drop=True).copy()
    return source_frames


def compute_standardization_transform(
    frames: object,
) -> StandardizationTransform:
    ancestry_arrays = [
        frame.loc[:, ANCESTRY_COLUMNS].to_numpy(dtype=float, copy=True)
        for frame in frames
    ]
    if not ancestry_arrays:
        raise ValueError("At least one source frame is required to compute ancestry standardization.")
    ancestry_matrix = np.vstack(ancestry_arrays)
    center = ancestry_matrix.mean(axis=0)
    distances = np.linalg.norm(ancestry_matrix - center, axis=1)
    radius = radius_covering_fraction(distances, 0.95)
    if radius <= 0.0:
        radius = 1.0
    return StandardizationTransform(
        center=tuple(float(value) for value in center),
        radius=float(radius),
        sample_count=int(ancestry_matrix.shape[0]),
    )


def radius_covering_fraction(distances: np.ndarray, fraction: float) -> float:
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("Distances must be a non-empty one-dimensional array.")
    rank = max(0, math.ceil(fraction * distances.size) - 1)
    return float(np.partition(distances, rank)[rank])


def standardize_ancestry_columns(
    frame: pd.DataFrame,
    transform: StandardizationTransform,
) -> pd.DataFrame:
    standardized = frame.copy()
    ancestry_values = standardized.loc[:, ANCESTRY_COLUMNS].to_numpy(dtype=float, copy=True)
    center = np.asarray(transform.center, dtype=float)
    standardized.loc[:, ANCESTRY_COLUMNS] = (ancestry_values - center) / transform.radius
    return standardized


def draw_positions(
    rng: np.random.Generator,
    available_positions: set[int],
    count: int,
    context: str,
) -> np.ndarray:
    if count < 0:
        raise ValueError(f"Sample count must be non-negative for {context}.")
    available = np.array(sorted(available_positions), dtype=int)
    if count > available.size:
        raise ValueError(
            f"Requested {count} samples for {context}, but only {available.size} are available."
        )
    if count == 0:
        return np.array([], dtype=int)
    selected = rng.choice(available, size=count, replace=False)
    return np.sort(selected.astype(int))


def validate_train_partitioning(
    *,
    spec: ConditionSpec,
    source_frames: dict[str, pd.DataFrame],
    heldout_positions: dict[str, np.ndarray],
    test_train_positions: dict[str, np.ndarray],
    remaining_positions: dict[str, set[int]],
) -> None:
    for source_name in spec.train_files:
        heldout_set = set(heldout_positions[source_name].tolist())
        test_set = set(test_train_positions[source_name].tolist())
        remaining_set = set(remaining_positions[source_name])
        if heldout_set & test_set:
            raise ValueError(f"Overlapping heldout/test selections detected for {source_name}.")
        if heldout_set & remaining_set:
            raise ValueError(f"Heldout/train overlap detected for {source_name}.")
        if test_set & remaining_set:
            raise ValueError(f"Test/train overlap detected for {source_name}.")
        total_partitioned = len(heldout_set) + len(test_set) + len(remaining_set)
        expected_total = len(source_frames[source_name])
        if total_partitioned != expected_total:
            raise ValueError(
                f"Partitioning mismatch for {source_name}: expected {expected_total}, got {total_partitioned}."
            )


def assemble_dataset(
    standardized_frames: dict[str, pd.DataFrame],
    source_paths: dict[str, Path],
    selections: list[tuple[str, np.ndarray]],
    *,
    output_name: str,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    tracking_parts: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    output_row_number = 0
    for source_name, positions in selections:
        if positions.size == 0:
            continue
        parts.append(standardized_frames[source_name].iloc[positions].copy())
        source_counts[source_name] = int(positions.size)
        tracking_parts.append(
            pd.DataFrame(
                {
                    "output_pickle_name": output_name,
                    "output_row_number": np.arange(
                        output_row_number,
                        output_row_number + len(positions),
                        dtype=int,
                    ),
                    "source_pickle_name": source_name,
                    "source_pickle_path": str(source_paths[source_name]),
                    "source_row_number": positions.astype(int),
                }
            )
        )
        output_row_number += len(positions)
    if not parts:
        empty_dataset = standardized_frames[next(iter(standardized_frames))].iloc[0:0].copy()
        empty_tracking = pd.DataFrame(
            columns=[
                "output_pickle_name",
                "output_row_number",
                "source_pickle_name",
                "source_pickle_path",
                "source_row_number",
            ]
        )
        return empty_dataset, {}, empty_tracking
    return (
        pd.concat(parts, ignore_index=True),
        source_counts,
        pd.concat(tracking_parts, ignore_index=True),
    )


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


def format_source_counts(source_counts: dict[str, int]) -> str:
    non_zero_counts = [
        f"{source_name}:{count}"
        for source_name, count in source_counts.items()
        if count > 0
    ]
    return ", ".join(non_zero_counts) if non_zero_counts else "none"


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()