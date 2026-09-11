from __future__ import annotations

"""Build AFR focal evaluator datasets with refit raw-coordinate standardization.

This builder reuses the exact subject membership and row order from
Data/FunsearchEvaluatorDataAFRFocal2, but it does not reuse that dataset's
standardized ancestry coordinates. Instead, for each condition it reloads the
matching raw OncoArray rows from Data/RawDataOncoArray, pools every subject in
that condition's train/test/heldout outputs, fits a fresh ancestry transform,
and rewrites PC1-PC16 as (a - a*) / r.

Outputs:
- no_covariates_{heldout,test,train}.pkl
- with_covariates_{heldout,test,train}.pkl
- output_row_tracking.pkl
- transformations.txt
- one histogram PNG per condition comparing AFRFocal2 vs AFRFocal3 ancestry
  coordinates for the same subjects
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from build_funsearch_evaluator_data import ANCESTRY_COLUMNS
from build_funsearch_evaluator_data import StandardizationTransform
from build_funsearch_evaluator_data import compute_standardization_transform
from build_funsearch_evaluator_data import configure_logger
from build_funsearch_evaluator_data import format_float
from build_funsearch_evaluator_data import format_source_counts
from build_funsearch_evaluator_data import standardize_ancestry_columns


DEFAULT_REFERENCE_DIR = DATA_DIR / "FunsearchEvaluatorDataAFRFocal2"
DEFAULT_RAW_DATA_DIR = DATA_DIR / "RawDataOncoArray"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILENAME = "build_funsearch_evaluator_data.log"
DEFAULT_TRANSFORMATIONS_FILENAME = "transformations.txt"
DEFAULT_TRACKING_FILENAME = "output_row_tracking.pkl"
DEFAULT_HISTOGRAM_FILENAME_TEMPLATE = "{condition_name}_distance_from_afrfocal2_histogram.png"
OUTPUT_SPLITS = ("heldout", "test", "train")


@dataclass(frozen=True)
class ConditionSpec:
    name: str


@dataclass(frozen=True)
class PreparedOutput:
    output_name: str
    reference_dataset: pd.DataFrame
    raw_dataset: pd.DataFrame
    tracking: pd.DataFrame


@dataclass(frozen=True)
class ConditionBuildResult:
    transform: StandardizationTransform
    tracking: pd.DataFrame
    distance_values: np.ndarray


CONDITIONS = (
    ConditionSpec(name="no_covariates"),
    ConditionSpec(name="with_covariates"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AFR focal evaluator datasets in Data/FunsearchEvaluatorDataAFRFocal3/ "
            "using AFRFocal2 subject membership and raw OncoArray ancestry coordinates."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing the existing AFRFocal2 outputs and output_row_tracking.pkl.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing the raw, unstandardized OncoArray shard pickles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where AFRFocal3 outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / DEFAULT_LOG_FILENAME
    transformations_path = output_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    tracking_path = output_dir / DEFAULT_TRACKING_FILENAME
    logger = configure_logger(log_path)

    reference_tracking_path = args.reference_dir / DEFAULT_TRACKING_FILENAME
    if not reference_tracking_path.exists():
        raise FileNotFoundError(f"Missing AFRFocal2 tracking file: {reference_tracking_path}")
    reference_tracking = pd.read_pickle(reference_tracking_path)

    logger.info(
        "Starting AFRFocal3 build reference_dir=%s raw_data_dir=%s output_dir=%s",
        args.reference_dir,
        args.raw_data_dir,
        output_dir,
    )

    transforms: dict[str, StandardizationTransform] = {}
    tracking_frames: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        result = build_condition_datasets(
            spec=condition,
            reference_dir=args.reference_dir,
            raw_data_dir=args.raw_data_dir,
            output_dir=output_dir,
            reference_tracking=reference_tracking,
            logger=logger,
        )
        transforms[condition.name] = result.transform
        tracking_frames.append(result.tracking)

    write_transformations_file(transformations_path, transforms)
    logger.info("Wrote ancestry transformations to %s", transformations_path)

    tracking_frame = pd.concat(tracking_frames, ignore_index=True)
    tracking_frame.to_pickle(tracking_path)
    logger.info("Wrote output row tracking to %s rows=%d", tracking_path, len(tracking_frame))
    logger.info("Log file path=%s", log_path)
    logger.info("Finished AFRFocal3 build")


def build_condition_datasets(
    *,
    spec: ConditionSpec,
    reference_dir: Path,
    raw_data_dir: Path,
    output_dir: Path,
    reference_tracking: pd.DataFrame,
    logger: logging.Logger,
) -> ConditionBuildResult:
    raw_frame_cache: dict[str, pd.DataFrame] = {}
    prepared_outputs: list[PreparedOutput] = []
    pooled_raw_frames: list[pd.DataFrame] = []
    for split_name in OUTPUT_SPLITS:
        prepared_output = prepare_output(
            spec=spec,
            split_name=split_name,
            reference_dir=reference_dir,
            raw_data_dir=raw_data_dir,
            reference_tracking=reference_tracking,
            raw_frame_cache=raw_frame_cache,
        )
        prepared_outputs.append(prepared_output)
        pooled_raw_frames.append(prepared_output.raw_dataset)

    transform = compute_standardization_transform(pooled_raw_frames)
    logger.info(
        "Condition %s standardization sample_count=%d r=%s a_star=%s",
        spec.name,
        transform.sample_count,
        format_float(transform.radius),
        ",".join(format_float(value) for value in transform.center),
    )

    distance_parts: list[np.ndarray] = []
    tracking_parts: list[pd.DataFrame] = []
    for prepared_output in prepared_outputs:
        standardized_dataset = standardize_ancestry_columns(prepared_output.raw_dataset, transform)
        validate_reference_alignment(
            reference_dataset=prepared_output.reference_dataset,
            raw_dataset=prepared_output.raw_dataset,
            standardized_dataset=standardized_dataset,
            output_name=prepared_output.output_name,
        )
        output_path = output_dir / prepared_output.output_name
        standardized_dataset.to_pickle(output_path)

        source_counts = prepared_output.tracking["source_pickle_name"].value_counts(sort=False).to_dict()
        logger.info(
            "Wrote %s rows=%d sources=%s",
            output_path,
            len(standardized_dataset),
            format_source_counts({str(key): int(value) for key, value in source_counts.items()}),
        )

        distances = ancestry_distance_between_datasets(
            prepared_output.reference_dataset,
            standardized_dataset,
        )
        logger.info(
            "Compared %s against AFRFocal2 mean_distance=%s median_distance=%s max_distance=%s",
            prepared_output.output_name,
            format_float(float(np.mean(distances))),
            format_float(float(np.median(distances))),
            format_float(float(np.max(distances))),
        )
        distance_parts.append(distances)
        tracking_parts.append(prepared_output.tracking)

    all_distances = np.concatenate(distance_parts)
    histogram_path = output_dir / DEFAULT_HISTOGRAM_FILENAME_TEMPLATE.format(condition_name=spec.name)
    write_distance_histogram(
        condition_name=spec.name,
        distances=all_distances,
        output_path=histogram_path,
    )
    logger.info(
        "Wrote distance histogram to %s sample_count=%d",
        histogram_path,
        len(all_distances),
    )

    return ConditionBuildResult(
        transform=transform,
        tracking=pd.concat(tracking_parts, ignore_index=True),
        distance_values=all_distances,
    )


def prepare_output(
    *,
    spec: ConditionSpec,
    split_name: str,
    reference_dir: Path,
    raw_data_dir: Path,
    reference_tracking: pd.DataFrame,
    raw_frame_cache: dict[str, pd.DataFrame],
) -> PreparedOutput:
    output_name = f"{spec.name}_{split_name}.pkl"
    reference_path = reference_dir / output_name
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing AFRFocal2 dataset: {reference_path}")

    reference_dataset = pd.read_pickle(reference_path).reset_index(drop=True)
    tracking = reference_tracking.loc[
        reference_tracking["output_pickle_name"] == output_name
    ].sort_values("output_row_number").reset_index(drop=True)
    if len(reference_dataset) != len(tracking):
        raise ValueError(
            f"Tracking length mismatch for {output_name}: dataset has {len(reference_dataset)} rows, "
            f"tracking has {len(tracking)} rows."
        )
    expected_output_rows = np.arange(len(tracking), dtype=int)
    actual_output_rows = tracking["output_row_number"].to_numpy(dtype=int, copy=False)
    if not np.array_equal(actual_output_rows, expected_output_rows):
        raise ValueError(f"Output row order mismatch for {output_name}.")

    raw_dataset = assemble_raw_dataset(
        tracking=tracking,
        raw_data_dir=raw_data_dir,
        raw_frame_cache=raw_frame_cache,
    )
    normalized_tracking = tracking.copy()
    normalized_tracking.loc[:, "source_pickle_path"] = normalized_tracking["source_pickle_name"].map(
        lambda source_name: str((raw_data_dir / str(source_name)).resolve())
    )
    return PreparedOutput(
        output_name=output_name,
        reference_dataset=reference_dataset,
        raw_dataset=raw_dataset,
        tracking=normalized_tracking,
    )


def assemble_raw_dataset(
    *,
    tracking: pd.DataFrame,
    raw_data_dir: Path,
    raw_frame_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    row_parts: list[pd.DataFrame] = []
    for row in tracking.itertuples(index=False):
        source_name = str(row.source_pickle_name)
        if source_name not in raw_frame_cache:
            source_path = raw_data_dir / source_name
            if not source_path.exists():
                raise FileNotFoundError(f"Missing raw source pickle: {source_path}")
            raw_frame_cache[source_name] = pd.read_pickle(source_path).reset_index(drop=True)
        raw_frame = raw_frame_cache[source_name]
        source_row_number = int(row.source_row_number)
        if source_row_number < 0 or source_row_number >= len(raw_frame):
            raise ValueError(
                f"Source row out of bounds for {source_name}: {source_row_number} not in [0, {len(raw_frame)})"
            )
        row_parts.append(raw_frame.iloc[[source_row_number]].copy())

    if not row_parts:
        return pd.DataFrame(columns=list(ANCESTRY_COLUMNS))
    return pd.concat(row_parts, ignore_index=True)


def validate_reference_alignment(
    *,
    reference_dataset: pd.DataFrame,
    raw_dataset: pd.DataFrame,
    standardized_dataset: pd.DataFrame,
    output_name: str,
) -> None:
    if list(reference_dataset.columns) != list(raw_dataset.columns):
        raise ValueError(f"Column mismatch between AFRFocal2 and raw reconstruction for {output_name}.")
    if list(reference_dataset.columns) != list(standardized_dataset.columns):
        raise ValueError(f"Column mismatch after standardization for {output_name}.")

    non_ancestry_columns = [column for column in reference_dataset.columns if column not in ANCESTRY_COLUMNS]
    if not reference_dataset.loc[:, non_ancestry_columns].equals(raw_dataset.loc[:, non_ancestry_columns]):
        raise ValueError(
            f"Non-ancestry column mismatch while reconstructing {output_name} from raw rows."
        )
    if not reference_dataset.loc[:, non_ancestry_columns].equals(
        standardized_dataset.loc[:, non_ancestry_columns]
    ):
        raise ValueError(
            f"Non-ancestry columns changed unexpectedly after standardization for {output_name}."
        )


def ancestry_distance_between_datasets(
    reference_dataset: pd.DataFrame,
    standardized_dataset: pd.DataFrame,
) -> np.ndarray:
    reference_values = reference_dataset.loc[:, ANCESTRY_COLUMNS].to_numpy(dtype=float, copy=True)
    standardized_values = standardized_dataset.loc[:, ANCESTRY_COLUMNS].to_numpy(dtype=float, copy=True)
    if reference_values.shape != standardized_values.shape:
        raise ValueError("Ancestry coordinate shape mismatch while computing dataset distances.")
    return np.linalg.norm(reference_values - standardized_values, axis=1)


def write_distance_histogram(
    *,
    condition_name: str,
    distances: np.ndarray,
    output_path: Path,
) -> None:
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError(f"Expected a non-empty one-dimensional distance array for {condition_name}.")

    figure, axis = plt.subplots(figsize=(9, 6))
    bin_count = max(10, min(60, int(np.sqrt(distances.size))))
    axis.hist(distances, bins=bin_count, color="#4c78a8", edgecolor="white", linewidth=0.5)
    axis.axvline(float(np.mean(distances)), color="#f58518", linewidth=2, linestyle="--")
    axis.set_title(f"{condition_name} ancestry shift: AFRFocal2 to AFRFocal3")
    axis.set_xlabel("Euclidean distance between standardized PC1-PC16 vectors")
    axis.set_ylabel("Sample count")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
                "a_star=" + ",".join(format_float(value) for value in transform.center),
                f"r={format_float(transform.radius)}",
                "",
            ]
        )
    transformations_path.write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()