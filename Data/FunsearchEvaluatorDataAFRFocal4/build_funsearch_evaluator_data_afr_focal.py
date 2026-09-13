from __future__ import annotations

"""Build AFR focal evaluator datasets with coordinate-wise raw PCA standardization.

This builder reuses the exact subject membership and row order from
Data/FunsearchEvaluatorDataAFRFocal2, but it does not reuse that dataset's
standardized ancestry coordinates. Instead, for each condition it reloads the
matching raw OncoArray rows from Data/RawDataOncoArray, pools every subject in
that condition's train/test/heldout outputs, fits a fresh ancestry transform,
and rewrites PC1-PC16 as (a - a_star) / r coordinate-wise.

Outputs:
- no_covariates_{heldout,test,train}.pkl
- with_covariates_{heldout,test,train}.pkl
- output_row_tracking.pkl
- transformations.txt
- AFRICAN_ANCESTRY.png
- ASIAN.png
- EUROPEAN.png
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from build_funsearch_evaluator_data import ANCESTRY_COLUMNS
from build_funsearch_evaluator_data import configure_logger
from build_funsearch_evaluator_data import format_float
from build_funsearch_evaluator_data import format_source_counts


DEFAULT_REFERENCE_DIR = DATA_DIR / "FunsearchEvaluatorDataAFRFocal2"
DEFAULT_RAW_DATA_DIR = DATA_DIR / "RawDataOncoArray"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILENAME = "build_funsearch_evaluator_data.log"
DEFAULT_TRANSFORMATIONS_FILENAME = "transformations.txt"
DEFAULT_TRACKING_FILENAME = "output_row_tracking.pkl"
OUTPUT_SPLITS = ("heldout", "test", "train")
RAW_PLOT_FILENAME_TEMPLATE = "{ancestry_group}.png"


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
class CoordinatewiseStandardizationTransform:
    center: tuple[float, ...]
    radii: tuple[float, ...]
    sample_count: int


@dataclass(frozen=True)
class ConditionBuildResult:
    transform: CoordinatewiseStandardizationTransform
    tracking: pd.DataFrame


@dataclass(frozen=True)
class RawPlotSpec:
    ancestry_group: str
    source_files: tuple[str, ...]


CONDITIONS = (
    ConditionSpec(name="no_covariates"),
    ConditionSpec(name="with_covariates"),
)

RAW_PLOT_SPECS = (
    RawPlotSpec(
        ancestry_group="AFRICAN_ANCESTRY",
        source_files=("train_African_Ancestry.pkl", "test_African_Ancestry.pkl"),
    ),
    RawPlotSpec(
        ancestry_group="ASIAN",
        source_files=("train_Asian.pkl", "test_Asian.pkl"),
    ),
    RawPlotSpec(
        ancestry_group="EUROPEAN",
        source_files=("train_European.pkl", "test_European.pkl"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AFR focal evaluator datasets in Data/FunsearchEvaluatorDataAFRFocal4/ "
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
        help="Directory where AFRFocal4 outputs are written.",
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
        "Starting AFRFocal4 build reference_dir=%s raw_data_dir=%s output_dir=%s",
        args.reference_dir,
        args.raw_data_dir,
        output_dir,
    )

    transforms: dict[str, CoordinatewiseStandardizationTransform] = {}
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

    write_raw_ancestry_boxplots(
        raw_data_dir=args.raw_data_dir,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("Log file path=%s", log_path)
    logger.info("Finished AFRFocal4 build")


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

    transform = compute_coordinatewise_standardization_transform(pooled_raw_frames)
    logger.info(
        "Condition %s standardization sample_count=%d r=%s a_star=%s",
        spec.name,
        transform.sample_count,
        ",".join(format_float(value) for value in transform.radii),
        ",".join(format_float(value) for value in transform.center),
    )

    tracking_parts: list[pd.DataFrame] = []
    for prepared_output in prepared_outputs:
        standardized_dataset = standardize_ancestry_columns_coordinatewise(
            prepared_output.raw_dataset,
            transform,
        )
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

        tracking_parts.append(prepared_output.tracking)

    return ConditionBuildResult(
        transform=transform,
        tracking=pd.concat(tracking_parts, ignore_index=True),
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


def compute_coordinatewise_standardization_transform(
    frames: object,
) -> CoordinatewiseStandardizationTransform:
    ancestry_arrays = [
        frame.loc[:, ANCESTRY_COLUMNS].to_numpy(dtype=float, copy=True)
        for frame in frames
    ]
    if not ancestry_arrays:
        raise ValueError("At least one source frame is required to compute ancestry standardization.")
    ancestry_matrix = np.vstack(ancestry_arrays)
    center = ancestry_matrix.mean(axis=0)
    deviations = np.abs(ancestry_matrix - center)

    radii: list[float] = []
    for column_index in range(deviations.shape[1]):
        radius = radius_covering_fraction(deviations[:, column_index], 0.95)
        if radius <= 0.0:
            radius = 1.0
        radii.append(float(radius))

    return CoordinatewiseStandardizationTransform(
        center=tuple(float(value) for value in center),
        radii=tuple(radii),
        sample_count=int(ancestry_matrix.shape[0]),
    )


def radius_covering_fraction(distances: np.ndarray, fraction: float) -> float:
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("Distances must be a non-empty one-dimensional array.")
    rank = max(0, math.ceil(fraction * distances.size) - 1)
    return float(np.partition(distances, rank)[rank])


def standardize_ancestry_columns_coordinatewise(
    frame: pd.DataFrame,
    transform: CoordinatewiseStandardizationTransform,
) -> pd.DataFrame:
    standardized = frame.copy()
    ancestry_values = standardized.loc[:, ANCESTRY_COLUMNS].to_numpy(dtype=float, copy=True)
    center = np.asarray(transform.center, dtype=float)
    radii = np.asarray(transform.radii, dtype=float)
    standardized.loc[:, ANCESTRY_COLUMNS] = (ancestry_values - center) / radii
    return standardized


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


def write_transformations_file(
    transformations_path: Path,
    transforms: dict[str, CoordinatewiseStandardizationTransform],
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
                "r=" + ",".join(format_float(value) for value in transform.radii),
                "",
            ]
        )
    transformations_path.write_text("\n".join(lines).rstrip() + "\n")


def write_raw_ancestry_boxplots(
    *,
    raw_data_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    raw_frames_by_group: dict[str, pd.DataFrame] = {}
    global_y_limits: tuple[float, float] | None = None
    for plot_spec in RAW_PLOT_SPECS:
        raw_frame = load_raw_plot_frame(raw_data_dir, plot_spec.source_files)
        raw_frames_by_group[plot_spec.ancestry_group] = raw_frame
        frame_limits = ancestry_y_limits(raw_frame)
        if global_y_limits is None:
            global_y_limits = frame_limits
        else:
            global_y_limits = (
                min(global_y_limits[0], frame_limits[0]),
                max(global_y_limits[1], frame_limits[1]),
            )

    if global_y_limits is None:
        raise ValueError("At least one raw ancestry plot frame is required.")

    for plot_spec in RAW_PLOT_SPECS:
        raw_frame = raw_frames_by_group[plot_spec.ancestry_group]
        output_path = output_dir / RAW_PLOT_FILENAME_TEMPLATE.format(
            ancestry_group=plot_spec.ancestry_group,
        )
        write_ancestry_boxplot(
            ancestry_group=plot_spec.ancestry_group,
            raw_frame=raw_frame,
            output_path=output_path,
            y_limits=global_y_limits,
        )
        logger.info(
            "Wrote raw ancestry boxplot to %s rows=%d y_limits=(%s,%s)",
            output_path,
            len(raw_frame),
            format_float(global_y_limits[0]),
            format_float(global_y_limits[1]),
        )


def load_raw_plot_frame(raw_data_dir: Path, source_files: tuple[str, ...]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for source_name in source_files:
        source_path = raw_data_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw source pickle for plot: {source_path}")
        frame = pd.read_pickle(source_path).reset_index(drop=True)
        missing_ancestry_columns = [column for column in ANCESTRY_COLUMNS if column not in frame.columns]
        if missing_ancestry_columns:
            raise ValueError(
                f"Source pickle {source_path} is missing ancestry columns: {missing_ancestry_columns}"
            )
        parts.append(frame.loc[:, ANCESTRY_COLUMNS].copy())
    return pd.concat(parts, ignore_index=True)


def ancestry_y_limits(raw_frame: pd.DataFrame) -> tuple[float, float]:
    ancestry_values = raw_frame.loc[:, ANCESTRY_COLUMNS].to_numpy(dtype=float, copy=True)
    return float(np.min(ancestry_values)), float(np.max(ancestry_values))


def write_ancestry_boxplot(
    *,
    ancestry_group: str,
    raw_frame: pd.DataFrame,
    output_path: Path,
    y_limits: tuple[float, float],
) -> None:
    figure, axes = plt.subplots(2, 8, figsize=(18, 8))
    flattened_axes = axes.flatten()
    for axis, column_name in zip(flattened_axes, ANCESTRY_COLUMNS):
        values = raw_frame[column_name].to_numpy(dtype=float, copy=True)
        axis.boxplot(
            values,
            vert=True,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#4c78a8", "edgecolor": "#2f3e4d", "linewidth": 1.0},
            medianprops={"color": "#f58518", "linewidth": 1.5},
            whiskerprops={"color": "#2f3e4d", "linewidth": 1.0},
            capprops={"color": "#2f3e4d", "linewidth": 1.0},
        )
        axis.set_title(column_name, fontsize=10, pad=8)
        axis.set_xticks([])
        axis.set_ylim(y_limits)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.ticklabel_format(axis="y", style="plain")
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for row_start in (0, 8):
        flattened_axes[row_start].set_ylabel("Value", fontsize=10)

    figure.suptitle(ancestry_group, fontsize=14, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()