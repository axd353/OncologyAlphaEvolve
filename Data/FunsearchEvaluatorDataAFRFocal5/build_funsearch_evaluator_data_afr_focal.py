from __future__ import annotations

"""Build AFR focal evaluator datasets with expanded European test cohorts.

This builder starts from the AFRFocal3 outputs, reuses the exact same subject
membership and row order for train and heldout, reuses the same base test rows,
then appends additional unused European samples to each test split before
refitting ancestry standardization from the underlying raw OncoArray rows.

For each condition:
- train and heldout match AFRFocal3 exactly in subject membership and row order
- test starts from the AFRFocal3 rows and appends 160 controls plus 160 cases
  drawn from unused European raw rows for that same condition
- ancestry coordinates PC1-PC16 are recomputed from raw rows using one pooled
  transform over that condition's train, test, and heldout rows, including the
  appended European test rows

Outputs:
- no_covariates_{heldout,test,train}.pkl
- with_covariates_{heldout,test,train}.pkl
- output_row_tracking.pkl
- transformations.txt
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

from build_funsearch_evaluator_data import ANCESTRY_COLUMNS
from build_funsearch_evaluator_data import StandardizationTransform
from build_funsearch_evaluator_data import compute_standardization_transform
from build_funsearch_evaluator_data import configure_logger
from build_funsearch_evaluator_data import format_float
from build_funsearch_evaluator_data import format_source_counts
from build_funsearch_evaluator_data import standardize_ancestry_columns


DEFAULT_REFERENCE_DIR = DATA_DIR / "FunsearchEvaluatorDataAFRFocal3"
DEFAULT_RAW_DATA_DIR = DATA_DIR / "RawDataOncoArray"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILENAME = "build_funsearch_evaluator_data.log"
DEFAULT_TRANSFORMATIONS_FILENAME = "transformations.txt"
DEFAULT_TRACKING_FILENAME = "output_row_tracking.pkl"
OUTPUT_SPLITS = ("heldout", "test", "train")
EXTRA_EUROPEAN_CONTROLS = 160
EXTRA_EUROPEAN_CASES = 160


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


CONDITIONS = (
    ConditionSpec(name="no_covariates"),
    ConditionSpec(name="with_covariates"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AFR focal evaluator datasets in Data/FunsearchEvaluatorDataAFRFocal5/ "
            "using AFRFocal3 subject membership plus additional unused European test rows."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing the existing AFRFocal3 outputs and output_row_tracking.pkl.",
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
        help="Directory where AFRFocal5 outputs are written.",
    )
    parser.add_argument(
        "--extra-european-controls",
        type=int,
        default=EXTRA_EUROPEAN_CONTROLS,
        help="Number of unused European controls to append to each test split.",
    )
    parser.add_argument(
        "--extra-european-cases",
        type=int,
        default=EXTRA_EUROPEAN_CASES,
        help="Number of unused European cases to append to each test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extra_european_controls < 0:
        raise ValueError("--extra-european-controls must be non-negative.")
    if args.extra_european_cases < 0:
        raise ValueError("--extra-european-cases must be non-negative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / DEFAULT_LOG_FILENAME
    transformations_path = output_dir / DEFAULT_TRANSFORMATIONS_FILENAME
    tracking_path = output_dir / DEFAULT_TRACKING_FILENAME
    logger = configure_logger(log_path)

    reference_tracking_path = args.reference_dir / DEFAULT_TRACKING_FILENAME
    if not reference_tracking_path.exists():
        raise FileNotFoundError(f"Missing AFRFocal3 tracking file: {reference_tracking_path}")
    reference_tracking = pd.read_pickle(reference_tracking_path)

    logger.info(
        "Starting AFRFocal5 build reference_dir=%s raw_data_dir=%s output_dir=%s extra_european_controls=%d extra_european_cases=%d",
        args.reference_dir,
        args.raw_data_dir,
        output_dir,
        args.extra_european_controls,
        args.extra_european_cases,
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
            extra_european_controls=int(args.extra_european_controls),
            extra_european_cases=int(args.extra_european_cases),
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
    logger.info("Finished AFRFocal5 build")


def build_condition_datasets(
    *,
    spec: ConditionSpec,
    reference_dir: Path,
    raw_data_dir: Path,
    output_dir: Path,
    reference_tracking: pd.DataFrame,
    extra_european_controls: int,
    extra_european_cases: int,
    logger: logging.Logger,
) -> ConditionBuildResult:
    raw_frame_cache: dict[str, pd.DataFrame] = {}
    condition_tracking = reference_tracking.loc[
        reference_tracking["output_pickle_name"].str.startswith(f"{spec.name}_")
    ].copy()
    used_source_rows = set(
        zip(
            condition_tracking["source_pickle_name"].astype(str),
            condition_tracking["source_row_number"].astype(int),
        )
    )

    extra_test_tracking = build_extra_test_tracking(
        spec=spec,
        raw_data_dir=raw_data_dir,
        raw_frame_cache=raw_frame_cache,
        used_source_rows=used_source_rows,
        extra_european_controls=extra_european_controls,
        extra_european_cases=extra_european_cases,
    )
    if not extra_test_tracking.empty:
        extra_source_counts = extra_test_tracking["source_pickle_name"].value_counts(sort=False).to_dict()
        extra_label_counts = extra_test_tracking["phenotype_label"].value_counts(sort=False).to_dict()
        logger.info(
            "Condition %s appended extra European test rows count=%d labels=%s sources=%s",
            spec.name,
            len(extra_test_tracking),
            format_source_counts({str(key): int(value) for key, value in extra_label_counts.items()}),
            format_source_counts({str(key): int(value) for key, value in extra_source_counts.items()}),
        )

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
            extra_tracking=extra_test_tracking if split_name == "test" else None,
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

        validate_tracking_uniqueness(prepared_output.tracking, prepared_output.output_name)
        tracking_parts.append(prepared_output.tracking)

    return ConditionBuildResult(
        transform=transform,
        tracking=pd.concat(tracking_parts, ignore_index=True),
    )


def build_extra_test_tracking(
    *,
    spec: ConditionSpec,
    raw_data_dir: Path,
    raw_frame_cache: dict[str, pd.DataFrame],
    used_source_rows: set[tuple[str, int]],
    extra_european_controls: int,
    extra_european_cases: int,
) -> pd.DataFrame:
    source_names = european_source_names(spec)
    candidate_records: list[dict[str, object]] = []
    selection_order = 0

    for source_name in source_names:
        raw_frame = load_raw_frame(
            source_name=source_name,
            raw_data_dir=raw_data_dir,
            raw_frame_cache=raw_frame_cache,
        )
        for source_row_number, phenotype in enumerate(raw_frame["phenotype"].to_numpy(copy=False)):
            if (source_name, source_row_number) in used_source_rows:
                continue
            phenotype_int = int(phenotype)
            if phenotype_int not in (0, 1):
                raise ValueError(
                    f"Unexpected phenotype value in {source_name} row {source_row_number}: {phenotype!r}"
                )
            candidate_records.append(
                {
                    "source_pickle_name": source_name,
                    "source_row_number": int(source_row_number),
                    "phenotype": phenotype_int,
                    "selection_order": selection_order,
                }
            )
            selection_order += 1

    candidates = pd.DataFrame(candidate_records)
    if candidates.empty:
        raise ValueError(f"No unused European rows available for {spec.name}.")

    selected_controls = candidates.loc[candidates["phenotype"] == 0].head(extra_european_controls)
    selected_cases = candidates.loc[candidates["phenotype"] == 1].head(extra_european_cases)
    if len(selected_controls) != extra_european_controls:
        raise ValueError(
            f"Needed {extra_european_controls} unused European controls for {spec.name}, "
            f"found {len(selected_controls)}."
        )
    if len(selected_cases) != extra_european_cases:
        raise ValueError(
            f"Needed {extra_european_cases} unused European cases for {spec.name}, "
            f"found {len(selected_cases)}."
        )

    selected = (
        pd.concat([selected_controls, selected_cases], ignore_index=True)
        .sort_values("selection_order")
        .reset_index(drop=True)
    )
    selected.loc[:, "phenotype_label"] = selected["phenotype"].map({0: "control", 1: "case"})
    return selected.loc[:, ["source_pickle_name", "source_row_number", "phenotype_label"]]


def prepare_output(
    *,
    spec: ConditionSpec,
    split_name: str,
    reference_dir: Path,
    raw_data_dir: Path,
    reference_tracking: pd.DataFrame,
    raw_frame_cache: dict[str, pd.DataFrame],
    extra_tracking: pd.DataFrame | None,
) -> PreparedOutput:
    output_name = f"{spec.name}_{split_name}.pkl"
    reference_path = reference_dir / output_name
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing AFRFocal3 dataset: {reference_path}")

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

    normalized_tracking = tracking.copy()
    normalized_tracking.loc[:, "source_pickle_path"] = normalized_tracking["source_pickle_name"].map(
        lambda source_name: str((raw_data_dir / str(source_name)).resolve())
    )
    if extra_tracking is not None and not extra_tracking.empty:
        extra_tracking_frame = extra_tracking.copy()
        extra_tracking_frame.loc[:, "output_pickle_name"] = output_name
        extra_tracking_frame.loc[:, "output_row_number"] = np.arange(
            len(normalized_tracking),
            len(normalized_tracking) + len(extra_tracking_frame),
            dtype=int,
        )
        extra_tracking_frame.loc[:, "source_pickle_path"] = extra_tracking_frame["source_pickle_name"].map(
            lambda source_name: str((raw_data_dir / str(source_name)).resolve())
        )
        extra_tracking_frame = extra_tracking_frame.loc[
            :,
            [
                "output_pickle_name",
                "output_row_number",
                "source_pickle_name",
                "source_pickle_path",
                "source_row_number",
            ],
        ]
        normalized_tracking = pd.concat(
            [normalized_tracking, extra_tracking_frame],
            ignore_index=True,
        )

    raw_dataset = assemble_raw_dataset(
        tracking=normalized_tracking,
        raw_data_dir=raw_data_dir,
        raw_frame_cache=raw_frame_cache,
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
        raw_frame = load_raw_frame(
            source_name=source_name,
            raw_data_dir=raw_data_dir,
            raw_frame_cache=raw_frame_cache,
        )
        source_row_number = int(row.source_row_number)
        if source_row_number < 0 or source_row_number >= len(raw_frame):
            raise ValueError(
                f"Source row out of bounds for {source_name}: {source_row_number} not in [0, {len(raw_frame)})"
            )
        row_parts.append(raw_frame.iloc[[source_row_number]].copy())

    if not row_parts:
        return pd.DataFrame(columns=list(ANCESTRY_COLUMNS))
    return pd.concat(row_parts, ignore_index=True)


def load_raw_frame(
    *,
    source_name: str,
    raw_data_dir: Path,
    raw_frame_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if source_name not in raw_frame_cache:
        source_path = raw_data_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw source pickle: {source_path}")
        raw_frame_cache[source_name] = pd.read_pickle(source_path).reset_index(drop=True)
    return raw_frame_cache[source_name]


def validate_reference_alignment(
    *,
    reference_dataset: pd.DataFrame,
    raw_dataset: pd.DataFrame,
    standardized_dataset: pd.DataFrame,
    output_name: str,
) -> None:
    if list(reference_dataset.columns) != list(raw_dataset.columns):
        raise ValueError(f"Column mismatch between AFRFocal3 and raw reconstruction for {output_name}.")
    if list(reference_dataset.columns) != list(standardized_dataset.columns):
        raise ValueError(f"Column mismatch after standardization for {output_name}.")
    if len(raw_dataset) < len(reference_dataset):
        raise ValueError(
            f"Raw reconstruction for {output_name} is missing rows: {len(raw_dataset)} < {len(reference_dataset)}."
        )

    reference_row_count = len(reference_dataset)
    raw_prefix = raw_dataset.iloc[:reference_row_count].reset_index(drop=True)
    standardized_prefix = standardized_dataset.iloc[:reference_row_count].reset_index(drop=True)
    non_ancestry_columns = [column for column in reference_dataset.columns if column not in ANCESTRY_COLUMNS]
    if not reference_dataset.loc[:, non_ancestry_columns].equals(raw_prefix.loc[:, non_ancestry_columns]):
        raise ValueError(
            f"Non-ancestry column mismatch while reconstructing {output_name} from raw rows."
        )
    if not reference_dataset.loc[:, non_ancestry_columns].equals(
        standardized_prefix.loc[:, non_ancestry_columns]
    ):
        raise ValueError(
            f"Non-ancestry columns changed unexpectedly after standardization for {output_name}."
        )


def validate_tracking_uniqueness(tracking: pd.DataFrame, output_name: str) -> None:
    duplicate_mask = tracking.duplicated(subset=["source_pickle_name", "source_row_number"], keep=False)
    if duplicate_mask.any():
        duplicates = tracking.loc[duplicate_mask, ["source_pickle_name", "source_row_number"]]
        raise ValueError(
            f"Duplicate source rows detected in {output_name}: {duplicates.drop_duplicates().to_dict(orient='records')}"
        )


def european_source_names(spec: ConditionSpec) -> tuple[str, str]:
    suffix = "" if spec.name == "no_covariates" else "_add_covs"
    return (
        f"train_European{suffix}.pkl",
        f"test_European{suffix}.pkl",
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
                "a_star=" + ",".join(format_float(value) for value in transform.center),
                f"r={format_float(transform.radius)}",
                "",
            ]
        )
    transformations_path.write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()