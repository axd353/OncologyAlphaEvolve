from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from Data.build_funsearch_evaluator_data import (
    ANCESTRY_COLUMNS,
    StandardizationTransform,
    compute_standardization_transform,
    standardize_ancestry_columns,
)
from GenomicsHelpers.oracle_data_adapter import DOSAGE_COLUMN_PREFIX


DEFAULT_OUTPUT_DIR = Path("../../auxilalry_data/CA_107_2")
DEFAULT_TRAIN_PICKLE_NAME = "aou_train.pkl"
DEFAULT_CALIBRATION_PICKLE_NAME = "aou_calibration.pkl"
DEFAULT_HELDOUT_PICKLE_NAME = "aou_heldout.pkl"
DEFAULT_TRACKING_PICKLE_NAME = "output_row_tracking.pkl"
DEFAULT_TRANSFORMATIONS_JSON_NAME = "aou_transformations.json"
DEFAULT_IMPUTATION_JSON_NAME = "dosage_imputation_summary.json"
DEFAULT_COLUMN_MAP_JSON_NAME = "aou_column_mapping.json"
DEFAULT_LOG_NAME = "prepare_aou_evaluator_data.log"

DEFAULT_SUPPORTED_ANCESTRY_GROUPS = ("EUR", "AFR", "AMR", "EAS", "SAS", "MID")

ANCESTRY_LABEL_MAP = {
    "eur": "EUR",
    "european": "EUR",
    "afr": "AFR",
    "african": "AFR",
    "amr": "AMR",
    "admixed american": "AMR",
    "eas": "EAS",
    "east asian": "EAS",
    "sas": "SAS",
    "south asian": "SAS",
    "mid": "MID",
    "middle eastern": "MID",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare All of Us train/calibration/heldout pickles for "
            "PostProcesingData.evaluate_priofunction."
        )
    )
    parser.add_argument("--tr-pickle", type=Path, required=True)
    parser.add_argument("--te-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-ancestries", type=str, default="eas,sas")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--ancestry-column", type=str, default="ancestry_pred_x")
    parser.add_argument("--sample-id-column", type=str, default="sample_id")
    parser.add_argument("--pc-prefix", type=str, default="anc_pc")
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="random",
        choices=("random", "stratify_label", "stratify_label_ancestry"),
        help=(
            "Default is random. Use stratify_label or stratify_label_ancestry "
            "only when you want a stratified train/calibration split."
        ),
    )
    parser.add_argument(
        "--heldout-single-class-policy",
        type=str,
        default="raise",
        choices=("raise", "drop_ancestry"),
        help=(
            "If a requested heldout ancestry has only one label class, either "
            "raise or drop that ancestry before writing aou_heldout.pkl."
        ),
    )
    parser.add_argument(
        "--dosage-columns",
        type=str,
        default="",
        help=(
            "Optional comma-separated dosage columns. If blank, dosage columns "
            "are inferred from tr_df columns that start with chr or dosage__."
        ),
    )
    parser.add_argument(
        "--no-impute-dosages",
        action="store_true",
        help="Disable dosage mean imputation before writing evaluator pickles.",
    )
    return parser.parse_args()


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("prepare_aou_evaluator_data")
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


def parse_csv_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_ancestry_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return ANCESTRY_LABEL_MAP.get(text.lower(), text.upper())


def normalize_ancestry_list(values: Iterable[str]) -> list[str]:
    normalized = []
    for value in values:
        ancestry = normalize_ancestry_label(value)
        if ancestry is not None:
            normalized.append(ancestry)
    return normalized


def resolve_pc_source_columns(
    frame: pd.DataFrame,
    *,
    pc_prefix: str,
    dataset_name: str,
) -> list[str]:
    prefixed_columns = [f"{pc_prefix}{index}" for index in range(1, 17)]
    if all(column in frame.columns for column in prefixed_columns):
        return prefixed_columns

    canonical_columns = list(ANCESTRY_COLUMNS)
    if all(column in frame.columns for column in canonical_columns):
        return canonical_columns

    missing_prefixed = [column for column in prefixed_columns if column not in frame.columns]
    missing_canonical = [column for column in canonical_columns if column not in frame.columns]
    raise ValueError(
        f"{dataset_name} is missing ancestry coordinate columns. "
        f"Missing {pc_prefix} columns={missing_prefixed}; missing PC columns={missing_canonical}."
    )


def infer_dosage_columns(frame: pd.DataFrame) -> list[str]:
    dosage_columns = []
    for column in frame.columns:
        column_text = str(column)
        lower_column = column_text.lower()
        if column_text.startswith(DOSAGE_COLUMN_PREFIX) or lower_column.startswith("chr"):
            dosage_columns.append(column_text)
    return dosage_columns


def make_safe_dosage_column_name(raw_column: str, used_names: set[str]) -> str:
    logical_name = raw_column
    if logical_name.startswith(DOSAGE_COLUMN_PREFIX):
        logical_name = logical_name[len(DOSAGE_COLUMN_PREFIX) :]

    safe_logical_name = re.sub(r"[^0-9A-Za-z_]+", "_", logical_name).strip("_")
    if not safe_logical_name:
        safe_logical_name = "variant"
    if safe_logical_name[0].isdigit():
        safe_logical_name = f"v_{safe_logical_name}"

    base_name = f"{DOSAGE_COLUMN_PREFIX}{safe_logical_name}"
    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1

    used_names.add(candidate)
    return candidate


def build_dosage_column_map(dosage_columns: Sequence[str]) -> dict[str, str]:
    used_names: set[str] = set()
    return {
        str(column): make_safe_dosage_column_name(str(column), used_names)
        for column in dosage_columns
    }


def prepare_common_layout(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    label_column: str,
    ancestry_column: str,
    sample_id_column: str,
    pc_prefix: str,
    dosage_column_map: dict[str, str],
    supported_ancestry_groups: Sequence[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_required = [
        column
        for column in (label_column, ancestry_column)
        if column not in frame.columns
    ]
    if missing_required:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_required}")

    pc_source_columns = resolve_pc_source_columns(
        frame,
        pc_prefix=pc_prefix,
        dataset_name=dataset_name,
    )

    prepared = pd.DataFrame(index=frame.index)
    prepared["_aou_source_row_number"] = np.arange(len(frame), dtype=int)

    if sample_id_column in frame.columns:
        prepared["sample_id"] = frame[sample_id_column].astype(str)
    else:
        prepared["sample_id"] = [
            f"{dataset_name}_{row_number}" for row_number in prepared["_aou_source_row_number"]
        ]

    prepared["phenotype"] = pd.to_numeric(frame[label_column], errors="coerce")
    prepared["aou_ancestry_label"] = frame[ancestry_column].map(normalize_ancestry_label)

    for source_column, output_column in zip(pc_source_columns, ANCESTRY_COLUMNS):
        prepared[output_column] = pd.to_numeric(frame[source_column], errors="coerce")

    for source_column, output_column in dosage_column_map.items():
        if source_column in frame.columns:
            prepared[output_column] = pd.to_numeric(frame[source_column], errors="coerce")
        else:
            prepared[output_column] = np.nan

    required_non_missing = ["phenotype", "aou_ancestry_label", *ANCESTRY_COLUMNS]
    before_rows = len(prepared)
    prepared = prepared.dropna(subset=required_non_missing).copy()
    after_rows = len(prepared)

    drop_summary = {
        "input_rows": int(before_rows),
        "dropped_missing_label_ancestry_or_pc": int(before_rows - after_rows),
        "output_rows": int(after_rows),
    }

    prepared["phenotype"] = prepared["phenotype"].astype(int)

    bad_labels = sorted(set(prepared["phenotype"].dropna().unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"{dataset_name} has non-binary labels after cleaning: {bad_labels}")

    supported_set = set(supported_ancestry_groups)
    observed_ancestries = set(prepared["aou_ancestry_label"].dropna().astype(str).unique())
    unsupported = sorted(observed_ancestries - supported_set)
    if unsupported:
        raise ValueError(
            f"{dataset_name} has ancestry labels not in supported_ancestry_groups: {unsupported}. "
            f"Supported={sorted(supported_set)}."
        )

    logger.info(
        "%s cleaned rows input=%d output=%d dropped=%d ancestry_counts=%s label_counts=%s",
        dataset_name,
        before_rows,
        after_rows,
        before_rows - after_rows,
        prepared["aou_ancestry_label"].value_counts().to_dict(),
        prepared["phenotype"].value_counts().to_dict(),
    )

    return prepared.reset_index(drop=True), drop_summary


def random_train_calibration_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    row_count = len(frame)
    calibration_count = int(round(calibration_fraction * row_count))
    calibration_count = min(max(calibration_count, 1), row_count - 1)

    calibration_positions = set(
        rng.choice(np.arange(row_count), size=calibration_count, replace=False).astype(int).tolist()
    )
    train_positions = [index for index in range(row_count) if index not in calibration_positions]
    calibration_positions_sorted = sorted(calibration_positions)

    split_info = {
        "requested_strategy": "random",
        "used_strategy": "random",
        "calibration_fraction": float(calibration_fraction),
        "train_rows": int(len(train_positions)),
        "calibration_rows": int(len(calibration_positions_sorted)),
    }

    return (
        frame.iloc[train_positions].reset_index(drop=True).copy(),
        frame.iloc[calibration_positions_sorted].reset_index(drop=True).copy(),
        split_info,
    )


def stratified_train_calibration_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    random_seed: int,
    strategy: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if strategy == "stratify_label":
        split_key = frame["phenotype"].astype(str)
    elif strategy == "stratify_label_ancestry":
        split_key = frame["aou_ancestry_label"].astype(str) + "__" + frame["phenotype"].astype(str)
    else:
        raise ValueError(f"Unknown stratified split strategy: {strategy}")

    group_sizes = split_key.value_counts()
    if (group_sizes < 2).any():
        logger.warning(
            "Cannot use %s because at least one stratum has fewer than 2 rows. "
            "Falling back to random split. Strata=%s",
            strategy,
            group_sizes.to_dict(),
        )
        train_df, calibration_df, split_info = random_train_calibration_split(
            frame,
            calibration_fraction=calibration_fraction,
            random_seed=random_seed,
        )
        split_info["requested_strategy"] = strategy
        split_info["fallback_reason"] = "at_least_one_stratum_has_fewer_than_2_rows"
        return train_df, calibration_df, split_info

    rng = np.random.default_rng(random_seed)
    calibration_positions: set[int] = set()

    for _, group_positions in split_key.groupby(split_key).groups.items():
        positions = np.asarray(list(group_positions), dtype=int)
        group_count = len(positions)
        group_calibration_count = int(round(calibration_fraction * group_count))
        group_calibration_count = min(max(group_calibration_count, 1), group_count - 1)
        chosen = rng.choice(positions, size=group_calibration_count, replace=False)
        calibration_positions.update(chosen.astype(int).tolist())

    train_positions = [index for index in range(len(frame)) if index not in calibration_positions]
    calibration_positions_sorted = sorted(calibration_positions)

    train_df = frame.iloc[train_positions].reset_index(drop=True).copy()
    calibration_df = frame.iloc[calibration_positions_sorted].reset_index(drop=True).copy()

    split_info = {
        "requested_strategy": strategy,
        "used_strategy": strategy,
        "calibration_fraction": float(calibration_fraction),
        "train_rows": int(len(train_df)),
        "calibration_rows": int(len(calibration_df)),
        "strata": group_sizes.to_dict(),
    }

    return train_df, calibration_df, split_info


def train_calibration_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    random_seed: int,
    strategy: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    if len(frame) < 2:
        raise ValueError("Need at least 2 training rows to split train/calibration.")

    if strategy == "random":
        return random_train_calibration_split(
            frame,
            calibration_fraction=calibration_fraction,
            random_seed=random_seed,
        )

    return stratified_train_calibration_split(
        frame,
        calibration_fraction=calibration_fraction,
        random_seed=random_seed,
        strategy=strategy,
        logger=logger,
    )


def assert_two_label_classes(frame: pd.DataFrame, *, dataset_name: str) -> None:
    label_count = frame["phenotype"].nunique(dropna=True)
    if label_count < 2:
        raise ValueError(
            f"{dataset_name} has fewer than two phenotype classes. "
            f"Label counts={frame['phenotype'].value_counts(dropna=False).to_dict()}."
        )


def apply_heldout_single_class_policy(
    heldout_df: pd.DataFrame,
    *,
    policy: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    bad_ancestries = []
    for ancestry_group, group_df in heldout_df.groupby("aou_ancestry_label"):
        if group_df["phenotype"].nunique(dropna=True) < 2:
            bad_ancestries.append(str(ancestry_group))

    if not bad_ancestries:
        return heldout_df

    if policy == "raise":
        raise ValueError(
            "The heldout set has requested ancestry groups with fewer than two "
            f"phenotype classes: {bad_ancestries}. "
            "The evaluator computes per-ancestry ROC AUC, so these groups will fail. "
            "Use heldout_single_class_policy='drop_ancestry' only if you want to drop them."
        )

    if policy == "drop_ancestry":
        logger.warning(
            "Dropping heldout ancestry groups with fewer than two phenotype classes: %s",
            bad_ancestries,
        )
        return (
            heldout_df.loc[~heldout_df["aou_ancestry_label"].isin(bad_ancestries)]
            .reset_index(drop=True)
            .copy()
        )

    raise ValueError(f"Unknown heldout_single_class_policy: {policy}")


def finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def impute_dosage_frames(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    *,
    dosage_columns: Sequence[str],
    enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not enabled:
        return train_df, calibration_df, heldout_df, {"enabled": False}

    train_df = train_df.copy()
    calibration_df = calibration_df.copy()
    heldout_df = heldout_df.copy()

    nonheldout_df = pd.concat([train_df, calibration_df], ignore_index=True)

    nonheldout_means = nonheldout_df.loc[:, dosage_columns].mean(skipna=True)
    heldout_means = heldout_df.loc[:, dosage_columns].mean(skipna=True)

    summary: dict[str, Any] = {
        "enabled": True,
        "policy": {
            "train_and_calibration": "fill from combined train+calibration mean",
            "heldout": "fill from heldout mean",
            "fallback": "if a split mean is NaN, use the other split mean; if still NaN, use 0.0",
        },
        "columns": {},
    }

    for column in dosage_columns:
        nonheldout_mean = finite_or_none(nonheldout_means[column])
        heldout_mean = finite_or_none(heldout_means[column])

        nonheldout_fill = nonheldout_mean
        if nonheldout_fill is None:
            nonheldout_fill = heldout_mean
        if nonheldout_fill is None:
            nonheldout_fill = 0.0

        heldout_fill = heldout_mean
        if heldout_fill is None:
            heldout_fill = nonheldout_mean
        if heldout_fill is None:
            heldout_fill = 0.0

        train_missing = int(train_df[column].isna().sum())
        calibration_missing = int(calibration_df[column].isna().sum())
        heldout_missing = int(heldout_df[column].isna().sum())

        train_df[column] = train_df[column].fillna(nonheldout_fill).astype("float64")
        calibration_df[column] = calibration_df[column].fillna(nonheldout_fill).astype("float64")
        heldout_df[column] = heldout_df[column].fillna(heldout_fill).astype("float64")

        summary["columns"][column] = {
            "train_missing_filled": train_missing,
            "calibration_missing_filled": calibration_missing,
            "heldout_missing_filled": heldout_missing,
            "nonheldout_mean": nonheldout_mean,
            "heldout_mean": heldout_mean,
            "train_calibration_fill_value": float(nonheldout_fill),
            "heldout_fill_value": float(heldout_fill),
        }

    return train_df, calibration_df, heldout_df, summary


def build_tracking_frame(
    frame: pd.DataFrame,
    *,
    output_pickle_name: str,
    source_prefix: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "output_pickle_name": output_pickle_name,
            "output_row_number": np.arange(len(frame), dtype=int),
            "source_pickle_name": [
                f"{source_prefix}_{ancestry_group}.pkl"
                for ancestry_group in frame["aou_ancestry_label"].astype(str).tolist()
            ],
            "source_pickle_path": [f"in_memory:{source_prefix}"] * len(frame),
            "source_row_number": frame["_aou_source_row_number"].astype(int).to_numpy(copy=False),
        }
    )


def output_frame_for_evaluator(
    frame: pd.DataFrame,
    *,
    dosage_columns: Sequence[str],
) -> pd.DataFrame:
    output_columns = [
        "sample_id",
        "aou_ancestry_label",
        "phenotype",
        *ANCESTRY_COLUMNS,
        *dosage_columns,
    ]
    return frame.loc[:, output_columns].reset_index(drop=True).copy()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_aou_evaluator_data(
    *,
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    eval_ancestries: Sequence[str] = ("eas", "sas"),
    label_column: str = "label",
    ancestry_column: str = "ancestry_pred_x",
    sample_id_column: str = "sample_id",
    pc_prefix: str = "anc_pc",
    dosage_columns: Sequence[str] | None = None,
    calibration_fraction: float = 0.20,
    random_seed: int = 7,
    split_strategy: str = "random",
    heldout_single_class_policy: str = "raise",
    impute_dosages: bool = True,
    supported_ancestry_groups: Sequence[str] = DEFAULT_SUPPORTED_ANCESTRY_GROUPS,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(output_dir / DEFAULT_LOG_NAME)
    logger.info("Starting AOU evaluator-data preparation output_dir=%s", output_dir)

    supported_ancestry_groups = tuple(str(group).upper() for group in supported_ancestry_groups)
    eval_ancestries_normalized = normalize_ancestry_list(eval_ancestries)

    if not eval_ancestries_normalized:
        raise ValueError("eval_ancestries must contain at least one ancestry label.")

    unknown_eval_ancestries = sorted(set(eval_ancestries_normalized) - set(supported_ancestry_groups))
    if unknown_eval_ancestries:
        raise ValueError(
            f"eval_ancestries contains unsupported groups: {unknown_eval_ancestries}. "
            f"Supported={list(supported_ancestry_groups)}."
        )

    if dosage_columns is None:
        dosage_columns = infer_dosage_columns(tr_df)
    dosage_columns = [str(column) for column in dosage_columns]

    if not dosage_columns:
        raise ValueError(
            "No dosage columns were found. Pass dosage_columns explicitly or make sure "
            "variant columns start with chr or dosage__."
        )

    dosage_column_map = build_dosage_column_map(dosage_columns)
    evaluator_dosage_columns = list(dosage_column_map.values())

    tr_clean, tr_drop_summary = prepare_common_layout(
        tr_df,
        dataset_name="tr_df",
        label_column=label_column,
        ancestry_column=ancestry_column,
        sample_id_column=sample_id_column,
        pc_prefix=pc_prefix,
        dosage_column_map=dosage_column_map,
        supported_ancestry_groups=supported_ancestry_groups,
        logger=logger,
    )
    te_clean, te_drop_summary = prepare_common_layout(
        te_df,
        dataset_name="te_df",
        label_column=label_column,
        ancestry_column=ancestry_column,
        sample_id_column=sample_id_column,
        pc_prefix=pc_prefix,
        dosage_column_map=dosage_column_map,
        supported_ancestry_groups=supported_ancestry_groups,
        logger=logger,
    )

    heldout_clean = (
        te_clean.loc[te_clean["aou_ancestry_label"].isin(eval_ancestries_normalized)]
        .reset_index(drop=True)
        .copy()
    )

    if heldout_clean.empty:
        raise ValueError(
            f"te_df has no rows after filtering to eval_ancestries={eval_ancestries_normalized}."
        )

    heldout_clean = apply_heldout_single_class_policy(
        heldout_clean,
        policy=heldout_single_class_policy,
        logger=logger,
    )

    if heldout_clean.empty:
        raise ValueError("aou_heldout would be empty after heldout ancestry filtering.")

    assert_two_label_classes(tr_clean, dataset_name="tr_df after cleaning")
    assert_two_label_classes(heldout_clean, dataset_name="aou_heldout after filtering")

    standardization_transform: StandardizationTransform = compute_standardization_transform(
        [tr_clean, heldout_clean]
    )

    tr_standardized = standardize_ancestry_columns(tr_clean, standardization_transform)
    heldout_standardized = standardize_ancestry_columns(heldout_clean, standardization_transform)

    train_df, calibration_df, split_info = train_calibration_split(
        tr_standardized,
        calibration_fraction=calibration_fraction,
        random_seed=random_seed,
        strategy=split_strategy,
        logger=logger,
    )

    assert_two_label_classes(train_df, dataset_name="aou_train")
    assert_two_label_classes(calibration_df, dataset_name="aou_calibration")
    assert_two_label_classes(heldout_standardized, dataset_name="aou_heldout")

    train_df, calibration_df, heldout_standardized, imputation_summary = impute_dosage_frames(
        train_df,
        calibration_df,
        heldout_standardized,
        dosage_columns=evaluator_dosage_columns,
        enabled=impute_dosages,
    )

    train_output = output_frame_for_evaluator(
        train_df,
        dosage_columns=evaluator_dosage_columns,
    )
    calibration_output = output_frame_for_evaluator(
        calibration_df,
        dosage_columns=evaluator_dosage_columns,
    )
    heldout_output = output_frame_for_evaluator(
        heldout_standardized,
        dosage_columns=evaluator_dosage_columns,
    )

    train_pickle_path = output_dir / DEFAULT_TRAIN_PICKLE_NAME
    calibration_pickle_path = output_dir / DEFAULT_CALIBRATION_PICKLE_NAME
    heldout_pickle_path = output_dir / DEFAULT_HELDOUT_PICKLE_NAME
    tracking_pickle_path = output_dir / DEFAULT_TRACKING_PICKLE_NAME
    transformations_json_path = output_dir / DEFAULT_TRANSFORMATIONS_JSON_NAME
    imputation_json_path = output_dir / DEFAULT_IMPUTATION_JSON_NAME
    column_map_json_path = output_dir / DEFAULT_COLUMN_MAP_JSON_NAME

    train_output.to_pickle(train_pickle_path)
    calibration_output.to_pickle(calibration_pickle_path)
    heldout_output.to_pickle(heldout_pickle_path)

    tracking_frame = pd.concat(
        [
            build_tracking_frame(
                train_df,
                output_pickle_name=DEFAULT_TRAIN_PICKLE_NAME,
                source_prefix="aou_train",
            ),
            build_tracking_frame(
                calibration_df,
                output_pickle_name=DEFAULT_CALIBRATION_PICKLE_NAME,
                source_prefix="aou_calibration",
            ),
            build_tracking_frame(
                heldout_standardized,
                output_pickle_name=DEFAULT_HELDOUT_PICKLE_NAME,
                source_prefix="aou_heldout",
            ),
        ],
        ignore_index=True,
    )
    tracking_frame.to_pickle(tracking_pickle_path)

    transformations_payload = {
        "standardization": {
            "method": "center from tr_df plus ancestry-filtered te_df; radius covers 95 percent of Euclidean distances",
            "center": [float(value) for value in standardization_transform.center],
            "radius": float(standardization_transform.radius),
            "sample_count": int(standardization_transform.sample_count),
            "fit_rows": {
                "tr_df_all_ancestries": int(len(tr_clean)),
                "te_df_eval_ancestries_only": int(len(heldout_clean)),
            },
        },
        "eval_ancestries": eval_ancestries_normalized,
        "supported_ancestry_groups": list(supported_ancestry_groups),
        "drop_summary": {
            "tr_df": tr_drop_summary,
            "te_df": te_drop_summary,
        },
        "split": split_info,
        "output_rows": {
            DEFAULT_TRAIN_PICKLE_NAME: int(len(train_output)),
            DEFAULT_CALIBRATION_PICKLE_NAME: int(len(calibration_output)),
            DEFAULT_HELDOUT_PICKLE_NAME: int(len(heldout_output)),
        },
        "label_counts": {
            "aou_train": train_output["phenotype"].value_counts().to_dict(),
            "aou_calibration": calibration_output["phenotype"].value_counts().to_dict(),
            "aou_heldout": heldout_output["phenotype"].value_counts().to_dict(),
        },
        "ancestry_counts": {
            "aou_train": train_output["aou_ancestry_label"].value_counts().to_dict(),
            "aou_calibration": calibration_output["aou_ancestry_label"].value_counts().to_dict(),
            "aou_heldout": heldout_output["aou_ancestry_label"].value_counts().to_dict(),
        },
    }
    write_json(transformations_json_path, transformations_payload)

    write_json(imputation_json_path, imputation_summary)

    write_json(
        column_map_json_path,
        {
            "label_column": label_column,
            "ancestry_column": ancestry_column,
            "sample_id_column": sample_id_column,
            "pc_output_columns": list(ANCESTRY_COLUMNS),
            "dosage_column_map": dosage_column_map,
        },
    )

    logger.info("Wrote %s rows=%d", train_pickle_path, len(train_output))
    logger.info("Wrote %s rows=%d", calibration_pickle_path, len(calibration_output))
    logger.info("Wrote %s rows=%d", heldout_pickle_path, len(heldout_output))
    logger.info("Wrote %s rows=%d", tracking_pickle_path, len(tracking_frame))
    logger.info("Finished AOU evaluator-data preparation")

    return {
        "train_pickle_path": train_pickle_path,
        "calibration_pickle_path": calibration_pickle_path,
        "heldout_pickle_path": heldout_pickle_path,
        "tracking_pickle_path": tracking_pickle_path,
        "transformations_json_path": transformations_json_path,
        "imputation_json_path": imputation_json_path,
        "column_map_json_path": column_map_json_path,
        "log_path": output_dir / DEFAULT_LOG_NAME,
    }


def main() -> int:
    args = parse_args()

    tr_df = pd.read_pickle(args.tr_pickle)
    te_df = pd.read_pickle(args.te_pickle)

    explicit_dosage_columns = parse_csv_list(args.dosage_columns)
    dosage_columns = explicit_dosage_columns if explicit_dosage_columns else None

    output_paths = build_aou_evaluator_data(
        tr_df=tr_df,
        te_df=te_df,
        output_dir=args.output_dir,
        eval_ancestries=parse_csv_list(args.eval_ancestries),
        label_column=args.label_column,
        ancestry_column=args.ancestry_column,
        sample_id_column=args.sample_id_column,
        pc_prefix=args.pc_prefix,
        dosage_columns=dosage_columns,
        calibration_fraction=args.calibration_fraction,
        random_seed=args.random_seed,
        split_strategy=args.split_strategy,
        heldout_single_class_policy=args.heldout_single_class_policy,
        impute_dosages=not args.no_impute_dosages,
    )

    print("AOU evaluator data written:")
    for name, path in output_paths.items():
        print(f"{name}={path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())