from __future__ import annotations

"""Build controlled no-covariates AFR evaluation datasets at fixed imbalance levels.

This script uses the no-covariates AFR focal outputs and the untouched
no-covariates OncoArray rows that were not selected into the AFR focal train,
test, or heldout pickles. It creates a shared heldout/calibration split and one
train split per imbalance level.

- heldout: 400 rows sampled from AFR focal no_covariates_heldout.pkl
- calibration: remaining heldout rows + 100 untouched non-African rows
- train: 160 untouched African rows + imbalance-specific untouched non-African rows

The same heldout and calibration datasets are written into each imbalance
directory under ControlledEvaluations/.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from build_funsearch_evaluator_data import configure_logger
from build_funsearch_evaluator_data import format_source_counts


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent
DEFAULT_ONCO_DIR = Path(__file__).resolve().parents[1] / "FunsearchEvaluatorDataOncoArray"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_DIR / "ControlledEvaluations"
DEFAULT_LOG_FILENAME = "build_funsearch_evaluator_exp_data_imb.log"
DEFAULT_TRACKING_FILENAME = "output_row_tracking.pkl"

ANCESTRY_AFR = "African_Ancestry"

NUM_NS = {
    "Very_high_imbalance": 3200,
    "high_imbalance": 2500,
    "modest_imbalance": 2000,
}


@dataclass(frozen=True)
class OutputSpec:
    name: str
    non_afr_train_count: int


OUTPUT_SPECS = tuple(
    OutputSpec(name=name, non_afr_train_count=count)
    for name, count in NUM_NS.items()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build controlled no-covariates AFR evaluation datasets for several "
            "non-African imbalance levels."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the AFR focal no-covariates pickles.",
    )
    parser.add_argument(
        "--onco-dir",
        type=Path,
        default=DEFAULT_ONCO_DIR,
        help="Directory containing the prepared OncoArray no-covariates pickles.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing the ControlledEvaluations subdirectories.",
    )
    parser.add_argument(
        "--heldout-size",
        type=int,
        default=400,
        help="Number of AFR focal heldout rows to keep in the controlled heldout split.",
    )
    parser.add_argument(
        "--calibration-extra-non-afr",
        type=int,
        default=100,
        help="Number of untouched non-African rows to append to calibration.",
    )
    parser.add_argument(
        "--train-afr-count",
        type=int,
        default=160,
        help="Number of untouched African rows in each train split.",
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
    for field_name in ("heldout_size", "calibration_extra_non_afr", "train_afr_count"):
        if getattr(args, field_name) < 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be non-negative.")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_root / DEFAULT_LOG_FILENAME)
    logger.info(
        (
            "Starting controlled AFR evaluation build input_dir=%s onco_dir=%s "
            "output_root=%s heldout_size=%d calibration_extra_non_afr=%d "
            "train_afr_count=%d random_seed=%d"
        ),
        args.input_dir,
        args.onco_dir,
        output_root,
        args.heldout_size,
        args.calibration_extra_non_afr,
        args.train_afr_count,
        args.random_seed,
    )

    heldout_dataset = pd.read_pickle(args.input_dir / "no_covariates_heldout.pkl").reset_index(drop=True)
    heldout_tracking = load_tracking(args.input_dir).loc[
        lambda frame: frame["output_pickle_name"] == "no_covariates_heldout.pkl"
    ].reset_index(drop=True)
    if len(heldout_dataset) != len(heldout_tracking):
        raise ValueError(
            "Heldout tracking mismatch for no_covariates_heldout.pkl: "
            f"dataset rows={len(heldout_dataset)} tracking rows={len(heldout_tracking)}."
        )

    untouched_dataset, untouched_tracking = load_untouched_no_covariates_pool(
        focal_dir=args.input_dir,
        onco_dir=args.onco_dir,
    )

    rng = np.random.default_rng(args.random_seed)
    heldout_indices, calibration_heldout_indices = split_heldout_indices(
        row_count=len(heldout_dataset),
        heldout_size=int(args.heldout_size),
        rng=rng,
    )

    untouched_afr_indices = ancestry_indices(untouched_tracking, ANCESTRY_AFR)
    untouched_non_afr_indices = ancestry_indices(untouched_tracking, ANCESTRY_AFR, invert=True)
    max_non_afr_needed = int(args.calibration_extra_non_afr) + max(
        spec.non_afr_train_count for spec in OUTPUT_SPECS
    )
    if len(untouched_afr_indices) < int(args.train_afr_count):
        raise ValueError(
            f"Requested {args.train_afr_count} untouched African rows, but only {len(untouched_afr_indices)} are available."
        )
    if len(untouched_non_afr_indices) < max_non_afr_needed:
        raise ValueError(
            "Requested calibration plus train non-African rows exceed the untouched pool: "
            f"needed={max_non_afr_needed} available={len(untouched_non_afr_indices)}."
        )

    selected_train_afr = choose_without_replacement(
        untouched_afr_indices,
        int(args.train_afr_count),
        rng=rng,
    )
    selected_non_afr = choose_without_replacement(
        untouched_non_afr_indices,
        max_non_afr_needed,
        rng=rng,
    )
    calibration_extra_non_afr = selected_non_afr[: int(args.calibration_extra_non_afr)]
    train_non_afr_pool = np.array(
        sorted(set(untouched_non_afr_indices.tolist()) - set(calibration_extra_non_afr.tolist())),
        dtype=int,
    )

    shared_heldout_dataset, shared_heldout_tracking = select_rows(
        heldout_dataset,
        heldout_tracking,
        heldout_indices,
        output_name="no_covariates_heldout.pkl",
    )
    calibration_from_heldout_dataset, calibration_from_heldout_tracking = select_rows(
        heldout_dataset,
        heldout_tracking,
        calibration_heldout_indices,
        output_name="no_covariates_calibration.pkl",
    )
    calibration_extra_dataset, calibration_extra_tracking = select_rows(
        untouched_dataset,
        untouched_tracking,
        calibration_extra_non_afr,
        output_name="no_covariates_calibration.pkl",
    )
    shared_calibration_dataset = pd.concat(
        [calibration_from_heldout_dataset, calibration_extra_dataset],
        ignore_index=True,
    )
    shared_calibration_tracking = concat_tracking(
        [calibration_from_heldout_tracking, calibration_extra_tracking],
        output_name="no_covariates_calibration.pkl",
    )

    logger.info(
        "Shared heldout rows=%d sources=%s",
        len(shared_heldout_dataset),
        summarize_tracking_sources(shared_heldout_tracking),
    )
    logger.info(
        "Shared calibration rows=%d heldout_remainder=%d extra_non_afr=%d sources=%s",
        len(shared_calibration_dataset),
        len(calibration_from_heldout_dataset),
        len(calibration_extra_dataset),
        summarize_tracking_sources(shared_calibration_tracking),
    )

    for spec in OUTPUT_SPECS:
        train_non_afr_indices = choose_without_replacement(
            train_non_afr_pool,
            spec.non_afr_train_count,
            rng=rng,
        )
        train_afr_dataset, train_afr_tracking = select_rows(
            untouched_dataset,
            untouched_tracking,
            selected_train_afr,
            output_name="no_covariates_train.pkl",
        )
        train_non_afr_dataset, train_non_afr_tracking = select_rows(
            untouched_dataset,
            untouched_tracking,
            train_non_afr_indices,
            output_name="no_covariates_train.pkl",
        )
        train_dataset = pd.concat([train_afr_dataset, train_non_afr_dataset], ignore_index=True)
        train_tracking = concat_tracking(
            [train_afr_tracking, train_non_afr_tracking],
            output_name="no_covariates_train.pkl",
        )

        target_dir = output_root / spec.name
        target_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_bundle(
            output_dir=target_dir,
            heldout_dataset=shared_heldout_dataset,
            heldout_tracking=shared_heldout_tracking,
            calibration_dataset=shared_calibration_dataset,
            calibration_tracking=shared_calibration_tracking,
            train_dataset=train_dataset,
            train_tracking=train_tracking,
            logger=logger,
        )
        logger.info(
            "Wrote %s train rows=%d afr=%d non_afr=%d sources=%s",
            spec.name,
            len(train_dataset),
            len(selected_train_afr),
            len(train_non_afr_indices),
            summarize_tracking_sources(train_tracking),
        )

    logger.info("Finished controlled AFR evaluation build")


def load_tracking(directory: Path) -> pd.DataFrame:
    tracking_path = directory / DEFAULT_TRACKING_FILENAME
    if not tracking_path.exists():
        raise FileNotFoundError(f"Missing tracking file: {tracking_path}")
    return pd.read_pickle(tracking_path).copy()


def logical_source_key(source_name: str) -> str:
    stem = Path(source_name).stem
    if stem.endswith("_add_covs"):
        stem = stem[: -len("_add_covs")]
    return stem


def source_name_to_ancestry_group(source_name: str) -> str:
    if ANCESTRY_AFR in source_name:
        return ANCESTRY_AFR
    return "non_African_Ancestry"


def load_untouched_no_covariates_pool(*, focal_dir: Path, onco_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    focal_tracking = load_tracking(focal_dir)
    focal_tracking = focal_tracking.loc[
        focal_tracking["output_pickle_name"].str.startswith("no_covariates_")
    ].copy()
    onco_tracking = load_tracking(onco_dir)
    onco_tracking = onco_tracking.loc[
        onco_tracking["output_pickle_name"].str.startswith("no_covariates_")
    ].copy()

    for frame in (focal_tracking, onco_tracking):
        frame["logical_source_name"] = frame["source_pickle_name"].map(logical_source_key)
        frame["ancestry_group"] = frame["source_pickle_name"].map(source_name_to_ancestry_group)

    focal_keys = set(
        map(
            tuple,
            focal_tracking[["logical_source_name", "source_row_number"]].itertuples(index=False, name=None),
        )
    )

    frames: list[pd.DataFrame] = []
    tracking_parts: list[pd.DataFrame] = []
    for split_name in ("heldout", "test", "train"):
        output_name = f"no_covariates_{split_name}.pkl"
        dataset = pd.read_pickle(onco_dir / output_name).reset_index(drop=True)
        split_tracking = onco_tracking.loc[
            onco_tracking["output_pickle_name"] == output_name
        ].sort_values("output_row_number").reset_index(drop=True)
        if len(dataset) != len(split_tracking):
            raise ValueError(
                f"Tracking length mismatch for {output_name}: dataset has {len(dataset)} rows, "
                f"tracking has {len(split_tracking)} rows."
            )
        frames.append(dataset)
        tracking_parts.append(split_tracking)

    pooled_dataset = pd.concat(frames, ignore_index=True)
    pooled_tracking = pd.concat(tracking_parts, ignore_index=True)
    pooled_keys = pooled_tracking[["logical_source_name", "source_row_number"]].apply(tuple, axis=1)
    untouched_mask = ~pooled_keys.isin(focal_keys)
    untouched_dataset = pooled_dataset.loc[untouched_mask].reset_index(drop=True)
    untouched_tracking = pooled_tracking.loc[untouched_mask].reset_index(drop=True)
    untouched_tracking = untouched_tracking.assign(
        output_pickle_name="untouched_no_covariates_pool.pkl",
        output_row_number=np.arange(len(untouched_tracking), dtype=int),
    )
    return untouched_dataset, untouched_tracking


def split_heldout_indices(*, row_count: int, heldout_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if heldout_size > row_count:
        raise ValueError(
            f"Requested heldout size {heldout_size}, but only {row_count} AFR focal heldout rows are available."
        )
    permutation = rng.permutation(row_count).astype(int)
    return permutation[:heldout_size], permutation[heldout_size:]


def ancestry_indices(tracking: pd.DataFrame, ancestry_name: str, *, invert: bool = False) -> np.ndarray:
    mask = tracking["ancestry_group"] == ancestry_name
    if invert:
        mask = ~mask
    return np.flatnonzero(mask.to_numpy())


def choose_without_replacement(indices: np.ndarray, count: int, *, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if count > len(indices):
        raise ValueError(f"Requested {count} rows, but only {len(indices)} are available.")
    if count == 0:
        return np.array([], dtype=int)
    chosen_positions = rng.choice(len(indices), size=count, replace=False)
    return np.sort(indices[np.asarray(chosen_positions, dtype=int)])


def select_rows(
    dataset: pd.DataFrame,
    tracking: pd.DataFrame,
    indices: np.ndarray,
    *,
    output_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = np.asarray(indices, dtype=int)
    selected_dataset = dataset.iloc[indices].reset_index(drop=True).copy()
    selected_tracking = tracking.iloc[indices].reset_index(drop=True).copy()
    selected_tracking = selected_tracking.assign(
        output_pickle_name=output_name,
        output_row_number=np.arange(len(selected_tracking), dtype=int),
    )
    selected_tracking = selected_tracking.loc[
        :,
        [
            "output_pickle_name",
            "output_row_number",
            "source_pickle_name",
            "source_pickle_path",
            "source_row_number",
        ],
    ]
    return selected_dataset, selected_tracking


def concat_tracking(parts: list[pd.DataFrame], *, output_name: str) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.assign(
        output_pickle_name=output_name,
        output_row_number=np.arange(len(combined), dtype=int),
    )
    return combined.loc[
        :,
        [
            "output_pickle_name",
            "output_row_number",
            "source_pickle_name",
            "source_pickle_path",
            "source_row_number",
        ],
    ]


def summarize_tracking_sources(tracking: pd.DataFrame) -> str:
    source_counts = tracking["source_pickle_name"].value_counts(sort=False).to_dict()
    return format_source_counts({str(key): int(value) for key, value in source_counts.items()})


def write_dataset_bundle(
    *,
    output_dir: Path,
    heldout_dataset: pd.DataFrame,
    heldout_tracking: pd.DataFrame,
    calibration_dataset: pd.DataFrame,
    calibration_tracking: pd.DataFrame,
    train_dataset: pd.DataFrame,
    train_tracking: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    outputs = {
        "no_covariates_heldout.pkl": heldout_dataset,
        "no_covariates_calibration.pkl": calibration_dataset,
        "no_covariates_train.pkl": train_dataset,
    }
    for file_name, dataset in outputs.items():
        dataset.to_pickle(output_dir / file_name)

    tracking = pd.concat([heldout_tracking, calibration_tracking, train_tracking], ignore_index=True)
    tracking.to_pickle(output_dir / DEFAULT_TRACKING_FILENAME)
    logger.info("Wrote tracking file %s rows=%d", output_dir / DEFAULT_TRACKING_FILENAME, len(tracking))


if __name__ == "__main__":
    main()