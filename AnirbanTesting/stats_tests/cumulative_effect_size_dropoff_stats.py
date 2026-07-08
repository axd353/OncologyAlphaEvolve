from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from funsearch_pipeline.evaluation.procedure2 import _build_priority_ancestry_coordinate
from funsearch_pipeline.evaluation.procedure2 import _build_priority_training_data_contract
from funsearch_pipeline.evaluation.procedure2 import _logical_variant_name
from funsearch_pipeline.priority_tools import PriorityAncestryCoordinate
from funsearch_pipeline.priority_tools import PriorityTargetVariant
from funsearch_pipeline.priority_tools import PriorityTrainingData
from funsearch_pipeline.priority_tools import effect_size_by_cumulative_radius


DEFAULT_DATA_ROOT = Path(
    "/nfs/home/adas23/projects/AlphaEvolve/AnirbanTesting/"
    "test_evaluator1/experiment/preprocessed/no_covariates"
)
DEFAULT_NUM_SAMPLES = 400
DEFAULT_NUM_BALLS = 15
DEFAULT_RANDOM_SEED = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample calibration-subject/variant pairs from prepared Procedure 2 "
            "artifacts, apply the cumulative effect-size drop-off priority rule, "
            "and save the sampled results to a pickled pandas DataFrame."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing oracle_train.pkl and calibration.pkl. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of (ancestry_coordinate, variant) pairs to sample. Default: {DEFAULT_NUM_SAMPLES}",
    )
    parser.add_argument(
        "--num-balls",
        type=int,
        default=DEFAULT_NUM_BALLS,
        help=f"Number of cumulative balls used by effect_size_by_cumulative_radius. Default: {DEFAULT_NUM_BALLS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducible sampling. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).with_suffix(".pkl"),
        help="Pickle path for the output DataFrame. Default: same stem as this script with .pkl extension.",
    )
    return parser.parse_args()


def _ancestry_fields(frame: pd.DataFrame) -> tuple[str, ...]:
    fields = [
        str(column)
        for column in frame.columns
        if re.fullmatch(r"PC\d+", str(column))
    ]
    if not fields:
        raise ValueError("No ancestry coordinate fields matching PC1, PC2, ... were found.")
    return tuple(sorted(fields, key=lambda field_name: int(field_name[2:])))


def _priority_with_standardized_return(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
    num_balls: int,
) -> tuple[float, float, list[tuple[float, float]], int | None]:
    radii_and_effects = effect_size_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        target_variant,
        num_balls,
    )
    if not radii_and_effects:
        return 0.0, math.nan, [], None
    if len(radii_and_effects) == 1:
        return radii_and_effects[0][0], math.nan, list(radii_and_effects), 1

    drop_index = max(
        range(1, len(radii_and_effects)),
        key=lambda index: abs(radii_and_effects[index - 1][1] - radii_and_effects[index][1]),
    )
    returned_radius = radii_and_effects[drop_index][0]
    r2 = radii_and_effects[1][0]
    rn = radii_and_effects[-1][0]
    denominator = rn - r2
    standardized_return = math.nan
    if denominator != 0.0:
        standardized_return = (returned_radius - r2) / denominator
    return returned_radius, standardized_return, list(radii_and_effects), drop_index + 1


def main() -> None:
    args = _parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.num_balls < 1:
        raise ValueError("--num-balls must be positive.")

    oracle_train_path = args.data_root / "oracle_train.pkl"
    calibration_path = args.data_root / "calibration.pkl"
    oracle_train = pd.read_pickle(oracle_train_path)
    calibration = pd.read_pickle(calibration_path)

    variant_fields = tuple(
        str(column)
        for column in oracle_train.columns
        if str(column).startswith("dosage__")
    )
    if not variant_fields:
        raise ValueError("No dosage columns were found in oracle_train.pkl.")

    variant_index_by_field = {
        dosage_field: column_index
        for column_index, dosage_field in enumerate(variant_fields)
    }
    ancestry_fields = _ancestry_fields(calibration)
    training_data = _build_priority_training_data_contract(oracle_train, variant_fields)
    calibration_indices = tuple(calibration.index)

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    for sample_index in range(args.num_samples):
        calibration_index = rng.choice(calibration_indices)
        sampled_row = calibration.loc[calibration_index]
        ancestry_values = tuple(float(sampled_row[field_name]) for field_name in ancestry_fields)
        ancestry_coordinate = _build_priority_ancestry_coordinate(ancestry_values)

        dosage_field = rng.choice(variant_fields)
        target_variant = PriorityTargetVariant(
            name=_logical_variant_name(dosage_field),
            dosage_field=dosage_field,
            column_index=variant_index_by_field[dosage_field],
        )

        returned_radius, standardized_return, radii_and_effects, returned_position = _priority_with_standardized_return(
            training_data,
            ancestry_coordinate,
            target_variant,
            args.num_balls,
        )

        rows.append(
            {
                "sample_index": sample_index,
                "calibration_index": calibration_index,
                "pair": (ancestry_values, target_variant.name),
                "ancestry_coordinate": ancestry_values,
                "variant_name": target_variant.name,
                "dosage_field": target_variant.dosage_field,
                "column_index": target_variant.column_index,
                "num_balls": args.num_balls,
                "returned_radius": returned_radius,
                "standardized_returned_radius": standardized_return,
                "returned_position": returned_position,
                "radii_and_effects": radii_and_effects,
            }
        )

    output_frame = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_pickle(args.output_path)

    print(f"wrote {len(output_frame)} sampled pairs to {args.output_path}")
    print(
        output_frame[
            [
                "sample_index",
                "variant_name",
                "returned_radius",
                "standardized_returned_radius",
                "returned_position",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()