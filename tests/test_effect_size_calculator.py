from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GenomicsHelpers.effect_size_calculator import effect_size_calculator
from GenomicsHelpers.effect_size_calculator import prepare_local_variant_data


def _record(phenotype: int, dosage: float) -> dict[str, float]:
    record = {"phenotype": phenotype, "dosage__rsid_rs1": dosage}
    for index in range(1, 17):
        record[f"PC{index}"] = 0.0
    return record


def _ancestry_coordinate() -> list[float]:
    return [0.0] * 16


def test_prepare_local_variant_data_mean_imputes_missing_dosage() -> None:
    training_data = [
        _record(0, 0.0),
        _record(1, 1.0),
        _record(0, float("nan")),
        _record(1, 2.0),
        _record(0, 1.0),
        _record(1, 2.0),
    ]

    local_data = prepare_local_variant_data(
        training_data=training_data,
        ancestry_coordinate=_ancestry_coordinate(),
        target_variant="dosage__rsid_rs1",
        radius=1.0,
    )

    assert local_data.sample_count == 6
    assert np.isfinite(local_data.genotype).all()
    assert np.isclose(local_data.genotype[2], 1.2)


def test_effect_size_calculator_falls_back_when_all_dosages_missing() -> None:
    training_data = [
        _record(0, float("nan")),
        _record(1, float("nan")),
        _record(0, float("nan")),
        _record(1, float("nan")),
        _record(0, float("nan")),
        _record(1, float("nan")),
    ]

    effect = effect_size_calculator(
        training_data=training_data,
        ancestry_coordinate=_ancestry_coordinate(),
        target_variant="dosage__rsid_rs1",
        radius=1.0,
        min_samples=3,
        fallback_effect=-7.0,
    )

    assert effect == -7.0