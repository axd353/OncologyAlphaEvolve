from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from PostProcesingData.evaluate_priofunction import evaluate_from_config_path
from PostProcesingData.evaluate_priofunction import format_report
from PostProcesingData.evaluate_priofunction import load_evaluation_config
from PostProcesingData.evaluate_priofunction import _resolve_partition_count
from tests.test_procedure2_evaluator import _make_synthetic_oracle_frame


def _write_pickle(path: Path, data_frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(path)
    return path


def _write_text(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def test_load_evaluation_config_filters_disabled_baselines(tmp_path: Path) -> None:
    priority_path = _write_text(
        tmp_path / "priority.py",
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 1.0\n",
    )
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "prio_function_path": str(priority_path),
                "baselines": [
                    {"name": "Mixture Learning", "enabled": False},
                    {"name": "Mixture Learning", "enabled": True, "alpha": 2.5},
                ],
            },
            indent=2,
        ),
    )

    config = load_evaluation_config(config_path)

    assert len(config.baselines) == 1
    assert config.baselines[0].name == "Mixture Learning"
    assert config.baselines[0].options == {"alpha": 2.5}


def test_load_evaluation_config_accepts_auto_partitions(tmp_path: Path) -> None:
    priority_path = _write_text(
        tmp_path / "priority.py",
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 1.0\n",
    )
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "prio_function_path": str(priority_path),
                "calibration_partitions": "auto",
                "scoring_partitions": 0,
            },
            indent=2,
        ),
    )

    config = load_evaluation_config(config_path)

    assert config.calibration_partitions is None
    assert config.scoring_partitions is None


def test_resolve_partition_count_uses_visible_cpus_when_auto(monkeypatch) -> None:
    monkeypatch.setattr(
        "PostProcesingData.evaluate_priofunction._visible_cpu_count",
        lambda: 37,
    )

    assert _resolve_partition_count(None) == 37
    assert _resolve_partition_count(12) == 12


def test_evaluate_from_config_path_scores_priority_function_and_baseline(tmp_path: Path) -> None:
    training_path = _write_pickle(tmp_path / "train.pkl", _make_synthetic_oracle_frame(50))
    calibration_path = _write_pickle(
        tmp_path / "calibration.pkl",
        _make_synthetic_oracle_frame(40, offset=50),
    )
    heldout_path = _write_pickle(
        tmp_path / "heldout.pkl",
        _make_synthetic_oracle_frame(30, offset=90),
    )
    priority_path = _write_text(
        tmp_path / "priority.py",
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 10.0\n",
    )
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "prio_function_path": "priority.py",
                "training_pickle_path": "train.pkl",
                "calibrating_pickle_path": "calibration.pkl",
                "heldout_pickle_path": "heldout.pkl",
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 1,
                "scoring_partitions": 1,
                "baselines": [
                    {"name": "Mixture Learning", "alpha": 1.0},
                ],
            },
            indent=2,
        ),
    )

    report = evaluate_from_config_path(config_path)

    assert report.prio_function_path == priority_path
    assert report.heldout_auc_roc > 0.9
    assert len(report.baseline_evaluations) == 1
    assert report.baseline_evaluations[0].name == "Mixture Learning"
    assert report.baseline_evaluations[0].auc_roc > 0.9

    report_text = format_report(report)
    assert f"prio_function_path={priority_path}" in report_text
    assert "heldout_auc_roc=" in report_text
    assert "baseline_auc_roc[Mixture Learning]=" in report_text
