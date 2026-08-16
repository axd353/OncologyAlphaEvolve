from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import PostProcesingData.evaluate_priofunction as evaluate_priofunction_module
from PostProcesingData.evaluate_priofunction import evaluate_from_config_path
from PostProcesingData.evaluate_priofunction import evaluate_priority_function
from PostProcesingData.evaluate_priofunction import BaselineEvaluation
from PostProcesingData.evaluate_priofunction import format_report
from PostProcesingData.evaluate_priofunction import HeldoutAncestryEvaluation
from PostProcesingData.evaluate_priofunction import EvaluationReport
from PostProcesingData.evaluate_priofunction import heldout_model_predictions_output_path_for_model
from PostProcesingData.evaluate_priofunction import load_existing_evaluation_report
from PostProcesingData.evaluate_priofunction import load_evaluation_config
from PostProcesingData.evaluate_priofunction import main
from PostProcesingData.evaluate_priofunction import merge_evaluation_reports
from PostProcesingData.evaluate_priofunction import report_output_path_for_priority_function
from PostProcesingData.evaluate_priofunction import write_evaluation_report_file
from PostProcesingData.evaluate_priofunction import _missing_baselines
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


def _write_tracking_pickle(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(path)
    return path


def _helper_based_priority_source() -> str:
    return (
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    near_radius = radius_for_percentage(training_data, ancestry_coordinate, 50.0)\n"
        "    far_radius = radius_for_percentage(training_data, ancestry_coordinate, 75.0)\n"
        "    novelty = ancestry_novelty_score(training_data, ancestry_coordinate)\n"
        "    return far_radius if novelty > 1.5 else near_radius\n"
    )


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
                    {"name": "Independent Learning Scheme", "enabled": False},
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


def test_load_evaluation_config_accepts_path_lists_and_tracking_path(tmp_path: Path) -> None:
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
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "tracking.pkl",
            },
            indent=2,
        ),
    )

    config = load_evaluation_config(config_path)

    assert config.training_pickle_paths == (
        tmp_path / "train_a.pkl",
        tmp_path / "train_b.pkl",
    )
    assert config.calibrating_pickle_paths == (
        tmp_path / "calibration_a.pkl",
        tmp_path / "calibration_b.pkl",
    )
    assert config.heldout_pickle_paths == (
        tmp_path / "heldout_a.pkl",
        tmp_path / "heldout_b.pkl",
    )
    assert config.output_row_tracking_path == tmp_path / "tracking.pkl"


def test_load_evaluation_config_reads_should_overwrite_flag(tmp_path: Path) -> None:
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
                "should_overwrite": False,
            },
            indent=2,
        ),
    )

    config = load_evaluation_config(config_path)

    assert config.should_overwrite is False


def test_load_evaluation_config_reads_report_file_name_and_supported_ancestries(
    tmp_path: Path,
) -> None:
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
                "report_file_name": "custom_eval_report.json",
                "supported_ancestry_groups": ["ea", "ja", "ea"],
            },
            indent=2,
        ),
    )

    config = load_evaluation_config(config_path)

    assert config.report_file_name == "custom_eval_report.json"
    assert config.supported_ancestry_groups == ("EA", "JA")


def test_resolve_partition_count_uses_visible_cpus_when_auto(monkeypatch) -> None:
    monkeypatch.setattr(
        "PostProcesingData.evaluate_priofunction._visible_cpu_count",
        lambda: 37,
    )

    assert _resolve_partition_count(None) == 37
    assert _resolve_partition_count(12) == 12


def test_evaluate_from_config_path_scores_priority_function_and_baseline(tmp_path: Path) -> None:
    _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(25))
    _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(25, offset=25),
    )
    _write_pickle(
        tmp_path / "calibration_a.pkl",
        _make_synthetic_oracle_frame(20, offset=50),
    )
    _write_pickle(
        tmp_path / "calibration_b.pkl",
        _make_synthetic_oracle_frame(20, offset=70),
    )
    _write_pickle(
        tmp_path / "heldout_a.pkl",
        _make_synthetic_oracle_frame(12, offset=90),
    )
    _write_pickle(
        tmp_path / "heldout_b.pkl",
        _make_synthetic_oracle_frame(12, offset=102),
    )
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "train_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "calibration_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "calibration_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        +
        [
            {
                "output_pickle_name": "heldout_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ]
        + [
            {
                "output_pickle_name": "heldout_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ],
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
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 1,
                "scoring_partitions": 1,
                "baselines": [
                    {"name": "Mixture Learning", "alpha": 1.0},
                    {"name": "Independent Learning Scheme", "alpha": 1.0},
                ],
            },
            indent=2,
        ),
    )

    report = evaluate_from_config_path(config_path)

    assert report.prio_function_path == priority_path
    assert report.heldout_auc_roc > 0.9
    assert report.heldout_ancestry_evaluations[0].ancestry_group == "AA"
    assert report.heldout_ancestry_evaluations[0].subject_count == 12
    assert report.heldout_ancestry_evaluations[0].auc_roc > 0.9
    assert report.heldout_ancestry_evaluations[1].ancestry_group == "JA"
    assert report.heldout_ancestry_evaluations[1].subject_count == 12
    assert report.heldout_ancestry_evaluations[1].auc_roc > 0.9
    assert len(report.baseline_evaluations) == 2
    assert report.baseline_evaluations[0].name == "Mixture Learning"
    assert report.baseline_evaluations[0].auc_roc > 0.9
    assert report.baseline_evaluations[0].heldout_ancestry_evaluations[0].ancestry_group == "AA"
    assert report.baseline_evaluations[0].heldout_ancestry_evaluations[0].subject_count == 12
    assert report.baseline_evaluations[0].heldout_ancestry_evaluations[0].auc_roc > 0.9
    assert report.baseline_evaluations[0].heldout_ancestry_evaluations[1].ancestry_group == "JA"
    assert report.baseline_evaluations[0].heldout_ancestry_evaluations[1].subject_count == 12
    assert report.baseline_evaluations[0].heldout_ancestry_evaluations[1].auc_roc > 0.9
    assert report.baseline_evaluations[1].name == "Independent Learning Scheme"
    assert report.baseline_evaluations[1].auc_roc > 0.9
    assert report.baseline_evaluations[1].heldout_ancestry_evaluations[0].ancestry_group == "AA"
    assert report.baseline_evaluations[1].heldout_ancestry_evaluations[0].subject_count == 12
    assert report.baseline_evaluations[1].heldout_ancestry_evaluations[0].auc_roc > 0.9
    assert report.baseline_evaluations[1].heldout_ancestry_evaluations[1].ancestry_group == "JA"
    assert report.baseline_evaluations[1].heldout_ancestry_evaluations[1].subject_count == 12
    assert report.baseline_evaluations[1].heldout_ancestry_evaluations[1].auc_roc > 0.9

    predictions_path = tmp_path / "distance_cache" / "heldout_model_predictions.pkl"
    assert predictions_path.exists()
    predictions_frame = pd.read_pickle(predictions_path)
    assert list(predictions_frame.columns) == [
        "heldout_subject_index",
        "heldout_output_pickle_name",
        "heldout_output_pickle_path",
        "heldout_output_row_number",
        "source_pickle_name",
        "source_pickle_path",
        "source_row_number",
        "ancestry_group",
        "label",
        "model_name",
        "model_slug",
        "risk_score",
        "risk_probability",
    ]
    assert len(predictions_frame) == 24 * 3
    assert predictions_frame["heldout_subject_index"].nunique() == 24
    assert set(predictions_frame["model_name"].tolist()) == {
        "Priority Function",
        "Mixture Learning",
        "Independent Learning Scheme",
    }
    assert predictions_frame.groupby("model_name").size().to_dict() == {
        "Priority Function": 24,
        "Mixture Learning": 24,
        "Independent Learning Scheme": 24,
    }
    assert predictions_frame["risk_probability"].between(0.0, 1.0).all()

    priority_rows = predictions_frame.loc[
        predictions_frame["model_name"] == "Priority Function"
    ].reset_index(drop=True)
    assert priority_rows["ancestry_group"].tolist().count("AA") == 12
    assert priority_rows["ancestry_group"].tolist().count("JA") == 12
    assert priority_rows["heldout_output_pickle_name"].tolist()[:12] == ["heldout_a.pkl"] * 12
    assert priority_rows["heldout_output_pickle_name"].tolist()[12:] == ["heldout_b.pkl"] * 12
    assert priority_rows["source_pickle_name"].tolist()[:12] == ["train_AA.pkl"] * 12
    assert priority_rows["source_pickle_name"].tolist()[12:] == ["train_JA.pkl"] * 12
    assert priority_rows["source_row_number"].tolist()[:12] == list(range(12))
    assert priority_rows["source_row_number"].tolist()[12:] == list(range(12))


def test_evaluate_priority_function_distance_cache_matches_uncached(tmp_path: Path) -> None:
    _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(25))
    _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(25, offset=25),
    )
    _write_pickle(
        tmp_path / "calibration_a.pkl",
        _make_synthetic_oracle_frame(20, offset=50),
    )
    _write_pickle(
        tmp_path / "calibration_b.pkl",
        _make_synthetic_oracle_frame(20, offset=70),
    )
    _write_pickle(
        tmp_path / "heldout_a.pkl",
        _make_synthetic_oracle_frame(12, offset=90),
    )
    _write_pickle(
        tmp_path / "heldout_b.pkl",
        _make_synthetic_oracle_frame(12, offset=102),
    )
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "train_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "calibration_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "calibration_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "heldout_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ]
        + [
            {
                "output_pickle_name": "heldout_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ],
    )
    priority_path = _write_text(tmp_path / "priority.py", _helper_based_priority_source())
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "prio_function_path": "priority.py",
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 2,
                "scoring_partitions": 2,
                "baselines": [],
            },
            indent=2,
        ),
    )

    base_config = load_evaluation_config(config_path)
    uncached_report = evaluate_priority_function(
        replace(base_config, distance_cache_enabled=False, baselines=()),
        baselines_to_run=(),
        progress_reporter=lambda _message: None,
    )
    cached_cache_dir = tmp_path / "distance_cache"
    cached_report = evaluate_priority_function(
        replace(
            base_config,
            distance_cache_enabled=True,
            distance_cache_dir=cached_cache_dir,
            baselines=(),
        ),
        baselines_to_run=(),
        progress_reporter=lambda _message: None,
    )

    assert np.isclose(cached_report.heldout_auc_roc, uncached_report.heldout_auc_roc)
    assert cached_report.prio_function_path == priority_path
    assert tuple(
        (evaluation.ancestry_group, evaluation.subject_count)
        for evaluation in cached_report.heldout_ancestry_evaluations
    ) == tuple(
        (evaluation.ancestry_group, evaluation.subject_count)
        for evaluation in uncached_report.heldout_ancestry_evaluations
    )
    for cached_evaluation, uncached_evaluation in zip(
        cached_report.heldout_ancestry_evaluations,
        uncached_report.heldout_ancestry_evaluations,
    ):
        assert np.isclose(cached_evaluation.auc_roc, uncached_evaluation.auc_roc)

    assert any(cached_cache_dir.glob("*.manifest.json"))

    report_text = format_report(cached_report)
    assert f"prio_function_path={priority_path}" in report_text
    assert "heldout_auc_roc=" in report_text
    assert "heldout_subject_count[AA]=12" in report_text
    assert "heldout_auc_roc[AA]=" in report_text
    assert "heldout_subject_count[JA]=12" in report_text
    assert "heldout_auc_roc[JA]=" in report_text


def test_evaluate_from_config_path_scores_tl_transfer_baselines(tmp_path: Path) -> None:
    _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(30))
    _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(30, offset=30),
    )
    _write_pickle(
        tmp_path / "calibration_a.pkl",
        _make_synthetic_oracle_frame(20, offset=60),
    )
    _write_pickle(
        tmp_path / "calibration_b.pkl",
        _make_synthetic_oracle_frame(20, offset=80),
    )
    _write_pickle(
        tmp_path / "heldout_a.pkl",
        _make_synthetic_oracle_frame(14, offset=100),
    )
    _write_pickle(
        tmp_path / "heldout_b.pkl",
        _make_synthetic_oracle_frame(14, offset=114),
    )
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(30)
        ]
        + [
            {
                "output_pickle_name": "train_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(30)
        ]
        + [
            {
                "output_pickle_name": "calibration_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "calibration_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "heldout_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(14)
        ]
        + [
            {
                "output_pickle_name": "heldout_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(14)
        ],
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
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 1,
                "scoring_partitions": 1,
                "baselines": [
                    {"name": "TL-GDES", "max_iter": 20, "source_n_iter": 200},
                    {"name": "TL-PR", "max_iter": 100, "source_n_iter": 200, "n_lambdas": 6},
                ],
            },
            indent=2,
        ),
    )

    report = evaluate_from_config_path(config_path)

    assert report.prio_function_path == priority_path
    assert [baseline.name for baseline in report.baseline_evaluations] == ["TL-GDES", "TL-PR"]
    for baseline in report.baseline_evaluations:
        assert baseline.auc_roc > 0.9
        assert [item.ancestry_group for item in baseline.heldout_ancestry_evaluations] == ["AA", "JA"]
        assert all(item.subject_count == 14 for item in baseline.heldout_ancestry_evaluations)
        assert all(item.auc_roc > 0.9 for item in baseline.heldout_ancestry_evaluations)

    report_text = format_report(report)
    assert "baseline_auc_roc[TL-GDES]=" in report_text
    assert "baseline_auc_roc[TL-GDES][AA]=" in report_text
    assert "baseline_auc_roc[TL-GDES][JA]=" in report_text
    assert "baseline_auc_roc[TL-PR]=" in report_text
    assert "baseline_auc_roc[TL-PR][AA]=" in report_text
    assert "baseline_auc_roc[TL-PR][JA]=" in report_text


def test_evaluate_from_config_path_uses_configured_supported_ancestry_groups(
    tmp_path: Path,
) -> None:
    _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(25))
    _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(25, offset=25),
    )
    _write_pickle(
        tmp_path / "calibration_a.pkl",
        _make_synthetic_oracle_frame(20, offset=50),
    )
    _write_pickle(
        tmp_path / "calibration_b.pkl",
        _make_synthetic_oracle_frame(20, offset=70),
    )
    _write_pickle(
        tmp_path / "heldout_a.pkl",
        _make_synthetic_oracle_frame(12, offset=90),
    )
    _write_pickle(
        tmp_path / "heldout_b.pkl",
        _make_synthetic_oracle_frame(12, offset=102),
    )
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_EA.pkl",
                "source_pickle_path": "/tmp/train_EA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "train_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "calibration_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_EA.pkl",
                "source_pickle_path": "/tmp/train_EA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "calibration_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "heldout_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_EA.pkl",
                "source_pickle_path": "/tmp/train_EA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ]
        + [
            {
                "output_pickle_name": "heldout_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ],
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
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "supported_ancestry_groups": ["EA", "JA"],
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 1,
                "scoring_partitions": 1,
            },
            indent=2,
        ),
    )

    report = evaluate_from_config_path(config_path)

    assert report.prio_function_path == priority_path
    assert [item.ancestry_group for item in report.heldout_ancestry_evaluations] == ["EA", "JA"]
    assert all(item.subject_count == 12 for item in report.heldout_ancestry_evaluations)

    report_text = format_report(report)
    assert "heldout_subject_count[EA]=12" in report_text
    assert "heldout_auc_roc[EA]=" in report_text
    assert "heldout_subject_count[JA]=12" in report_text


def test_evaluate_from_config_path_supports_multi_token_supported_ancestry_groups(
    tmp_path: Path,
) -> None:
    _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(25))
    _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(25, offset=25),
    )
    _write_pickle(
        tmp_path / "calibration_a.pkl",
        _make_synthetic_oracle_frame(20, offset=50),
    )
    _write_pickle(
        tmp_path / "calibration_b.pkl",
        _make_synthetic_oracle_frame(20, offset=70),
    )
    _write_pickle(
        tmp_path / "heldout_a.pkl",
        _make_synthetic_oracle_frame(12, offset=90),
    )
    _write_pickle(
        tmp_path / "heldout_b.pkl",
        _make_synthetic_oracle_frame(12, offset=102),
    )
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_African_Ancestry.pkl",
                "source_pickle_path": "/tmp/train_African_Ancestry.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "train_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_Asian.pkl",
                "source_pickle_path": "/tmp/train_Asian.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "calibration_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_African_Ancestry.pkl",
                "source_pickle_path": "/tmp/train_African_Ancestry.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "calibration_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_Asian.pkl",
                "source_pickle_path": "/tmp/train_Asian.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "heldout_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "test_African_Ancestry_add_covs.pkl",
                "source_pickle_path": "/tmp/test_African_Ancestry_add_covs.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ]
        + [
            {
                "output_pickle_name": "heldout_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "test_Asian.pkl",
                "source_pickle_path": "/tmp/test_Asian.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ],
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
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "supported_ancestry_groups": ["African_Ancestry", "Asian"],
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 1,
                "scoring_partitions": 1,
            },
            indent=2,
        ),
    )

    report = evaluate_from_config_path(config_path)

    assert report.prio_function_path == priority_path
    assert [item.ancestry_group for item in report.heldout_ancestry_evaluations] == [
        "AFRICAN_ANCESTRY",
        "ASIAN",
    ]
    assert all(item.subject_count == 12 for item in report.heldout_ancestry_evaluations)

    report_text = format_report(report)
    assert "heldout_subject_count[AFRICAN_ANCESTRY]=12" in report_text
    assert "heldout_auc_roc[AFRICAN_ANCESTRY]=" in report_text
    assert "heldout_subject_count[ASIAN]=12" in report_text


def test_write_evaluation_report_file_writes_clean_json_next_to_priority_function(
    tmp_path: Path,
) -> None:
    priority_dir = tmp_path / "cycle_0006"
    priority_path = _write_text(
        priority_dir / "best_prio.py",
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 1.0\n",
    )
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "prio_function_path": str(priority_path),
                "report_file_name": "custom_report.json",
            },
            indent=2,
        ),
    )
    config = load_evaluation_config(config_path)
    report = EvaluationReport(
        prio_function_path=priority_path,
        heldout_auc_roc=0.91,
        heldout_ancestry_evaluations=(
            HeldoutAncestryEvaluation(
                ancestry_group="AA",
                subject_count=10,
                auc_roc=0.92,
            ),
            HeldoutAncestryEvaluation(
                ancestry_group="JA",
                subject_count=4,
                auc_roc=0.88,
            ),
        ),
        baseline_evaluations=(
            BaselineEvaluation(
                name="Mixture Learning",
                auc_roc=0.87,
                heldout_ancestry_evaluations=(
                    HeldoutAncestryEvaluation(
                        ancestry_group="AA",
                        subject_count=10,
                        auc_roc=0.86,
                    ),
                ),
            ),
        ),
    )

    output_path = write_evaluation_report_file(
        config_path=config_path,
        config=config,
        report=report,
    )
    overwritten_path = write_evaluation_report_file(
        config_path=config_path,
        config=config,
        report=EvaluationReport(
            prio_function_path=priority_path,
            heldout_auc_roc=0.5,
            heldout_ancestry_evaluations=(),
            baseline_evaluations=(),
        ),
    )

    assert output_path == report_output_path_for_priority_function(
        priority_path,
        "custom_report.json",
    )
    assert overwritten_path == output_path
    payload = json.loads(output_path.read_text())
    assert payload["config_path"] == str(config_path)
    assert payload["config"]["prio_function_path"] == str(priority_path)
    assert payload["config"]["report_file_name"] == "custom_report.json"
    assert payload["results"]["prio_function_path"] == str(priority_path)
    assert payload["results"]["heldout_auc_roc"] == 0.5
    assert payload["results"]["heldout_model_predictions_path"] == str(
        priority_dir / "distance_cache" / "heldout_model_predictions.pkl"
    )
    assert payload["results"]["heldout_model_prediction_paths"] == {}
    assert payload["results"]["baseline_evaluations"] == []
    assert payload["results"]["summary_text"] == "prio_function_path=" + str(priority_path) + "\nheldout_auc_roc=0.500000"


def test_evaluate_priority_function_persists_partial_report_and_predictions_after_each_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(25))
    _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(25, offset=25),
    )
    _write_pickle(
        tmp_path / "calibration_a.pkl",
        _make_synthetic_oracle_frame(20, offset=50),
    )
    _write_pickle(
        tmp_path / "calibration_b.pkl",
        _make_synthetic_oracle_frame(20, offset=70),
    )
    _write_pickle(
        tmp_path / "heldout_a.pkl",
        _make_synthetic_oracle_frame(12, offset=90),
    )
    _write_pickle(
        tmp_path / "heldout_b.pkl",
        _make_synthetic_oracle_frame(12, offset=102),
    )
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "train_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "calibration_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "calibration_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "heldout_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ]
        + [
            {
                "output_pickle_name": "heldout_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ],
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
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 1,
                "scoring_partitions": 1,
                "distance_cache_dir": str(tmp_path / "distance_cache"),
                "should_overwrite": True,
                "baselines": [
                    {"name": "Mixture Learning", "alpha": 1.0},
                    {"name": "Independent Learning Scheme", "alpha": 1.0},
                ],
            },
            indent=2,
        ),
    )

    original_evaluate_baseline = evaluate_priofunction_module.evaluate_baseline

    def _failing_evaluate_baseline(name: str, **kwargs):
        baseline_name = evaluate_priofunction_module.normalize_baseline_name(name)
        if baseline_name == "Independent Learning Scheme":
            raise RuntimeError("forced baseline failure")
        return original_evaluate_baseline(name, **kwargs)

    monkeypatch.setattr(evaluate_priofunction_module, "evaluate_baseline", _failing_evaluate_baseline)

    with pytest.raises(RuntimeError, match="forced baseline failure"):
        evaluate_from_config_path(config_path)

    cache_root = tmp_path / "distance_cache"
    aggregate_predictions_path = cache_root / "heldout_model_predictions.pkl"
    assert aggregate_predictions_path.exists()
    aggregate_predictions = pd.read_pickle(aggregate_predictions_path)
    assert set(aggregate_predictions["model_name"].tolist()) == {
        "Priority Function",
        "Mixture Learning",
    }

    priority_predictions_path = heldout_model_predictions_output_path_for_model(
        cache_root,
        "Priority Function",
    )
    mixture_predictions_path = heldout_model_predictions_output_path_for_model(
        cache_root,
        "Mixture Learning",
    )
    missing_predictions_path = heldout_model_predictions_output_path_for_model(
        cache_root,
        "Independent Learning Scheme",
    )
    assert priority_predictions_path.exists()
    assert mixture_predictions_path.exists()
    assert not missing_predictions_path.exists()

    report_path = report_output_path_for_priority_function(priority_path)
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert [baseline["name"] for baseline in payload["results"]["baseline_evaluations"]] == [
        "Mixture Learning",
    ]
    assert payload["results"]["heldout_model_prediction_paths"] == {
        "Mixture Learning": str(mixture_predictions_path),
        "Priority Function": str(priority_predictions_path),
    }


def test_main_rebuilds_heldout_model_predictions_when_report_exists(tmp_path: Path) -> None:
    _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(25))
    _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(25, offset=25),
    )
    _write_pickle(
        tmp_path / "calibration_a.pkl",
        _make_synthetic_oracle_frame(20, offset=50),
    )
    _write_pickle(
        tmp_path / "calibration_b.pkl",
        _make_synthetic_oracle_frame(20, offset=70),
    )
    _write_pickle(
        tmp_path / "heldout_a.pkl",
        _make_synthetic_oracle_frame(12, offset=90),
    )
    _write_pickle(
        tmp_path / "heldout_b.pkl",
        _make_synthetic_oracle_frame(12, offset=102),
    )
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "train_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(25)
        ]
        + [
            {
                "output_pickle_name": "calibration_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "calibration_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(20)
        ]
        + [
            {
                "output_pickle_name": "heldout_a.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl",
                "source_pickle_path": "/tmp/train_AA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ]
        + [
            {
                "output_pickle_name": "heldout_b.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_JA.pkl",
                "source_pickle_path": "/tmp/train_JA.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(12)
        ],
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
                "training_pickle_path": ["train_a.pkl", "train_b.pkl"],
                "calibrating_pickle_path": ["calibration_a.pkl", "calibration_b.pkl"],
                "heldout_pickle_path": ["heldout_a.pkl", "heldout_b.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "calibration_penalties": [0.1, 1.0],
                "calibration_partitions": 1,
                "scoring_partitions": 1,
                "should_overwrite": False,
                "baselines": [],
            },
            indent=2,
        ),
    )

    config = load_evaluation_config(config_path)
    report = evaluate_from_config_path(config_path)
    report_path = write_evaluation_report_file(
        config_path=config_path,
        config=config,
        report=report,
    )
    predictions_path = tmp_path / "distance_cache" / "heldout_model_predictions.pkl"
    assert predictions_path.exists()
    predictions_path.unlink()

    exit_code = main([str(config_path)])

    assert exit_code == 0
    assert report_path.exists()
    assert predictions_path.exists()
    rebuilt_predictions = pd.read_pickle(predictions_path)
    assert set(rebuilt_predictions["model_name"].tolist()) == {"Priority Function"}
    assert rebuilt_predictions["heldout_subject_index"].nunique() == 24


def test_missing_baselines_only_returns_unreported_configured_baselines(tmp_path: Path) -> None:
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
                    {"name": "Mixture Learning", "alpha": 1.0},
                    {"name": "Independent Learning Scheme", "alpha": 1.0},
                ],
            },
            indent=2,
        ),
    )
    config = load_evaluation_config(config_path)
    existing_report = EvaluationReport(
        prio_function_path=priority_path,
        heldout_auc_roc=0.9,
        heldout_ancestry_evaluations=(),
        baseline_evaluations=(
            BaselineEvaluation(
                name="Mixture Learning",
                auc_roc=0.8,
                heldout_ancestry_evaluations=(),
            ),
        ),
    )

    missing = _missing_baselines(config.baselines, existing_report)

    assert len(missing) == 1
    assert missing[0].name == "Independent Learning Scheme"


def test_load_existing_report_and_merge_preserves_existing_main_result_and_appends_baseline(
    tmp_path: Path,
) -> None:
    priority_path = _write_text(
        tmp_path / "best_prio.py",
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 1.0\n",
    )
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "prio_function_path": str(priority_path),
            },
            indent=2,
        ),
    )
    config = load_evaluation_config(config_path)
    existing_report = EvaluationReport(
        prio_function_path=priority_path,
        heldout_auc_roc=0.91,
        heldout_ancestry_evaluations=(
            HeldoutAncestryEvaluation("AA", 10, 0.92),
        ),
        baseline_evaluations=(
            BaselineEvaluation("Mixture Learning", 0.87, ()),
        ),
    )
    additional_report = EvaluationReport(
        prio_function_path=priority_path,
        heldout_auc_roc=0.12,
        heldout_ancestry_evaluations=(),
        baseline_evaluations=(
            BaselineEvaluation("Independent Learning Scheme", 0.88, ()),
        ),
    )

    output_path = write_evaluation_report_file(
        config_path=config_path,
        config=config,
        report=existing_report,
    )
    loaded_report = load_existing_evaluation_report(output_path)
    merged_report = merge_evaluation_reports(loaded_report, additional_report)

    assert merged_report.heldout_auc_roc == 0.91
    assert merged_report.heldout_ancestry_evaluations[0].ancestry_group == "AA"
    assert [baseline.name for baseline in merged_report.baseline_evaluations] == [
        "Independent Learning Scheme",
        "Mixture Learning",
    ]
