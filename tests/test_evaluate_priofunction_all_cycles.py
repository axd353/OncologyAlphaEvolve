from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

import PostProcesingData.evaluate_priofunction_all_cycles as all_cycles_module
from tests.test_procedure2_evaluator import _make_synthetic_oracle_frame


def _write_pickle(path: Path, data_frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(path)
    return path


def _write_text(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _write_tracking_pickle(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(path)
    return path


def test_load_all_cycles_evaluation_config_normalizes_target_ancestry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "run_dir": str(run_dir),
                "target_ancestry_group": "aa",
                "supported_ancestry_groups": ["aa", "ja", "la"],
            },
            indent=2,
        ),
    )

    config = all_cycles_module.load_all_cycles_evaluation_config(config_path)

    assert config.run_dir == run_dir
    assert config.target_ancestry_group == "AA"
    assert config.report_file_name == "best_prio.all_cycles.AA.evaluation_report.json"


def test_evaluate_all_cycles_filters_target_ancestry_and_reuses_duplicate_cycles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    cycle_0001 = _write_text(
        run_dir / "cycle_0001" / "best_prio.py",
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 1.0\n",
    )
    _write_text(
        run_dir / "cycle_0002" / "best_prio.py",
        cycle_0001.read_text(encoding="utf-8"),
    )
    _write_text(
        run_dir / "cycle_0003" / "best_prio.py",
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 2.0\n",
    )

    _write_pickle(tmp_path / "train.pkl", _make_synthetic_oracle_frame(4))
    _write_pickle(tmp_path / "calibration.pkl", _make_synthetic_oracle_frame(4, offset=4))
    _write_pickle(tmp_path / "heldout.pkl", _make_synthetic_oracle_frame(4, offset=8))
    _write_tracking_pickle(
        tmp_path / "output_row_tracking.pkl",
        [
            {
                "output_pickle_name": "train.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "train_AA.pkl" if row_number < 2 else "train_JA.pkl",
                "source_pickle_path": f"/tmp/train_{row_number}.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(4)
        ]
        + [
            {
                "output_pickle_name": "calibration.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "test_AA.pkl" if row_number < 2 else "test_JA.pkl",
                "source_pickle_path": f"/tmp/calibration_{row_number}.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(4)
        ]
        + [
            {
                "output_pickle_name": "heldout.pkl",
                "output_row_number": row_number,
                "source_pickle_name": "heldout_AA.pkl" if row_number < 2 else "heldout_JA.pkl",
                "source_pickle_path": f"/tmp/heldout_{row_number}.pkl",
                "source_row_number": row_number,
            }
            for row_number in range(4)
        ],
    )
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "run_dir": str(run_dir),
                "target_ancestry_group": "AA",
                "training_pickle_path": ["train.pkl"],
                "calibrating_pickle_path": ["calibration.pkl"],
                "heldout_pickle_path": ["heldout.pkl"],
                "output_row_tracking_path": "output_row_tracking.pkl",
                "supported_ancestry_groups": ["AA", "JA"],
                "distance_cache_enabled": False,
                "calibration_partitions": 1,
                "scoring_partitions": 1,
            },
            indent=2,
        ),
    )
    config = all_cycles_module.load_all_cycles_evaluation_config(config_path)

    calibration_calls: list[tuple[int, int, float]] = []
    scoring_calls: list[tuple[int, int, float]] = []

    def fake_build_calibration_oracle_feature_matrix(**kwargs):
        constant = 1.0 if "return 1.0" in kwargs["candidate_source"] else 2.0
        calibration_calls.append(
            (len(kwargs["training_data"]), len(kwargs["calibration_data"]), constant)
        )
        return np.full((len(kwargs["calibration_data"]), 1), constant, dtype=float)

    def fake_build_scoring_oracle_feature_matrix(**kwargs):
        constant = 1.0 if "return 1.0" in kwargs["candidate_source"] else 2.0
        scoring_calls.append(
            (len(kwargs["training_data"]), len(kwargs["scoring_data"]), constant)
        )
        scores = np.linspace(constant, constant + 0.25, len(kwargs["scoring_data"]))
        return scores.reshape(-1, 1)

    class FakeCalibrationModel:
        def __init__(self, penalty: float) -> None:
            self.penalty = penalty

    monkeypatch.setattr(
        all_cycles_module,
        "_build_calibration_oracle_feature_matrix",
        fake_build_calibration_oracle_feature_matrix,
    )
    monkeypatch.setattr(
        all_cycles_module,
        "_build_scoring_oracle_feature_matrix",
        fake_build_scoring_oracle_feature_matrix,
    )
    monkeypatch.setattr(
        all_cycles_module,
        "_fit_best_calibration_model",
        lambda **kwargs: FakeCalibrationModel(kwargs["penalties"][0]),
    )
    monkeypatch.setattr(
        all_cycles_module,
        "_predict_linear_score",
        lambda model, *, oracle_features, covariates: oracle_features[:, 0],
    )
    monkeypatch.setattr(
        all_cycles_module,
        "_safe_roc_auc",
        lambda labels, scores: float(np.mean(np.asarray(scores, dtype=float))),
    )

    report = all_cycles_module.evaluate_all_cycles(config)

    assert report.target_ancestry_group == "AA"
    assert report.calibration_subject_count == 2
    assert report.heldout_subject_count == 2
    assert len(report.baseline_evaluations) == 0
    assert len(report.cycle_evaluations) == 3
    assert [cycle.reused_from_cycle_index for cycle in report.cycle_evaluations] == [None, 1, None]
    assert len(calibration_calls) == 2
    assert len(scoring_calls) == 2
    assert calibration_calls == [(4, 2, 1.0), (4, 2, 2.0)]
    assert scoring_calls == [(8, 2, 1.0), (8, 2, 2.0)]
    assert report.cycle_evaluations[0].heldout_auc_roc == report.cycle_evaluations[1].heldout_auc_roc
    assert report.cycle_evaluations[0].heldout_auc_roc != report.cycle_evaluations[2].heldout_auc_roc
    assert report.cycle_evaluations[0].heldout_ancestry_evaluations[0].ancestry_group == "AA"
    assert report.cycle_evaluations[0].heldout_ancestry_evaluations[0].subject_count == 2
    assert report.cycle_evaluations[2].heldout_ancestry_evaluations[0].ancestry_group == "AA"
    assert report.cycle_evaluations[2].heldout_ancestry_evaluations[0].subject_count == 2