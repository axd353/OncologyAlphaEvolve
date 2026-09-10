from __future__ import annotations

from pathlib import Path

import pandas as pd

from PostProcesingData.heldout_model_auc_ci import HeldoutAucCiConfig
from PostProcesingData.heldout_model_auc_ci import _build_auc_ci_summary
from PostProcesingData.heldout_model_auc_ci import write_all_auc_ci_summaries
from PostProcesingData.heldout_model_auc_ci import write_auc_ci_summary


def _write_prediction_frame(path: Path, *, model_name: str, model_slug: str, ancestry_group: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ancestry_group": [ancestry_group] * 8,
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "model_name": [model_name] * 8,
            "model_slug": [model_slug] * 8,
            "risk_score": [0.10, 0.25, 0.40, 0.45, 0.35, 0.55, 0.75, 0.90],
        }
    ).to_pickle(path)
    return path


def test_write_auc_ci_summary_writes_markdown_with_three_significant_digits(tmp_path: Path) -> None:
    precomputed_directory = tmp_path / "cache"
    by_model_dir = precomputed_directory / "heldout_model_predictions_by_model"
    _write_prediction_frame(
        by_model_dir / "priority_function.pkl",
        model_name="Priority Function",
        model_slug="priority_function",
        ancestry_group="AA",
    )
    _write_prediction_frame(
        by_model_dir / "mixture_learning.pkl",
        model_name="Mixture Learning",
        model_slug="mixture_learning",
        ancestry_group="AA",
    )
    config = HeldoutAucCiConfig(
        precomputed_directory=precomputed_directory,
        target_ancestry_group="AA",
        ci_level=0.95,
        bootstrap_iterations=20,
        random_seed=0,
        output_file_name="heldout_auc_ci_aa.md",
    )

    summary_frame = _build_auc_ci_summary(config)
    output_path = write_auc_ci_summary(config)

    markdown = output_path.read_text(encoding="utf-8")
    assert output_path.name == "heldout_auc_ci_aa.md"
    assert "# Heldout ROC AUC Confidence Intervals for AA" in markdown
    assert "| model_name | subject_count | auc_roc | ci_lower | ci_hi |" in markdown
    for column_name in ("auc_roc", "ci_lower", "ci_hi"):
        assert format(float(summary_frame.iloc[0][column_name]), ".3g") in markdown
    assert "prediction_pickle_path" not in markdown
    assert (precomputed_directory / "heldout_auc_ci_aa.png").exists()


def test_write_all_auc_ci_summaries_writes_one_file_per_ancestry(tmp_path: Path) -> None:
    precomputed_directory = tmp_path / "cache"
    by_model_dir = precomputed_directory / "heldout_model_predictions_by_model"
    _write_prediction_frame(
        by_model_dir / "priority_function_aa.pkl",
        model_name="Priority Function",
        model_slug="priority_function",
        ancestry_group="AA",
    )
    _write_prediction_frame(
        by_model_dir / "priority_function_ja.pkl",
        model_name="Priority Function",
        model_slug="priority_function",
        ancestry_group="JA",
    )

    output_paths = write_all_auc_ci_summaries(
        precomputed_directory=precomputed_directory,
        ci_level=0.95,
        bootstrap_iterations=10,
        allow_overwrite=True,
    )

    assert output_paths == (
        precomputed_directory / "heldout_auc_ci_aa.md",
        precomputed_directory / "heldout_auc_ci_ja.md",
    )
    assert all(path.exists() for path in output_paths)