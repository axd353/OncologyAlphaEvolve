from __future__ import annotations

from pathlib import Path

import pandas as pd

from PostProcesingData.funsearch_run_postprocess import COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE
from PostProcesingData.funsearch_run_postprocess import EVALUATION_COMPLETED_COUNTS_PICKLE
from PostProcesingData.funsearch_run_postprocess import ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE
from PostProcesingData.funsearch_run_postprocess import VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE
from PostProcesingData.funsearch_run_postprocess import count_completed_priority_functions_in_sampler_log
from PostProcesingData.funsearch_run_postprocess import count_empty_completions_in_sampler_log
from PostProcesingData.funsearch_run_postprocess import count_evaluation_completions_in_sampler_log
from PostProcesingData.funsearch_run_postprocess import count_island_best_improvements_in_sampler_log
from PostProcesingData.funsearch_run_postprocess import count_total_sampler_attempts_in_sampler_log
from PostProcesingData.funsearch_run_postprocess import count_validation_passes_in_sampler_log
from PostProcesingData.funsearch_run_postprocess import extract_sampler_log_metrics
from PostProcesingData.funsearch_run_postprocess import postprocess_funsearch_run


def _write_text(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def test_extract_sampler_log_metrics_counts_registered_attempt_once(tmp_path: Path) -> None:
    log_path = _write_text(
        tmp_path / "cycle_0001" / "sampler_logs" / "island_000.log",
        "\n".join(
            [
                "cycle=1 island=0 sampler_start",
                "sample_index=0 attempt=1 rejected=empty_completion",
                "sample_index=0 attempt=2 rejected=invalid_priority_function error=ValueError: bad",
                "sample_index=0 attempt=3 rejected=evaluation_failed",
                "sample_index=0 attempt=4 registered=true first_success=true full_priority_function_begin",
                "  return 0.5",
                "sample_index=0 attempt=4 registered=true after_attempts=4 better_than_present_best=True full_priority_function_end",
                "sample_index=1 attempt=1 registered=true after_attempts=1 better_than_present_best=False",
            ]
        )
        + "\n",
    )

    metrics = extract_sampler_log_metrics(log_path)

    assert metrics.cycle_index == 1
    assert metrics.island_id == 0
    assert metrics.completed_priority_function_count == 4
    assert metrics.empty_completion_count == 1
    assert metrics.total_sampler_attempt_count == 5
    assert metrics.validated_priority_function_count == 3
    assert metrics.evaluation_completed_count == 2
    assert metrics.island_best_improvement_count == 1
    assert count_completed_priority_functions_in_sampler_log(log_path) == 4
    assert count_empty_completions_in_sampler_log(log_path) == 1
    assert count_total_sampler_attempts_in_sampler_log(log_path) == 5
    assert count_validation_passes_in_sampler_log(log_path) == 3
    assert count_evaluation_completions_in_sampler_log(log_path) == 2
    assert count_island_best_improvements_in_sampler_log(log_path) == 1


def test_postprocess_funsearch_run_moves_logger_and_writes_pickles(tmp_path: Path) -> None:
    run_dir = tmp_path / "prio_func_disc_runs" / "oracle_priority_20260716_050704"
    _write_text(
        run_dir / "config.used.json",
        "\n".join(
            [
                "{",
                '  "sampler": {',
                '    "candidates_per_island_per_cycle": 6',
                "  }",
                "}",
            ]
        )
        + "\n",
    )
    _write_text(
        run_dir / "cycle_0001" / "sampler_logs" / "island_000.log",
        "\n".join(
            [
                f"cycle=1 island=0 experiment_dir={run_dir} sampler_start",
                "sample_index=0 attempt=1 rejected=empty_completion",
                "sample_index=0 attempt=2 rejected=invalid_priority_function error=ValueError: bad",
                "sample_index=0 attempt=3 rejected=evaluation_failed",
                "sample_index=0 attempt=4 registered=true after_attempts=4 better_than_present_best=True",
            ]
        )
        + "\n",
    )
    _write_text(
        run_dir / "cycle_0001" / "sampler_logs" / "island_001.log",
        "\n".join(
            [
                f"cycle=1 island=1 experiment_dir={run_dir} sampler_start",
                "sample_index=0 attempt=1 registered=true after_attempts=1 better_than_present_best=False",
                "sample_index=1 attempt=1 rejected=invalid_priority_function error=ValueError: bad",
            ]
        )
        + "\n",
    )
    _write_text(
        run_dir / "cycle_0002" / "sampler_logs" / "island_000.log",
        "\n".join(
            [
                f"cycle=2 island=0 experiment_dir={run_dir} sampler_start",
                "sample_index=0 attempt=1 registered=true after_attempts=1 better_than_present_best=True",
                "sample_index=1 attempt=1 registered=true after_attempts=1 better_than_present_best=False",
            ]
        )
        + "\n",
    )
    logger_path = _write_text(
        tmp_path / "prio_func_disc_runs" / "logger_20260716_010700.log",
        "\n".join(
            [
                f"Created experiment directory {run_dir}",
                str(run_dir),
            ]
        )
        + "\n",
    )

    outputs = postprocess_funsearch_run(run_dir)

    moved_logger_path = run_dir / logger_path.name
    assert outputs.logger_path == moved_logger_path
    assert moved_logger_path.exists()
    assert not logger_path.exists()

    completed_counts = pd.read_pickle(run_dir / COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE)
    validated_counts = pd.read_pickle(run_dir / VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE)
    evaluation_completed_counts = pd.read_pickle(run_dir / EVALUATION_COMPLETED_COUNTS_PICKLE)
    improvement_counts = pd.read_pickle(run_dir / ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE)

    assert completed_counts.to_dict("records") == [
        {
            "cycle_index": 1,
            "island_id": 0,
            "completed_priority_function_count": 3,
            "empty_completion_count": 1,
            "total_sampler_attempt_count": 4,
            "configured_candidates_per_island_per_cycle": 6,
        },
        {
            "cycle_index": 1,
            "island_id": 1,
            "completed_priority_function_count": 2,
            "empty_completion_count": 0,
            "total_sampler_attempt_count": 2,
            "configured_candidates_per_island_per_cycle": 6,
        },
        {
            "cycle_index": 2,
            "island_id": 0,
            "completed_priority_function_count": 2,
            "empty_completion_count": 0,
            "total_sampler_attempt_count": 2,
            "configured_candidates_per_island_per_cycle": 6,
        },
    ]
    assert validated_counts.to_dict("records") == [
        {"cycle_index": 1, "island_id": 0, "validated_priority_function_count": 2},
        {"cycle_index": 1, "island_id": 1, "validated_priority_function_count": 1},
        {"cycle_index": 2, "island_id": 0, "validated_priority_function_count": 2},
    ]
    assert evaluation_completed_counts.to_dict("records") == [
        {"cycle_index": 1, "island_id": 0, "evaluation_completed_count": 1},
        {"cycle_index": 1, "island_id": 1, "evaluation_completed_count": 1},
        {"cycle_index": 2, "island_id": 0, "evaluation_completed_count": 2},
    ]
    assert improvement_counts.to_dict("records") == [
        {"cycle_index": 1, "island_best_improvement_count": 1},
        {"cycle_index": 2, "island_best_improvement_count": 1},
    ]