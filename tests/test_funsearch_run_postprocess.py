from __future__ import annotations

from pathlib import Path
import json
import pickle

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
from PostProcesingData.funsearch_run_postprocess import write_cycle_best_priority_files
from funsearch_pipeline.config import ProgramDatabaseSettings
from funsearch_pipeline.program_database import CycleProgramsDatabase
from funsearch_pipeline.orchestration.runner import run_experiment


def _write_text(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def _write_cycle_snapshot(
    cycle_dir: Path,
    *,
    returned_score: float,
    num_islands: int = 1,
) -> Path:
    seed_program_text = (
        "from __future__ import annotations\n"
        "from collections.abc import Sequence\n"
        "from typing import Any\n\n"
        "def priority(training_data: Any, ancestry_coordinate: Sequence[float], target_variant: Any) -> float:\n"
        f"    return {returned_score:.6f}\n"
    )
    database = CycleProgramsDatabase.from_seed_program_text(
        ProgramDatabaseSettings(
            functions_per_prompt=2,
            num_islands=num_islands,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=100,
        ),
        seed_program_text,
        "priority",
    )
    database.register_seed({"mean": returned_score})
    snapshot_path = cycle_dir / "program_db_end.pkl"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("wb") as handle:
        pickle.dump(database, handle)
    return snapshot_path


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
    _write_cycle_snapshot(run_dir / "cycle_0001", returned_score=11.01)
    _write_cycle_snapshot(run_dir / "cycle_0002", returned_score=21.01)
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
    assert outputs.best_priority_paths == (
        run_dir / "cycle_0001" / "best_prio.py",
        run_dir / "cycle_0002" / "best_prio.py",
    )
    assert moved_logger_path.exists()
    assert not logger_path.exists()
    assert "# Best priority function at end of cycle_0001." in (
        run_dir / "cycle_0001" / "best_prio.py"
    ).read_text()
    assert "return 11.010000" in (run_dir / "cycle_0001" / "best_prio.py").read_text()
    assert "# Best priority function at end of cycle_0002." in (
        run_dir / "cycle_0002" / "best_prio.py"
    ).read_text()
    assert "return 21.010000" in (run_dir / "cycle_0002" / "best_prio.py").read_text()

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


def test_write_cycle_best_priority_files_materializes_best_program_per_cycle(
    tmp_path: Path,
) -> None:
    seed_path = _write_text(
        tmp_path / "seed_priority.py",
        "from __future__ import annotations\n"
        "from collections.abc import Sequence\n"
        "from typing import Any\n\n"
        "def priority(training_data: Any, ancestry_coordinate: Sequence[float], target_variant: Any) -> float:\n"
        "    return 0.5\n",
    )
    system_prompt_path = _write_text(
        tmp_path / "system_prompt.txt",
        "Return only a valid indented Python function body.\n",
    )
    output_root = tmp_path / "runs"
    config_path = _write_text(
        tmp_path / "config.json",
        json.dumps(
            {
                "experiment": {
                    "name": "postprocess_best_prio",
                    "main_output_dir": str(output_root),
                    "seed_priority_path": str(seed_path),
                    "function_to_evolve": "priority",
                    "max_cycles": 2,
                    "stop_after_no_improvement_cycles": 2,
                    "random_seed": 7,
                },
                "program_database": {
                    "functions_per_prompt": 2,
                    "num_islands": 2,
                    "cluster_sampling_temperature_init": 0.1,
                    "cluster_sampling_temperature_period": 100,
                },
                "sampler": {
                    "backend": "deterministic",
                    "system_prompt_path": str(system_prompt_path),
                    "model": "deterministic-model",
                    "candidates_per_island_per_cycle": 1,
                    "parallel_workers": 1,
                },
                "evaluator": {
                    "backend": "deterministic",
                    "metric": "synthetic",
                    "oracle_train_fraction": 0.8,
                    "preprocessed_dirname": "preprocessed",
                    "calibration_penalties": [1.0],
                },
                "logging": {"level": "INFO"},
                "priority_tools": {"module_names": []},
            },
            indent=2,
        ),
    )

    experiment_dir = run_experiment(config_path)
    (experiment_dir / "cycle_9999").mkdir()

    written_paths = write_cycle_best_priority_files(experiment_dir)

    assert written_paths == (
        experiment_dir / "cycle_0001" / "best_prio.py",
        experiment_dir / "cycle_0002" / "best_prio.py",
    )
    cycle_0001_best = (experiment_dir / "cycle_0001" / "best_prio.py").read_text()
    cycle_0002_best = (experiment_dir / "cycle_0002" / "best_prio.py").read_text()

    assert "# Best priority function at end of cycle_0001." in cycle_0001_best
    assert "def priority(" in cycle_0001_best
    assert "return 11.010000" in cycle_0001_best
    assert "# Best priority function at end of cycle_0002." in cycle_0002_best
    assert "def priority(" in cycle_0002_best
    assert "return 21.010000" in cycle_0002_best
    assert not (experiment_dir / "cycle_9999" / "best_prio.py").exists()