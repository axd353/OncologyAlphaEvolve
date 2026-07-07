from __future__ import annotations

import json
from pathlib import Path

from funsearch_pipeline.orchestration.runner import run_experiment


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_funsearch_runner_creates_cycle_snapshots_and_resets_islands(tmp_path: Path) -> None:
    seed_path = _write_file(
        tmp_path / "seed_priority.py",
        "from __future__ import annotations\n"
        "from collections.abc import Sequence\n"
        "from typing import Any\n\n"
        "def priority(training_data: Any, ancestry_coordinate: Sequence[float], target_variant: Any) -> float:\n"
        "    return 0.5\n",
    )
    system_prompt_path = _write_file(
        tmp_path / "system_prompt.txt",
        "Return only a valid indented Python function body.\n",
    )
    output_root = tmp_path / "runs"
    config_path = _write_file(
        tmp_path / "config.json",
        json.dumps(
            {
                "experiment": {
                    "name": "smoke_test",
                    "main_output_dir": str(output_root),
                    "seed_priority_path": str(seed_path),
                    "function_to_evolve": "priority",
                    "max_cycles": 2,
                    "stop_after_no_improvement_cycles": 2,
                    "random_seed": 7,
                },
                "program_database": {
                    "functions_per_prompt": 2,
                    "num_islands": 4,
                    "cluster_sampling_temperature_init": 0.1,
                    "cluster_sampling_temperature_period": 100,
                },
                "sampler": {
                    "backend": "deterministic",
                    "system_prompt_path": str(system_prompt_path),
                    "model": "deterministic-model",
                    "candidates_per_island_per_cycle": 2,
                    "parallel_workers": 1,
                },
                "evaluator": {
                    "backend": "deterministic",
                    "metric": "synthetic",
                    "oracle_train_fraction": 0.8,
                    "preprocessed_dirname": "preprocessed",
                    "calibration_penalties": [1.0],
                },
                "logging": {
                    "level": "INFO",
                },
                "priority_tools": {
                    "module_names": [],
                },
            },
            indent=2,
        ),
    )

    experiment_dir = run_experiment(config_path)

    assert (experiment_dir / "main.log").exists()
    assert (experiment_dir / "program_db" / "bootstrap.pkl").exists()
    assert (experiment_dir / "cycle_0001" / "program_db_start.pkl").exists()
    assert (experiment_dir / "cycle_0001" / "program_db_end.pkl").exists()
    assert (experiment_dir / "cycle_0002" / "program_db_start.pkl").exists()

    first_island_second_prompt = (
        experiment_dir
        / "cycle_0001"
        / "sampler_outputs"
        / "island_000"
        / "sample_001"
        / "prompt.py"
    ).read_text()
    assert "def priority_v0" in first_island_second_prompt
    assert "def priority_v1" in first_island_second_prompt
    assert "def priority_v2" in first_island_second_prompt

    cycle_0001_end = json.loads(
        (experiment_dir / "cycle_0001" / "program_db_end_summary.json").read_text()
    )
    cycle_0002_start = json.loads(
        (experiment_dir / "cycle_0002" / "program_db_start_summary.json").read_text()
    )
    sampler_log = (
        experiment_dir / "cycle_0001" / "sampler_logs" / "island_000.log"
    ).read_text()

    assert all(island["num_programs"] == 3 for island in cycle_0001_end["islands"])
    assert any(island["num_programs"] == 1 for island in cycle_0002_start["islands"])
    assert "sampler_cpu pid=" in sampler_log
    assert "allowed_cpus=" in sampler_log
