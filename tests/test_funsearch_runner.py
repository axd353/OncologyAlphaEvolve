from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from funsearch_pipeline.orchestration import runner as runner_module
from funsearch_pipeline.config import load_pipeline_config
from funsearch_pipeline.orchestration.runner import resume_experiment
from funsearch_pipeline.orchestration.runner import run_experiment
from funsearch_pipeline.evaluation.interfaces import EvaluatedCandidate
from funsearch_pipeline.evaluation.interfaces import PairScore
from funsearch_pipeline.program_database.database import CandidateProgram


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
    assert (
        experiment_dir / "cycle_0001" / "sampler_outputs" / "island_000" / "island_progress.pkl"
    ).exists()
    assert "sampler_cpu pid=" in sampler_log
    assert "allowed_cpus=" in sampler_log


def test_run_experiment_writes_resolved_config_for_resume(tmp_path: Path) -> None:
    config_root = tmp_path / "relative_config"
    seed_path = _write_file(
        config_root / "seed_priority.py",
        "from __future__ import annotations\n"
        "from collections.abc import Sequence\n"
        "from typing import Any\n\n"
        "def priority(training_data: Any, ancestry_coordinate: Sequence[float], target_variant: Any) -> float:\n"
        "    return 0.5\n",
    )
    system_prompt_path = _write_file(
        config_root / "system_prompt.txt",
        "Return only a valid indented Python function body.\n",
    )
    output_root = tmp_path / "resolved_runs"
    config_path = _write_file(
        config_root / "config.json",
        json.dumps(
            {
                "experiment": {
                    "name": "resolved_config_test",
                    "main_output_dir": str(output_root),
                    "seed_priority_path": "seed_priority.py",
                    "function_to_evolve": "priority",
                    "max_cycles": 1,
                    "stop_after_no_improvement_cycles": 1,
                    "random_seed": 5,
                },
                "program_database": {
                    "functions_per_prompt": 2,
                    "num_islands": 2,
                    "cluster_sampling_temperature_init": 0.1,
                    "cluster_sampling_temperature_period": 100,
                },
                "sampler": {
                    "backend": "deterministic",
                    "system_prompt_path": "system_prompt.txt",
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
    resumed_config = load_pipeline_config(experiment_dir / "config.used.json")

    assert resumed_config.experiment.seed_priority_path == seed_path.resolve()
    assert resumed_config.sampler.system_prompt_path == system_prompt_path.resolve()


def test_island_sampler_retries_failed_evaluation_before_registering(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    output_root = tmp_path / "retry_runs"
    config_path = _write_file(
        tmp_path / "config.json",
        json.dumps(
            {
                "experiment": {
                    "name": "retry_test",
                    "main_output_dir": str(output_root),
                    "seed_priority_path": str(seed_path),
                    "function_to_evolve": "priority",
                    "max_cycles": 1,
                    "stop_after_no_improvement_cycles": 1,
                    "random_seed": 11,
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

    class FakeEvaluator:
        def __init__(self) -> None:
            self.calls = 0

        def prepare(self, experiment_dir: Path) -> None:
            return None

        def evaluate_candidate(self, candidate: CandidateProgram):
            self.calls += 1
            if self.calls == 1:
                return None
            return EvaluatedCandidate(
                candidate=candidate,
                pair_scores=(PairScore(name="pair", score=1.0),),
                reduced_score=1.0,
                metadata={},
            )

    fake_evaluator = FakeEvaluator()
    monkeypatch.setattr(
        "funsearch_pipeline.sampling.island_sampler.build_evaluator",
        lambda *args, **kwargs: fake_evaluator,
    )

    experiment_dir = run_experiment(config_path)
    island_log = (experiment_dir / "cycle_0001" / "sampler_logs" / "island_000.log").read_text()

    assert "sample_index=0 attempt=1 rejected=evaluation_failed" in island_log
    assert "sample_index=0 attempt=2 registered=true after_attempts=2" in island_log
    assert "no_priority_function_generated" not in island_log


def test_resume_experiment_reruns_latest_incomplete_cycle_without_touching_completed_cycles(
    tmp_path: Path,
) -> None:
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
    output_root = tmp_path / "resume_runs"
    config_path = _write_file(
        tmp_path / "config.json",
        json.dumps(
            {
                "experiment": {
                    "name": "resume_test",
                    "main_output_dir": str(output_root),
                    "seed_priority_path": str(seed_path),
                    "function_to_evolve": "priority",
                    "max_cycles": 2,
                    "stop_after_no_improvement_cycles": 2,
                    "random_seed": 19,
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
    cycle_0001_summary_before = json.loads(
        (experiment_dir / "cycle_0001" / "cycle_summary.json").read_text()
    )
    stale_marker = "stale resume marker"
    (experiment_dir / "cycle_0002" / "sampler_logs" / "island_000.log").write_text(
        stale_marker
    )
    _write_file(
        experiment_dir / "cycle_0002" / "sampler_outputs" / "stale.txt",
        stale_marker,
    )
    (experiment_dir / "cycle_0002" / "program_db_end.pkl").unlink()
    (experiment_dir / "cycle_0002" / "program_db_end_summary.json").unlink()
    (experiment_dir / "cycle_0002" / "cycle_summary.json").unlink()

    resumed_dir = resume_experiment(experiment_dir)

    assert resumed_dir == experiment_dir
    assert json.loads((experiment_dir / "cycle_0001" / "cycle_summary.json").read_text()) == (
        cycle_0001_summary_before
    )
    assert stale_marker not in (
        experiment_dir / "cycle_0002" / "sampler_logs" / "island_000.log"
    ).read_text()
    assert not (experiment_dir / "cycle_0002" / "sampler_outputs" / "stale.txt").exists()
    assert (experiment_dir / "cycle_0002" / "program_db_end.pkl").exists()
    assert (experiment_dir / "cycle_0002" / "cycle_summary.json").exists()
    assert "Resuming experiment directory" in (experiment_dir / "main.log").read_text()


def test_resume_experiment_rejects_completed_run(tmp_path: Path) -> None:
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
    output_root = tmp_path / "completed_runs"
    config_path = _write_file(
        tmp_path / "config.json",
        json.dumps(
            {
                "experiment": {
                    "name": "completed_resume_test",
                    "main_output_dir": str(output_root),
                    "seed_priority_path": str(seed_path),
                    "function_to_evolve": "priority",
                    "max_cycles": 1,
                    "stop_after_no_improvement_cycles": 1,
                    "random_seed": 23,
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

    with pytest.raises(ValueError, match="already completed"):
        resume_experiment(experiment_dir)


def test_run_sampling_phase_keeps_checkpointed_island_progress_after_timeout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeFuture:
        def __init__(self, result: runner_module.IslandSamplerResult | None = None) -> None:
            self._result = result

        def result(self) -> runner_module.IslandSamplerResult:
            if self._result is None:
                raise AssertionError("Timed-out future result should not be requested.")
            return self._result

    class FakeExecutor:
        def __init__(self) -> None:
            self._processes = {
                1001: SimpleNamespace(pid=1001),
                1002: SimpleNamespace(pid=1002),
            }
            self.shutdown_calls: list[tuple[bool, bool]] = []

        def submit(self, fn, request):
            return future_by_request[id(request)]

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            self.shutdown_calls.append((wait, cancel_futures))

    fast_request = SimpleNamespace(
        cycle_index=6,
        island_shard=SimpleNamespace(island_id=0, state="mutated"),
        output_dir=tmp_path / "island_000",
        log_path=tmp_path / "island_000.log",
    )
    slow_original_shard = SimpleNamespace(island_id=1, state="original")
    slow_checkpoint_shard = SimpleNamespace(island_id=1, state="checkpointed")
    slow_request = SimpleNamespace(
        cycle_index=6,
        island_shard=slow_original_shard,
        output_dir=tmp_path / "island_001",
        log_path=tmp_path / "island_001.log",
    )
    fast_result = runner_module.IslandSamplerResult(
        cycle_index=6,
        island_id=0,
        island_shard=SimpleNamespace(island_id=0, state="completed"),
        generated_candidates=4,
        accepted_candidates=3,
    )
    fast_future = FakeFuture(fast_result)
    slow_future = FakeFuture()
    slow_request.output_dir.mkdir(parents=True, exist_ok=True)
    with (slow_request.output_dir / "island_progress.pkl").open("wb") as handle:
        pickle.dump(
            runner_module.IslandSamplerResult(
                cycle_index=6,
                island_id=1,
                island_shard=slow_checkpoint_shard,
                generated_candidates=5,
                accepted_candidates=2,
            ),
            handle,
        )
    future_by_request = {
        id(fast_request): fast_future,
        id(slow_request): slow_future,
    }
    fake_executor = FakeExecutor()

    monkeypatch.setattr(
        runner_module,
        "ProcessPoolExecutor",
        lambda max_workers: fake_executor,
    )

    def fake_wait(pending, timeout=None, return_when=None):
        if pending == {fast_future, slow_future}:
            return {fast_future}, {slow_future}
        raise AssertionError("wait should not be called again after the timeout fires.")

    monotonic_values = iter((100.0, 161.0))
    aborted_executors: list[FakeExecutor] = []

    monkeypatch.setattr(runner_module, "wait", fake_wait)
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        runner_module,
        "_kill_sampler_executor_process_groups",
        lambda executor, logger: aborted_executors.append(executor),
    )

    results = runner_module._run_sampling_phase(
        [fast_request, slow_request],
        parallel_workers=2,
        logger=logging.getLogger("funsearch_runner_test"),
        last_island_abort_after_minutes=1.0,
    )

    results_by_island = {result.island_id: result for result in results}

    assert results_by_island[0] == fast_result
    assert results_by_island[1].island_shard.state == slow_checkpoint_shard.state
    assert results_by_island[1].generated_candidates == 5
    assert results_by_island[1].accepted_candidates == 2
    assert aborted_executors == [fake_executor]
    assert fake_executor.shutdown_calls == [(True, True)]
    assert "sampler_aborted reason=last_island_timeout" in slow_request.log_path.read_text()
    assert "checkpoint_recovered=True" in slow_request.log_path.read_text()
