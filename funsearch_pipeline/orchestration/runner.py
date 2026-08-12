from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import pickle
import shutil

import numpy as np

from funsearch_pipeline.config import PipelineConfig
from funsearch_pipeline.config import load_pipeline_config
from funsearch_pipeline.evaluation import build_evaluator
from funsearch_pipeline.logging_utils import configure_main_logger
from funsearch_pipeline.program_database import CycleProgramsDatabase
from funsearch_pipeline.program_database import IslandShard
from funsearch_pipeline.sampling import IslandSamplerRequest
from funsearch_pipeline.sampling import IslandSamplerResult
from funsearch_pipeline.sampling import run_island_sampler


@dataclass(frozen=True)
class CycleSummary:
    cycle_index: int
    start_best_score: float | None
    end_best_score: float | None
    accepted_candidates: int
    generated_candidates: int


@dataclass(frozen=True)
class ResumeState:
    start_cycle_index: int
    database_snapshot_path: Path
    reuse_existing_cycle_dir: bool
    stagnation_cycles: int


def _create_experiment_dir(config: PipelineConfig) -> Path:
    """Create the reproducible experiment output directory.

    Input:
        config: Parsed pipeline config containing experiment name and main
            output directory.

    Output:
        Newly created timestamped experiment directory.
    """

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    experiment_dir = config.experiment.main_output_dir / f"{config.experiment.name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def _write_cycle_summary(cycle_dir: Path, summary: CycleSummary) -> None:
    """Write one cycle's high-level accounting summary.

    Input:
        cycle_dir: Directory for the current cycle.
        summary: Candidate counts and best scores for the cycle.

    Output:
        Writes `cycle_summary.json` under `cycle_dir`.
    """

    (cycle_dir / "cycle_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True)
    )


def _cycle_dir(experiment_dir: Path, cycle_index: int) -> Path:
    return experiment_dir / f"cycle_{cycle_index:04d}"


def _list_existing_cycle_indices(experiment_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in experiment_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("cycle_"):
            continue
        suffix = path.name.removeprefix("cycle_")
        if not suffix.isdigit():
            continue
        indices.append(int(suffix))
    return sorted(indices)


def _cycle_is_complete(cycle_dir: Path) -> bool:
    required_paths = (
        cycle_dir / "program_db_end.pkl",
        cycle_dir / "program_db_end_summary.json",
        cycle_dir / "cycle_summary.json",
    )
    return all(path.exists() for path in required_paths)


def _load_cycle_summary(cycle_dir: Path) -> CycleSummary:
    raw_summary = json.loads((cycle_dir / "cycle_summary.json").read_text())
    return CycleSummary(
        cycle_index=int(raw_summary["cycle_index"]),
        start_best_score=(
            float(raw_summary["start_best_score"])
            if raw_summary["start_best_score"] is not None
            else None
        ),
        end_best_score=(
            float(raw_summary["end_best_score"])
            if raw_summary["end_best_score"] is not None
            else None
        ),
        accepted_candidates=int(raw_summary["accepted_candidates"]),
        generated_candidates=int(raw_summary["generated_candidates"]),
    )


def _compute_stagnation_cycles(experiment_dir: Path, completed_cycles: int) -> int:
    stagnation_cycles = 0
    for cycle_index in range(1, completed_cycles + 1):
        summary = _load_cycle_summary(_cycle_dir(experiment_dir, cycle_index))
        improved = (
            summary.start_best_score is None
            or summary.end_best_score is not None
            and summary.end_best_score > summary.start_best_score + 1e-12
        )
        if improved:
            stagnation_cycles = 0
        else:
            stagnation_cycles += 1
    return stagnation_cycles


def _load_database_snapshot(snapshot_path: Path) -> CycleProgramsDatabase:
    with snapshot_path.open("rb") as handle:
        database = pickle.load(handle)
    if not isinstance(database, CycleProgramsDatabase):
        raise TypeError(
            f"Snapshot {snapshot_path} did not contain a CycleProgramsDatabase instance."
        )
    return database


def _cleanup_incomplete_cycle_dir(cycle_dir: Path) -> None:
    for child_name in (
        "sampler_logs",
        "sampler_outputs",
        "program_db_end.pkl",
        "program_db_end_summary.json",
        "cycle_summary.json",
        "best_prio.py",
    ):
        path = cycle_dir / child_name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _build_resume_state(experiment_dir: Path, config: PipelineConfig) -> ResumeState:
    existing_cycle_indices = _list_existing_cycle_indices(experiment_dir)
    if existing_cycle_indices:
        expected_indices = list(range(1, existing_cycle_indices[-1] + 1))
        if existing_cycle_indices != expected_indices:
            raise ValueError(
                "Cycle directories must be contiguous from cycle_0001 to the latest cycle to "
                "resume safely."
            )

    completed_cycles = 0
    for cycle_index in existing_cycle_indices:
        cycle_dir = _cycle_dir(experiment_dir, cycle_index)
        start_snapshot_path = cycle_dir / "program_db_start.pkl"
        if not start_snapshot_path.exists():
            raise ValueError(
                f"Cannot resume because {start_snapshot_path} is missing."
            )
        if _cycle_is_complete(cycle_dir):
            completed_cycles = cycle_index
            continue

        if cycle_index != existing_cycle_indices[-1]:
            raise ValueError(
                "Only the latest started cycle may be incomplete when resuming a run."
            )
        return ResumeState(
            start_cycle_index=cycle_index,
            database_snapshot_path=start_snapshot_path,
            reuse_existing_cycle_dir=True,
            stagnation_cycles=_compute_stagnation_cycles(experiment_dir, cycle_index - 1),
        )

    bootstrap_snapshot_path = experiment_dir / "program_db" / "bootstrap.pkl"
    if not bootstrap_snapshot_path.exists():
        raise ValueError(
            f"Cannot resume because {bootstrap_snapshot_path} is missing."
        )

    if completed_cycles == 0 and not existing_cycle_indices:
        return ResumeState(
            start_cycle_index=1,
            database_snapshot_path=bootstrap_snapshot_path,
            reuse_existing_cycle_dir=False,
            stagnation_cycles=0,
        )

    final_stagnation_cycles = _compute_stagnation_cycles(experiment_dir, completed_cycles)
    if (
        completed_cycles >= config.experiment.max_cycles
        or final_stagnation_cycles >= config.experiment.stop_after_no_improvement_cycles
    ):
        raise ValueError("This run already completed; there is nothing to resume.")

    raise ValueError(
        "This run does not contain an incomplete started cycle to resume. "
        "Resume is currently supported from bootstrap or from the latest started cycle's "
        "program_db_start snapshot."
    )


def _run_cycles(
    config: PipelineConfig,
    experiment_dir: Path,
    database: CycleProgramsDatabase,
    logger,
    *,
    start_cycle_index: int,
    initial_stagnation_cycles: int,
    reuse_existing_cycle_dir: bool,
) -> Path:
    system_prompt = config.sampler.system_prompt_path.read_text()
    stagnation_cycles = initial_stagnation_cycles
    for cycle_index in range(start_cycle_index, config.experiment.max_cycles + 1):
        cycle_dir = _cycle_dir(experiment_dir, cycle_index)
        if cycle_index == start_cycle_index and reuse_existing_cycle_dir:
            cycle_dir.mkdir(parents=True, exist_ok=True)
        else:
            cycle_dir.mkdir(parents=True, exist_ok=False)
        start_best_score = database.global_best_score()
        database.save_snapshot(
            cycle_dir / "program_db_start.pkl",
            cycle_dir / "program_db_start_summary.json",
            extra_metadata={"cycle_index": cycle_index, "stage": "start"},
        )

        island_shards = database.export_island_shards()
        sampler_requests = _build_island_sampler_requests(
            config=config,
            cycle_index=cycle_index,
            experiment_dir=experiment_dir,
            cycle_dir=cycle_dir,
            system_prompt=system_prompt,
            database=database,
            island_shards=island_shards,
        )
        sampler_results = _run_sampling_phase(
            sampler_requests,
            config.sampler.parallel_workers,
            logger,
        )
        database.combine_island_shards([result.island_shard for result in sampler_results])
        generated_candidates = sum(result.generated_candidates for result in sampler_results)
        accepted_candidates = sum(result.accepted_candidates for result in sampler_results)

        end_best_score = database.global_best_score()
        database.save_snapshot(
            cycle_dir / "program_db_end.pkl",
            cycle_dir / "program_db_end_summary.json",
            extra_metadata={"cycle_index": cycle_index, "stage": "end"},
        )
        _write_cycle_summary(
            cycle_dir,
            CycleSummary(
                cycle_index=cycle_index,
                start_best_score=start_best_score,
                end_best_score=end_best_score,
                accepted_candidates=accepted_candidates,
                generated_candidates=generated_candidates,
            ),
        )

        improved = (
            start_best_score is None
            or end_best_score is not None and end_best_score > start_best_score + 1e-12
        )
        if improved:
            stagnation_cycles = 0
        else:
            stagnation_cycles += 1
        logger.info(
            "Completed cycle %d with %d accepted candidates. start_best=%s end_best=%s stagnation=%d",
            cycle_index,
            accepted_candidates,
            start_best_score,
            end_best_score,
            stagnation_cycles,
        )

        if stagnation_cycles >= config.experiment.stop_after_no_improvement_cycles:
            logger.info(
                "Stopping after %d stagnant cycles.",
                stagnation_cycles,
            )
            break

        if cycle_index < config.experiment.max_cycles:
            database.reset_weak_islands()
            logger.info("Reset the weaker half of islands after cycle %d.", cycle_index)

    return experiment_dir


def _build_island_sampler_requests(
    config: PipelineConfig,
    cycle_index: int,
    experiment_dir: Path,
    cycle_dir: Path,
    system_prompt: str,
    database: CycleProgramsDatabase,
    island_shards: list[IslandShard],
) -> list[IslandSamplerRequest]:
    """Build one sampler worker request per island shard.

    Input:
        config: Parsed pipeline config.
        cycle_index: Current evolution cycle.
        experiment_dir: Root experiment directory containing copied config and
            prepared evaluator artifacts.
        cycle_dir: Directory for this cycle's logs and sampler artifacts.
        system_prompt: LLM system prompt text loaded from config.
        database: Parent program database wrapper that owns the shared template.
        island_shards: Deep-copied island shards exported from the database at
            the start of the cycle.

    Output:
        List of `IslandSamplerRequest` objects, one per island worker.
    """

    requests: list[IslandSamplerRequest] = []
    for island_shard in island_shards:
        requests.append(
            IslandSamplerRequest(
                cycle_index=cycle_index,
                island_shard=island_shard,
                template=database.template,
                function_to_evolve=database.function_to_evolve,
                sampler_settings=config.sampler,
                evaluator_settings=config.evaluator,
                experiment_random_seed=config.experiment.random_seed,
                logging_level=config.logging.level,
                experiment_dir=experiment_dir,
                system_prompt=system_prompt,
                output_dir=cycle_dir / "sampler_outputs" / f"island_{island_shard.island_id:03d}",
                log_path=cycle_dir / "sampler_logs" / f"island_{island_shard.island_id:03d}.log",
            )
        )
    return requests


def _run_sampling_phase(
    requests: list[IslandSamplerRequest],
    parallel_workers: int,
    logger,
) -> list[IslandSamplerResult]:
    """Run one cycle's per-island sampler workers.

    Input:
        requests: One `IslandSamplerRequest` per island shard.
        parallel_workers: Maximum process count for concurrent samplers.
        logger: Main experiment logger used for worker failures.

    Output:
        Completed sampler results containing updated island shards. Evaluation
        and local registration have already happened inside each worker.
    """

    if parallel_workers == 1:
        results: list[IslandSamplerResult] = []
        for request in requests:
            results.append(run_island_sampler(request))
        return results

    results: list[IslandSamplerResult] = []
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(run_island_sampler, request): request for request in requests}
        for future in as_completed(futures):
            request = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.exception(
                    "Sampling failed for island %d during cycle %d: %s",
                    request.island_shard.island_id,
                    request.cycle_index,
                    exc,
                )
    return results


def run_experiment(config_path: str | Path) -> Path:
    """Run a full FunSearch priority-function experiment.

    Input:
        config_path: Path to a JSON config file. The resolved config is copied
            into the experiment directory as `config.used.json`.

    Output:
        Path to the created experiment directory containing the main log,
        program database snapshots, cycle summaries, sampler logs, and sampler
        artifacts.
    """

    config = load_pipeline_config(config_path)
    experiment_dir = _create_experiment_dir(config)
    shutil.copy2(config.config_path, experiment_dir / "config.used.json")
    logger = configure_main_logger(experiment_dir / "main.log", config.logging.level)

    np.random.seed(config.experiment.random_seed)
    logger.info("Created experiment directory %s", experiment_dir)

    seed_program_text = config.experiment.seed_priority_path.read_text()
    database = CycleProgramsDatabase.from_seed_program_text(
        settings=config.program_database,
        seed_program_text=seed_program_text,
        function_to_evolve=config.experiment.function_to_evolve,
    )

    evaluator = build_evaluator(
        config.evaluator,
        function_name=config.experiment.function_to_evolve,
        random_seed=config.experiment.random_seed,
        logger=logger,
    )
    evaluator.prepare(experiment_dir)

    seed_candidate = database.build_seed_candidate()
    seed_evaluation = evaluator.evaluate_candidate(seed_candidate)
    if seed_evaluation is None:
        raise ValueError("The seed priority function could not be evaluated.")
    database.register_seed(dict(seed_evaluation.scores_per_test()))
    database.save_snapshot(
        experiment_dir / "program_db" / "bootstrap.pkl",
        experiment_dir / "program_db" / "bootstrap_summary.json",
        extra_metadata={"stage": "bootstrap"},
    )
    logger.info(f'''Registered seed priority function with reduced score {seed_evaluation.reduced_score}.
                The full score is {seed_evaluation.scores_per_test()}.''')
    return _run_cycles(
        config,
        experiment_dir,
        database,
        logger,
        start_cycle_index=1,
        initial_stagnation_cycles=0,
        reuse_existing_cycle_dir=False,
    )


def resume_experiment(experiment_dir: str | Path) -> Path:
    """Resume an interrupted FunSearch priority-function experiment.

    Input:
        experiment_dir: Existing experiment directory containing `config.used.json`
            plus bootstrap and per-cycle snapshots.

    Output:
        The same experiment directory after rerunning the latest incomplete
        started cycle from its `program_db_start.pkl` snapshot.
    """

    resolved_experiment_dir = Path(experiment_dir).expanduser().resolve()
    config_path = resolved_experiment_dir / "config.used.json"
    if not config_path.exists():
        raise ValueError(
            f"Cannot resume because {config_path} does not exist."
        )

    config = load_pipeline_config(config_path)
    resume_state = _build_resume_state(resolved_experiment_dir, config)
    logger = configure_main_logger(resolved_experiment_dir / "main.log", config.logging.level)

    np.random.seed(config.experiment.random_seed)

    if resume_state.reuse_existing_cycle_dir:
        _cleanup_incomplete_cycle_dir(
            _cycle_dir(resolved_experiment_dir, resume_state.start_cycle_index)
        )

    database = _load_database_snapshot(resume_state.database_snapshot_path)
    evaluator = build_evaluator(
        config.evaluator,
        function_name=config.experiment.function_to_evolve,
        random_seed=config.experiment.random_seed,
        logger=logger,
    )
    evaluator.prepare(resolved_experiment_dir)
    logger.info(
        "Resuming experiment directory %s from cycle %d using snapshot %s",
        resolved_experiment_dir,
        resume_state.start_cycle_index,
        resume_state.database_snapshot_path,
    )

    return _run_cycles(
        config,
        resolved_experiment_dir,
        database,
        logger,
        start_cycle_index=resume_state.start_cycle_index,
        initial_stagnation_cycles=resume_state.stagnation_cycles,
        reuse_existing_cycle_dir=resume_state.reuse_existing_cycle_dir,
    )
