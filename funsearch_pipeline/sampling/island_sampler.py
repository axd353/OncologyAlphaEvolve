from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from funsearch.implementation import code_manipulation
from funsearch_pipeline.config import EvaluatorSettings
from funsearch_pipeline.config import SamplerSettings
from funsearch_pipeline.evaluation import build_evaluator
from funsearch_pipeline.logging_utils import configure_file_logger
from funsearch_pipeline.program_database import IslandPrompt
from funsearch_pipeline.program_database import IslandShard
from funsearch_pipeline.sampling.backends import run_sampling_request
from funsearch_pipeline.sampling.priority_candidate_validation import build_candidate_program
from funsearch_pipeline.sampling.priority_candidate_validation import validate_candidate_priority_function
from funsearch_pipeline.sampling.interfaces import SamplerRequest
from funsearch_pipeline.sampling.interfaces import append_sampler_log


@dataclass(frozen=True)
class IslandSamplerRequest:
    """Inputs needed for one sampler process to work on one island shard.

    Input fields:
        cycle_index: Current outer evolution cycle.
        island_shard: Deep-copied island shard owned by this worker.
        template: Parsed seed program template used to materialize candidates.
        function_to_evolve: Unversioned priority function name.
        sampler_settings: LLM backend and sampling hyperparameters.
        evaluator_settings: Evaluator backend and scoring hyperparameters.
        logging_level: Logging level used for the worker-local evaluator logger.
        experiment_dir: Shared experiment directory containing prepared data.
        system_prompt: Text prompt supplied as LLM instructions.
        output_dir: Per-island directory for raw sampler artifacts.
        log_path: Per-island sampler log file.

    Output use:
        Passed to `run_island_sampler`, which returns an updated shard and
        candidate counts for cycle summary logging.
    """

    cycle_index: int
    island_shard: IslandShard
    template: code_manipulation.Program
    function_to_evolve: str
    sampler_settings: SamplerSettings
    evaluator_settings: EvaluatorSettings
    logging_level: str
    experiment_dir: Path
    system_prompt: str
    output_dir: Path
    log_path: Path


@dataclass(frozen=True)
class IslandSamplerResult:
    """Result returned after one sampler process finishes its island shard.

    Input fields:
        cycle_index: Cycle handled by this worker.
        island_id: Island id represented by `island_shard`.
        island_shard: Mutated shard after all local sampling/evaluation.
        generated_candidates: Number of completions returned by the LLM backend.
        accepted_candidates: Number of generated candidates registered locally.

    Output use:
        The runner combines `island_shard` objects from all workers to rebuild
        the parent program database at the end of the cycle.
    """

    cycle_index: int
    island_id: int
    island_shard: IslandShard
    generated_candidates: int
    accepted_candidates: int


def _prompt_file(output_dir: Path, sample_index: int) -> Path:
    """Return the prompt artifact path for one island interaction.

    Input:
        output_dir: Per-island sampler output directory.
        sample_index: Candidate interaction number in this cycle.

    Output:
        Path where the exact prompt sent to the backend should be stored.
    """

    return output_dir / f"sample_{sample_index:03d}" / "prompt.py"


def _write_prompt(output_dir: Path, prompt: IslandPrompt, sample_index: int) -> Path:
    """Persist the exact island prompt used for one LLM interaction.

    Input:
        output_dir: Per-island sampler output directory.
        prompt: Prompt generated from the shard's current state.
        sample_index: Candidate interaction number in this cycle.

    Output:
        Path to the written prompt file.
    """

    prompt_path = _prompt_file(output_dir, sample_index)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt.code)
    return prompt_path


def _log_prompt(log_path: Path, prompt: IslandPrompt, sample_index: int) -> None:
    """Log the first full prompt for one island during a cycle.

    Input:
        log_path: Per-island sampler log file.
        prompt: Prompt generated from the shard's current state.
        sample_index: Candidate interaction number in this cycle.

    Output:
        Appends the full first prompt for the island/cycle to the sampler log.
        Only the first prompt of the cycle is logged verbatim to avoid bloating
        the log with repeated prompt bodies.
    """

    if sample_index != 0:
        return

    append_sampler_log(log_path, f"sample_index=0 full_prompt_begin island={prompt.island_id}")
    append_sampler_log(log_path, prompt.code.rstrip("\n"))
    append_sampler_log(log_path, f"sample_index=0 full_prompt_end island={prompt.island_id}")


def _build_single_completion_request(
    request: IslandSamplerRequest,
    prompt: IslandPrompt,
    sample_index: int,
    attempt_index: int,
) -> SamplerRequest:
    """Create one backend request from the shard's current prompt.

    Input:
        request: Per-island sampler worker request.
        prompt: Fresh prompt generated after the latest local registration.
        sample_index: Candidate interaction number in this cycle.

    Output:
        `SamplerRequest` asking the backend for one completion.
    """

    return SamplerRequest(
        backend=request.sampler_settings.backend,
        cycle_index=request.cycle_index,
        island_id=request.island_shard.island_id,
        version_generated=prompt.version_generated,
        prompt_code=prompt.code,
        system_prompt=request.system_prompt,
        model=request.sampler_settings.model,
        candidates_per_island_per_cycle=1,
        output_dir=request.output_dir / f"sample_{sample_index:03d}" / f"attempt_{attempt_index:02d}",
        log_path=request.log_path,
        temperature=request.sampler_settings.temperature,
        max_output_tokens=request.sampler_settings.max_output_tokens,
    )


def _format_sampler_cpu_binding() -> tuple[str, str]:
    current_cpu = "unknown"
    if hasattr(os, "sched_getcpu"):
        try:
            current_cpu = str(int(os.sched_getcpu()))
        except OSError:
            current_cpu = "unknown"

    allowed_cpus = "unknown"
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed_cpus = str(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
        except OSError:
            allowed_cpus = "unknown"

    return current_cpu, allowed_cpus


def run_island_sampler(request: IslandSamplerRequest) -> IslandSamplerResult:
    """Sample, evaluate, and register candidates on one island shard.

    Input:
        request: Complete worker request for one island. The worker owns the
            shard for the duration of the cycle, creates its own evaluator, and
            writes only to its per-island output/log paths.

    Output:
        `IslandSamplerResult` containing the mutated island shard and counts of
        generated and accepted candidates. The parent runner combines all shard
        results after every sampler process finishes.
    """

    request.output_dir.mkdir(parents=True, exist_ok=True)
    append_sampler_log(
        request.log_path,
        (
            f"cycle={request.cycle_index} island={request.island_shard.island_id} "
            f"experiment_dir={request.experiment_dir} sampler_start"
        ),
    )
    current_cpu, allowed_cpus = _format_sampler_cpu_binding()
    append_sampler_log(
        request.log_path,
        (
            f"cycle={request.cycle_index} island={request.island_shard.island_id} "
            f"sampler_cpu pid={os.getpid()} current_cpu={current_cpu} "
            f"allowed_cpus={allowed_cpus}"
        ),
    )

    evaluator_logger = configure_file_logger(
        request.log_path,
        request.logging_level,
        logger_name=(
            f"funsearch_pipeline.sampler.cycle_{request.cycle_index:04d}."
            f"island_{request.island_shard.island_id:03d}"
        ),
    )

    evaluator = build_evaluator(
        request.evaluator_settings,
        function_name=request.function_to_evolve,
        logger=evaluator_logger,
    )
    evaluator.prepare(request.experiment_dir)
    generated_candidates = 0
    accepted_candidates = 0
    logged_first_success = False
    max_attempts_per_prompt = 5

    for sample_index in range(request.sampler_settings.candidates_per_island_per_cycle):
        prompt = request.island_shard.get_prompt()
        prompt_path = _write_prompt(request.output_dir, prompt, sample_index)
        _log_prompt(request.log_path, prompt, sample_index)
        append_sampler_log(
            request.log_path,
            (
                f"sample_index={sample_index} prompt_path={prompt_path} "
                f"version_generated={prompt.version_generated}"
            ),
        )

        registered = False
        for attempt_index in range(1, max_attempts_per_prompt + 1):
            completions = run_sampling_request(
                _build_single_completion_request(request, prompt, sample_index, attempt_index)
            )
            generated_candidates += 1

            if not completions:
                append_sampler_log(
                    request.log_path,
                    (
                        f"sample_index={sample_index} attempt={attempt_index} "
                        "rejected=empty_completion"
                    ),
                )
                continue

            completion = completions[0]
            try:
                candidate_program = build_candidate_program(
                    template=request.template,
                    function_to_evolve=request.function_to_evolve,
                    island_id=request.island_shard.island_id,
                    version_generated=prompt.version_generated,
                    raw_completion=completion.raw_completion,
                    sample_index=sample_index,
                )
                validate_candidate_priority_function(
                    candidate_program,
                    request.function_to_evolve,
                )
            except Exception as exc:
                append_sampler_log(
                    request.log_path,
                    (
                        f"sample_index={sample_index} attempt={attempt_index} "
                        f"rejected=invalid_priority_function error={type(exc).__name__}: {exc}"
                    ),
                )
                continue

            evaluated_candidate = evaluator.evaluate_candidate(candidate_program)
            if evaluated_candidate is None:
                append_sampler_log(
                    request.log_path,
                    (
                        f"sample_index={sample_index} attempt={attempt_index} "
                        "rejected=evaluation_failed"
                    ),
                )
                continue

            improved = request.island_shard.register_candidate(
                evaluated_candidate.candidate,
                dict(evaluated_candidate.scores_per_test()),
            )
            accepted_candidates += 1
            if not logged_first_success:
                append_sampler_log(
                    request.log_path,
                    (
                        f"sample_index={sample_index} attempt={attempt_index} "
                        "registered=true first_success=true full_priority_function_begin"
                    ),
                )
                append_sampler_log(
                    request.log_path,
                    candidate_program.raw_completion.rstrip("\n"),
                )
                append_sampler_log(
                    request.log_path,
                    (
                        f"sample_index={sample_index} attempt={attempt_index} "
                        f"registered=true after_attempts={attempt_index} "
                        f"better_than_present_best={improved} full_priority_function_end"
                    ),
                )
                logged_first_success = True
            else:
                append_sampler_log(
                    request.log_path,
                    (
                        f"sample_index={sample_index} attempt={attempt_index} "
                        f"registered=true after_attempts={attempt_index} "
                        f"better_than_present_best={improved}"
                    ),
                )
            registered = True
            break

        if not registered:
            append_sampler_log(
                request.log_path,
                (
                    f"sample_index={sample_index} no_priority_function_generated "
                    f"after_attempts={max_attempts_per_prompt}"
                ),
            )

    append_sampler_log(
        request.log_path,
        (
            f"cycle={request.cycle_index} island={request.island_shard.island_id} "
            f"generated={generated_candidates} accepted={accepted_candidates} sampler_end"
        ),
    )
    return IslandSamplerResult(
        cycle_index=request.cycle_index,
        island_id=request.island_shard.island_id,
        island_shard=request.island_shard,
        generated_candidates=generated_candidates,
        accepted_candidates=accepted_candidates,
    )
