from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
import shutil

import pandas as pd

COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE = "sampler_completed_priority_function_counts.pkl"
VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE = "sampler_validated_priority_function_counts.pkl"
EVALUATION_COMPLETED_COUNTS_PICKLE = "sampler_evaluation_completed_counts.pkl"
ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE = "sampler_island_best_improvement_counts.pkl"

_CYCLE_DIR_RE = re.compile(r"^cycle_(\d+)$")
_ISLAND_LOG_RE = re.compile(r"^island_(\d+)\.log$")
_SAMPLE_INDEX_RE = re.compile(r"sample_index=(\d+)")
_ATTEMPT_RE = re.compile(r"attempt=(\d+)")


@dataclass(frozen=True)
class SamplerLogMetrics:
    """Counts extracted from one per-island sampler log.

    Input:
        cycle_index: Cycle number inferred from the run directory layout.
        island_id: Island id inferred from the sampler log filename.
        completed_priority_function_count: Attempts whose completion was not
            logged as `rejected=empty_completion`.
        empty_completion_count: Attempts logged as `rejected=empty_completion`.
        total_sampler_attempt_count: All logged sampler attempts, including
            empty and non-empty completions.
        validated_priority_function_count: Attempts that got past
            `validate_candidate_priority_function(...)` and reached evaluator
            execution.
        evaluation_completed_count: Attempts where `evaluated_candidate is not
            None`, logged as `registered=true`.
        island_best_improvement_count: Registered attempts logged with
            `better_than_present_best=True`.

    Output use:
        Aggregated into the run-level DataFrames saved by
        `postprocess_funsearch_run(...)`.
    """

    cycle_index: int
    island_id: int
    completed_priority_function_count: int
    empty_completion_count: int
    total_sampler_attempt_count: int
    validated_priority_function_count: int
    evaluation_completed_count: int
    island_best_improvement_count: int


@dataclass(frozen=True)
class PostProcessingOutputs:
    """Paths written by run-level post-processing."""

    logger_path: Path | None
    completed_priority_function_counts_path: Path
    validated_priority_function_counts_path: Path
    evaluation_completed_counts_path: Path
    island_best_improvement_counts_path: Path


@dataclass
class _AttemptRecord:
    empty_completion: bool = False
    invalid_priority_function: bool = False
    evaluation_failed: bool = False
    registered: bool = False
    better_than_present_best: bool = False


def _normalize_run_dir(run_dir: str | Path) -> Path:
    run_dir_path = Path(run_dir).expanduser().resolve()
    if not run_dir_path.exists() or not run_dir_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir_path}")
    return run_dir_path


def _iter_sampler_logs(run_dir: Path) -> list[Path]:
    log_paths = sorted(run_dir.glob("cycle_*/sampler_logs/island_*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No sampler logs found under {run_dir}")
    return log_paths


def _infer_cycle_and_island(log_path: Path) -> tuple[int, int]:
    cycle_match = _CYCLE_DIR_RE.match(log_path.parent.parent.name)
    island_match = _ISLAND_LOG_RE.match(log_path.name)
    if cycle_match is None or island_match is None:
        raise ValueError(f"Sampler log path does not match expected layout: {log_path}")
    return int(cycle_match.group(1)), int(island_match.group(1))


def _extract_attempt_key(line: str) -> tuple[int, int] | None:
    sample_index_match = _SAMPLE_INDEX_RE.search(line)
    attempt_match = _ATTEMPT_RE.search(line)
    if sample_index_match is None or attempt_match is None:
        return None
    return int(sample_index_match.group(1)), int(attempt_match.group(1))


def _collect_attempt_records(log_path: Path) -> dict[tuple[int, int], _AttemptRecord]:
    attempts: dict[tuple[int, int], _AttemptRecord] = {}
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        attempt_key = _extract_attempt_key(raw_line)
        if attempt_key is None:
            continue
        attempt_record = attempts.setdefault(attempt_key, _AttemptRecord())
        if "rejected=empty_completion" in raw_line:
            attempt_record.empty_completion = True
        if "rejected=invalid_priority_function" in raw_line:
            attempt_record.invalid_priority_function = True
        if "rejected=evaluation_failed" in raw_line:
            attempt_record.evaluation_failed = True
        if "registered=true" in raw_line:
            attempt_record.registered = True
        if "better_than_present_best=True" in raw_line:
            attempt_record.better_than_present_best = True
    return attempts


def extract_sampler_log_metrics(log_path: str | Path) -> SamplerLogMetrics:
    """Extract all requested counts from one per-island sampler log.

    Input:
        log_path: Path to one `cycle_XXXX/sampler_logs/island_YYY.log` file.

    Output:
        `SamplerLogMetrics` for that single island and cycle.
    """

    normalized_log_path = Path(log_path).expanduser().resolve()
    cycle_index, island_id = _infer_cycle_and_island(normalized_log_path)
    attempts = _collect_attempt_records(normalized_log_path)
    attempt_records = list(attempts.values())
    empty_completion_count = sum(1 for record in attempt_records if record.empty_completion)
    completed_priority_function_count = len(attempt_records) - empty_completion_count
    return SamplerLogMetrics(
        cycle_index=cycle_index,
        island_id=island_id,
        completed_priority_function_count=completed_priority_function_count,
        empty_completion_count=empty_completion_count,
        total_sampler_attempt_count=len(attempt_records),
        validated_priority_function_count=sum(
            1 for record in attempt_records if record.evaluation_failed or record.registered
        ),
        evaluation_completed_count=sum(1 for record in attempt_records if record.registered),
        island_best_improvement_count=sum(
            1
            for record in attempt_records
            if record.registered and record.better_than_present_best
        ),
    )


def count_completed_priority_functions_in_sampler_log(log_path: str | Path) -> int:
    """Count non-empty LLM completions in one sampler log."""

    return extract_sampler_log_metrics(log_path).completed_priority_function_count


def count_empty_completions_in_sampler_log(log_path: str | Path) -> int:
    """Count sampler attempts logged as `rejected=empty_completion`."""

    return extract_sampler_log_metrics(log_path).empty_completion_count


def count_total_sampler_attempts_in_sampler_log(log_path: str | Path) -> int:
    """Count all logged sampler attempts in one sampler log."""

    return extract_sampler_log_metrics(log_path).total_sampler_attempt_count


def count_validation_passes_in_sampler_log(log_path: str | Path) -> int:
    """Count attempts that passed candidate validation in one sampler log."""

    return extract_sampler_log_metrics(log_path).validated_priority_function_count


def count_evaluation_completions_in_sampler_log(log_path: str | Path) -> int:
    """Count attempts with `evaluated_candidate is not None` in one sampler log."""

    return extract_sampler_log_metrics(log_path).evaluation_completed_count


def count_island_best_improvements_in_sampler_log(log_path: str | Path) -> int:
    """Count registered attempts that improved the current island best."""

    return extract_sampler_log_metrics(log_path).island_best_improvement_count


def move_corresponding_logger_file(run_dir: str | Path) -> Path | None:
    """Move the shell-captured `logger_*` file into the experiment directory.

    Input:
        run_dir: FunSearch experiment directory under `prio_func_disc_runs/`.

    Output:
        Destination path under `run_dir`, or `None` when no matching shell log
        is found. Existing `logger_*` files already inside `run_dir` are reused.
    """

    normalized_run_dir = _normalize_run_dir(run_dir)
    existing_logger_paths = sorted(normalized_run_dir.glob("logger_*"))
    if existing_logger_paths:
        return existing_logger_paths[0]

    run_dir_text = str(normalized_run_dir)
    matching_logger_paths: list[Path] = []
    for logger_path in sorted(normalized_run_dir.parent.glob("logger_*")):
        if not logger_path.is_file():
            continue
        if run_dir_text in logger_path.read_text(encoding="utf-8", errors="ignore"):
            matching_logger_paths.append(logger_path)

    if not matching_logger_paths:
        return None
    if len(matching_logger_paths) != 1:
        raise ValueError(
            f"Found multiple logger files for {normalized_run_dir}: {matching_logger_paths}"
        )

    source_path = matching_logger_paths[0]
    destination_path = normalized_run_dir / source_path.name
    shutil.move(str(source_path), str(destination_path))
    return destination_path


def _build_sampler_metrics_dataframe(run_dir: Path) -> pd.DataFrame:
    metric_rows = [asdict(extract_sampler_log_metrics(log_path)) for log_path in _iter_sampler_logs(run_dir)]
    return pd.DataFrame(
        metric_rows,
        columns=[
            "cycle_index",
            "island_id",
            "completed_priority_function_count",
            "empty_completion_count",
            "total_sampler_attempt_count",
            "validated_priority_function_count",
            "evaluation_completed_count",
            "island_best_improvement_count",
        ],
    ).sort_values(["cycle_index", "island_id"]).reset_index(drop=True)


def _configured_candidates_per_island_per_cycle(run_dir: Path) -> int:
    config_path = run_dir / "config.used.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        return int(config_payload["sampler"]["candidates_per_island_per_cycle"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Could not read sampler.candidates_per_island_per_cycle from config.used.json"
        ) from exc


def _per_island_count_dataframe(metrics_frame: pd.DataFrame, count_column: str) -> pd.DataFrame:
    return metrics_frame[["cycle_index", "island_id", count_column]].copy()


def _completed_priority_function_counts_dataframe(
    metrics_frame: pd.DataFrame,
    *,
    configured_candidates_per_island_per_cycle: int,
) -> pd.DataFrame:
    completed_counts = _per_island_count_dataframe(
        metrics_frame,
        "completed_priority_function_count",
    )
    completed_counts["empty_completion_count"] = metrics_frame["empty_completion_count"]
    completed_counts["total_sampler_attempt_count"] = metrics_frame[
        "total_sampler_attempt_count"
    ]
    completed_counts["configured_candidates_per_island_per_cycle"] = (
        configured_candidates_per_island_per_cycle
    )
    return completed_counts


def _per_cycle_improvement_dataframe(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    if metrics_frame.empty:
        return pd.DataFrame(columns=["cycle_index", "island_best_improvement_count"])
    return (
        metrics_frame.groupby("cycle_index", as_index=False)["island_best_improvement_count"]
        .sum()
        .sort_values(["cycle_index"])
        .reset_index(drop=True)
    )


def postprocess_funsearch_run(run_dir: str | Path) -> PostProcessingOutputs:
    """Move the matching shell log and persist the requested summary pickles.

    Input:
        run_dir: FunSearch experiment directory such as
            `prio_func_disc_runs/oracle_priority_20260716_050704`.

    Output:
        `PostProcessingOutputs` pointing at the moved logger file, when found,
        and the four pickle files written into `run_dir`.
    """

    normalized_run_dir = _normalize_run_dir(run_dir)
    logger_path = move_corresponding_logger_file(normalized_run_dir)
    metrics_frame = _build_sampler_metrics_dataframe(normalized_run_dir)
    configured_candidates_per_island_per_cycle = _configured_candidates_per_island_per_cycle(
        normalized_run_dir
    )

    completed_priority_function_counts = _completed_priority_function_counts_dataframe(
        metrics_frame,
        configured_candidates_per_island_per_cycle=configured_candidates_per_island_per_cycle,
    )
    validated_priority_function_counts = _per_island_count_dataframe(
        metrics_frame,
        "validated_priority_function_count",
    )
    evaluation_completed_counts = _per_island_count_dataframe(
        metrics_frame,
        "evaluation_completed_count",
    )
    island_best_improvement_counts = _per_cycle_improvement_dataframe(metrics_frame)

    completed_priority_function_counts_path = (
        normalized_run_dir / COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE
    )
    validated_priority_function_counts_path = (
        normalized_run_dir / VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE
    )
    evaluation_completed_counts_path = normalized_run_dir / EVALUATION_COMPLETED_COUNTS_PICKLE
    island_best_improvement_counts_path = normalized_run_dir / ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE

    completed_priority_function_counts.to_pickle(completed_priority_function_counts_path)
    validated_priority_function_counts.to_pickle(validated_priority_function_counts_path)
    evaluation_completed_counts.to_pickle(evaluation_completed_counts_path)
    island_best_improvement_counts.to_pickle(island_best_improvement_counts_path)

    return PostProcessingOutputs(
        logger_path=logger_path,
        completed_priority_function_counts_path=completed_priority_function_counts_path,
        validated_priority_function_counts_path=validated_priority_function_counts_path,
        evaluation_completed_counts_path=evaluation_completed_counts_path,
        island_best_improvement_counts_path=island_best_improvement_counts_path,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for post-processing one FunSearch run directory."""

    parser = argparse.ArgumentParser(
        description="Move the matching shell logger file and write FunSearch sampler summary pickles."
    )
    parser.add_argument("run_dir", help="FunSearch experiment directory to post-process.")
    args = parser.parse_args(argv)

    outputs = postprocess_funsearch_run(args.run_dir)
    print(f"logger_path={outputs.logger_path}")
    print(
        f"completed_priority_function_counts_path={outputs.completed_priority_function_counts_path}"
    )
    print(
        f"validated_priority_function_counts_path={outputs.validated_priority_function_counts_path}"
    )
    print(f"evaluation_completed_counts_path={outputs.evaluation_completed_counts_path}")
    print(f"island_best_improvement_counts_path={outputs.island_best_improvement_counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())