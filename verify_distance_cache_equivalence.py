from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import argparse
import json
import math

from PostProcesingData.evaluate_priofunction import evaluate_priority_function
from PostProcesingData.evaluate_priofunction import load_evaluation_config
from funsearch_pipeline.config import load_pipeline_config
from funsearch_pipeline.evaluation.procedure2 import Procedure2PriorityEvaluator
from funsearch_pipeline.program_database.database import CandidateProgram


def _assert_close(name: str, left: float, right: float, *, atol: float = 1e-12, rtol: float = 1e-12) -> None:
    if not math.isclose(left, right, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{name} mismatch: left={left!r} right={right!r}.")


def _candidate_from_path(candidate_path: Path, *, function_name: str) -> CandidateProgram:
    program_source = candidate_path.read_text()
    return CandidateProgram(
        island_id=-1,
        version_generated=None,
        sample_index=0,
        raw_completion=program_source,
        evolved_function=None,
        program_source=program_source,
        function_name=function_name,
    )


def verify_pipeline(
    *,
    config_path: Path,
    candidate_path: Path | None,
) -> dict[str, Any]:
    config = load_pipeline_config(config_path)
    if config.evaluator.backend != "procedure2":
        raise ValueError(
            f"Pipeline verification currently supports only evaluator.backend='procedure2'. Got {config.evaluator.backend!r}."
        )

    resolved_candidate_path = candidate_path or config.experiment.seed_priority_path
    candidate = _candidate_from_path(
        resolved_candidate_path,
        function_name=config.experiment.function_to_evolve,
    )

    with TemporaryDirectory(prefix="distance-cache-verify-") as temporary_root:
        temporary_root_path = Path(temporary_root)
        experiment_dir = temporary_root_path / "pipeline_experiment"
        cache_root = temporary_root_path / "distance_cache"

        cached_evaluator = Procedure2PriorityEvaluator(
            settings=replace(
                config.evaluator,
                distance_cache_enabled=True,
                distance_cache_dir=cache_root,
            ),
            function_name=config.experiment.function_to_evolve,
            random_seed=config.experiment.random_seed,
        )
        cached_evaluator.prepare(experiment_dir)

        uncached_evaluator = Procedure2PriorityEvaluator(
            settings=replace(
                config.evaluator,
                distance_cache_enabled=False,
                distance_cache_dir=None,
            ),
            function_name=config.experiment.function_to_evolve,
            random_seed=config.experiment.random_seed,
        )
        uncached_evaluator.prepare(experiment_dir)

        cached_result = cached_evaluator.evaluate_candidate(candidate)
        uncached_result = uncached_evaluator.evaluate_candidate(candidate)
        if cached_result is None or uncached_result is None:
            raise RuntimeError("Pipeline verification could not score the candidate in one of the two modes.")

        cached_scores = cached_result.scores_per_test()
        uncached_scores = uncached_result.scores_per_test()
        if list(cached_scores) != list(uncached_scores):
            raise AssertionError(
                f"Pipeline score-name mismatch: cached={list(cached_scores)!r} uncached={list(uncached_scores)!r}."
            )
        for score_name in cached_scores:
            _assert_close(f"pipeline score {score_name}", cached_scores[score_name], uncached_scores[score_name])

        cached_pairs = cached_result.metadata["procedure2_pairs"]
        uncached_pairs = uncached_result.metadata["procedure2_pairs"]
        if set(cached_pairs) != set(uncached_pairs):
            raise AssertionError(
                f"Pipeline pair-name mismatch: cached={set(cached_pairs)!r} uncached={set(uncached_pairs)!r}."
            )
        for pair_name in sorted(cached_pairs):
            cached_folds = cached_pairs[pair_name]["folds"]
            uncached_folds = uncached_pairs[pair_name]["folds"]
            if set(cached_folds) != set(uncached_folds):
                raise AssertionError(
                    f"Pipeline fold-name mismatch for pair {pair_name!r}: "
                    f"cached={set(cached_folds)!r} uncached={set(uncached_folds)!r}."
                )
            for fold_name in sorted(cached_folds):
                cached_fold = cached_folds[fold_name]
                uncached_fold = uncached_folds[fold_name]
                if cached_fold["selected_calibration_penalty"] != uncached_fold["selected_calibration_penalty"]:
                    raise AssertionError(
                        f"Pipeline selected penalty mismatch for {pair_name}.{fold_name}: "
                        f"cached={cached_fold['selected_calibration_penalty']!r} "
                        f"uncached={uncached_fold['selected_calibration_penalty']!r}."
                    )
                _assert_close(
                    f"pipeline direct_auc {pair_name}.{fold_name}",
                    float(cached_fold["direct_auc"]),
                    float(uncached_fold["direct_auc"]),
                )
                _assert_close(
                    f"pipeline bootstrap_auc_median {pair_name}.{fold_name}",
                    float(cached_fold["bootstrap_auc_median"]),
                    float(uncached_fold["bootstrap_auc_median"]),
                )

        cache_manifests = sorted(str(path) for path in cache_root.rglob("*.manifest.json"))
        return {
            "candidate_path": str(resolved_candidate_path),
            "score_names": list(cached_scores),
            "scores": {name: float(value) for name, value in cached_scores.items()},
            "cache_manifest_count": len(cache_manifests),
            "cache_manifests": cache_manifests,
        }


def verify_postprocessing(
    *,
    config_path: Path,
    include_baselines: bool,
) -> dict[str, Any]:
    config = load_evaluation_config(config_path)
    baselines = config.baselines if include_baselines else ()
    cached_cache_dir = config.distance_cache_dir or (config.prio_function_path.parent / "distance_cache")

    uncached_report = evaluate_priority_function(
        replace(config, distance_cache_enabled=False, distance_cache_dir=None, baselines=baselines),
        baselines_to_run=baselines,
    )
    cached_report = evaluate_priority_function(
        replace(config, distance_cache_enabled=True, distance_cache_dir=cached_cache_dir, baselines=baselines),
        baselines_to_run=baselines,
    )

    _assert_close("postprocessing heldout_auc_roc", cached_report.heldout_auc_roc, uncached_report.heldout_auc_roc)
    if len(cached_report.heldout_ancestry_evaluations) != len(uncached_report.heldout_ancestry_evaluations):
        raise AssertionError(
            "Heldout ancestry-evaluation count mismatch: "
            f"cached={len(cached_report.heldout_ancestry_evaluations)} "
            f"uncached={len(uncached_report.heldout_ancestry_evaluations)}."
        )
    for cached_evaluation, uncached_evaluation in zip(
        cached_report.heldout_ancestry_evaluations,
        uncached_report.heldout_ancestry_evaluations,
    ):
        if cached_evaluation.ancestry_group != uncached_evaluation.ancestry_group:
            raise AssertionError(
                "Heldout ancestry-group mismatch: "
                f"cached={cached_evaluation.ancestry_group!r} uncached={uncached_evaluation.ancestry_group!r}."
            )
        if cached_evaluation.subject_count != uncached_evaluation.subject_count:
            raise AssertionError(
                "Heldout ancestry subject-count mismatch for group "
                f"{cached_evaluation.ancestry_group!r}: cached={cached_evaluation.subject_count} "
                f"uncached={uncached_evaluation.subject_count}."
            )
        _assert_close(
            f"postprocessing heldout ancestry auc {cached_evaluation.ancestry_group}",
            cached_evaluation.auc_roc,
            uncached_evaluation.auc_roc,
        )

    if len(cached_report.baseline_evaluations) != len(uncached_report.baseline_evaluations):
        raise AssertionError(
            "Baseline-evaluation count mismatch: "
            f"cached={len(cached_report.baseline_evaluations)} uncached={len(uncached_report.baseline_evaluations)}."
        )
    for cached_baseline, uncached_baseline in zip(
        cached_report.baseline_evaluations,
        uncached_report.baseline_evaluations,
    ):
        if cached_baseline.name != uncached_baseline.name:
            raise AssertionError(
                f"Baseline name mismatch: cached={cached_baseline.name!r} uncached={uncached_baseline.name!r}."
            )
        _assert_close(
            f"baseline auc {cached_baseline.name}",
            cached_baseline.auc_roc,
            uncached_baseline.auc_roc,
        )

    cache_manifests = sorted(str(path) for path in cached_cache_dir.rglob("*.manifest.json"))
    return {
        "prio_function_path": str(config.prio_function_path),
        "heldout_auc_roc": float(cached_report.heldout_auc_roc),
        "cache_manifest_count": len(cache_manifests),
        "cache_manifests": cache_manifests,
        "baseline_names": [baseline.name for baseline in cached_report.baseline_evaluations],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cached and uncached ancestry-distance evaluation outputs."
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        help="Path to a funsearch_pipeline JSON config to verify the Procedure 2 evaluator slice.",
    )
    parser.add_argument(
        "--pipeline-candidate",
        type=Path,
        help="Optional concrete priority-function .py file for the pipeline verifier. Defaults to experiment.seed_priority_path.",
    )
    parser.add_argument(
        "--heldout-config",
        type=Path,
        help="Path to a PostProcesingData.evaluate_priofunction JSON config to verify heldout evaluation.",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also compare configured heldout baselines. Defaults to verifying only the priority-function path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the verification summary JSON.",
    )
    arguments = parser.parse_args()

    if arguments.pipeline_config is None and arguments.heldout_config is None:
        raise SystemExit("At least one of --pipeline-config or --heldout-config is required.")

    summary: dict[str, Any] = {}
    if arguments.pipeline_config is not None:
        summary["pipeline"] = verify_pipeline(
            config_path=arguments.pipeline_config.expanduser().resolve(),
            candidate_path=(
                arguments.pipeline_candidate.expanduser().resolve()
                if arguments.pipeline_candidate is not None
                else None
            ),
        )
    if arguments.heldout_config is not None:
        summary["postprocessing"] = verify_postprocessing(
            config_path=arguments.heldout_config.expanduser().resolve(),
            include_baselines=arguments.include_baselines,
        )

    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    print(summary_text)
    if arguments.output_json is not None:
        arguments.output_json.expanduser().resolve().write_text(summary_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()