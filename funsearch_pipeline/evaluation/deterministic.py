from __future__ import annotations

from pathlib import Path
import json
import math

from funsearch_pipeline.evaluation.interfaces import EvaluatedCandidate
from funsearch_pipeline.evaluation.interfaces import PairScore
from funsearch_pipeline.program_database.database import CandidateProgram


class DeterministicPriorityEvaluator:
    """Synthetic evaluator used to validate orchestration before oracle wiring."""

    def __init__(self, *, function_name: str) -> None:
        """Create a synthetic evaluator for smoke tests.

        Input:
            function_name: Unversioned priority function to execute.

        Output:
            Evaluator instance with deterministic synthetic scoring behavior.
        """

        self._function_name = function_name

    def prepare(self, experiment_dir: Path) -> None:
        """Write a deterministic evaluator manifest.

        Input:
            experiment_dir: Root experiment directory.

        Output:
            Creates `preprocessed/deterministic_manifest.json` for traceability.
        """

        manifest_path = experiment_dir / "preprocessed" / "deterministic_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "backend": "deterministic",
                    "purpose": "Smoke-test evaluator for program-db and cycle orchestration.",
                },
                indent=2,
                sort_keys=True,
            )
        )

    def evaluate_candidate(self, candidate: CandidateProgram) -> EvaluatedCandidate | None:
        """Score one candidate by executing it on synthetic inputs.

        Input:
            candidate: Runnable priority-function candidate.

        Output:
            Synthetic two-pair score result, or `None` if execution fails or the
            returned score is not finite.
        """

        namespace: dict[str, object] = {}
        try:
            exec(candidate.program_source, namespace)
            priority_function = namespace[self._function_name]
            raw_value = priority_function(
                training_data={"records": []},
                ancestry_coordinate=[0.25, -0.5, 0.75],
                target_variant="dosage__rsSynthetic",
            )
            score = float(raw_value)
        except Exception:
            return None

        if not math.isfinite(score):
            return None

        pair_scores = (
            PairScore(name="synthetic_pair_no_covariates", score=score),
            PairScore(name="synthetic_pair_with_covariates", score=score - 0.1),
        )
        reduced_score = sum(item.score for item in pair_scores) / len(pair_scores)
        return EvaluatedCandidate(
            candidate=candidate,
            pair_scores=pair_scores,
            reduced_score=reduced_score,
            metadata={"synthetic_priority_output": score},
        )
