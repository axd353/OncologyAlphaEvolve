from __future__ import annotations

from funsearch_pipeline.config import EvaluatorSettings
from funsearch_pipeline.evaluation.deterministic import DeterministicPriorityEvaluator
from funsearch_pipeline.evaluation.interfaces import PriorityFunctionEvaluator
from funsearch_pipeline.evaluation.procedure2 import Procedure2PriorityEvaluator


def build_evaluator(
    settings: EvaluatorSettings,
    *,
    function_name: str,
) -> PriorityFunctionEvaluator:
    """Construct the configured priority-function evaluator.

    Input:
        settings: Evaluator subsection parsed from the TOML config.
        function_name: Unversioned priority function name to execute.

    Output:
        Evaluator object implementing `prepare` and `evaluate_candidate`.
    """

    if settings.backend == "deterministic":
        return DeterministicPriorityEvaluator(function_name=function_name)
    if settings.backend == "procedure2":
        return Procedure2PriorityEvaluator(settings=settings, function_name=function_name)
    raise ValueError(f"Unsupported evaluator backend: {settings.backend}")


__all__ = ["PriorityFunctionEvaluator", "build_evaluator"]
