from __future__ import annotations

import ast
import math
from typing import Callable

from funsearch.implementation import code_manipulation
from funsearch.implementation import evaluator as upstream_evaluator
from funsearch_pipeline.evaluation.procedure2 import _call_priority_function
from funsearch_pipeline.evaluation.procedure2 import _load_priority_function
from funsearch_pipeline.evaluation.procedure2 import _validate_priority_signature
from funsearch_pipeline.program_database.database import _normalize_function_body_indentation
from funsearch_pipeline.priority_tools.contracts import PriorityAncestryCoordinate
from funsearch_pipeline.priority_tools.contracts import PriorityTargetVariant
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingData
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingRecord
from funsearch_pipeline.program_database import CandidateProgram

PriorityFunction = Callable[[PriorityTrainingData, PriorityAncestryCoordinate, PriorityTargetVariant], float]


def build_candidate_program(
    *,
    template: code_manipulation.Program,
    function_to_evolve: str,
    island_id: int,
    version_generated: int | None,
    raw_completion: str,
    sample_index: int,
) -> CandidateProgram:
    """Materialize one raw completion into a runnable candidate program.

    Input:
        template: Parsed seed program template.
        function_to_evolve: Unversioned priority function name.
        island_id: Island that produced the completion.
        version_generated: Prompt version that produced the completion.
        raw_completion: Raw function-body text from the LLM backend.
        sample_index: Candidate slot index inside the current cycle.

    Output:
        `CandidateProgram` containing the compiled function and full source.
    """

    normalized_completion = _normalize_completion(
        raw_completion,
        version_generated=version_generated,
        function_to_evolve=function_to_evolve,
    )
    evolved_function, program_source = upstream_evaluator._sample_to_program(
        normalized_completion,
        version_generated,
        template,
        function_to_evolve,
    )
    return CandidateProgram(
        island_id=island_id,
        version_generated=version_generated,
        sample_index=sample_index,
        raw_completion=normalized_completion,
        evolved_function=evolved_function,
        program_source=program_source,
        function_name=function_to_evolve,
    )


def _normalize_completion(
    raw_completion: str,
    *,
    version_generated: int | None,
    function_to_evolve: str,
) -> str:
    """Accept either a bare body or a full versioned function definition.

    Input:
        raw_completion: Raw model text.
        version_generated: Version suffix expected for this prompt.
        function_to_evolve: Base priority function name.

    Output:
        Function body text consumable by upstream `_sample_to_program`.
    """

    stripped = raw_completion.strip()
    if not stripped:
        return raw_completion
    if stripped.startswith("def "):
        if version_generated is None:
            raise ValueError("A full function definition is not valid for the seed candidate.")
        return _normalize_function_body_indentation(_extract_function_body_from_definition(
            stripped,
            expected_function_name=f"{function_to_evolve}_v{version_generated}",
        ))
    return _normalize_function_body_indentation(raw_completion)


def _extract_function_body_from_definition(
    source: str,
    *,
    expected_function_name: str,
) -> str:
    """Extract the body from a full generated function definition."""

    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == expected_function_name:
            function_text = ast.get_source_segment(source, node)
            if function_text is None:
                break
            parsed_function = code_manipulation.text_to_function(function_text)
            return parsed_function.body + "\n"
    raise ValueError(
        f"Expected completion to define {expected_function_name!r}."
    )


def validate_candidate_priority_function(
    candidate_program: CandidateProgram,
    function_name: str,
    *,
    smoke_test_subject_count: int = 5,
    ancestry_dimension: int = 16,
) -> None:
    """Extract, validate, and smoke-test one candidate priority function.

    Input:
        candidate_program: Materialized runnable candidate.
        function_name: Unversioned priority function name to load.
        smoke_test_subject_count: Number of dummy subjects to include in the
            validation `PriorityTrainingData` payload.
        ancestry_dimension: Dimensionality of the dummy ancestry coordinate.

    Output:
        Returns `None` when the function is syntactically loadable, has the
        expected signature, and returns a finite non-negative score on a small
        synthetic validation payload. Raises an exception otherwise.
    """

    priority_function = _load_priority_function(candidate_program.program_source, function_name)
    _validate_priority_signature(priority_function)
    _run_priority_function_smoke_test(
        priority_function,
        smoke_test_subject_count=smoke_test_subject_count,
        ancestry_dimension=ancestry_dimension,
    )


def _run_priority_function_smoke_test(
    priority_function: PriorityFunction,
    *,
    smoke_test_subject_count: int,
    ancestry_dimension: int,
) -> None:
    """Run the candidate on a tiny synthetic Oracle-style payload.

    Input:
        priority_function: Callable extracted from the candidate source.
        smoke_test_subject_count: Number of dummy subjects to create.
        ancestry_dimension: Dimensionality of each dummy ancestry vector.

    Output:
        Raises if the function is not callable on a minimal normalized payload
        or if it returns a non-finite / negative radius.
    """

    dummy_data = _build_smoke_test_training_data(
        smoke_test_subject_count=smoke_test_subject_count,
        ancestry_dimension=ancestry_dimension,
    )
    dummy_coordinate = PriorityAncestryCoordinate(
        values=tuple(0.0 for _ in range(ancestry_dimension)),
        dimension=ancestry_dimension,
    )
    dummy_variant = PriorityTargetVariant(
        name="validation_variant",
        dosage_field="dosage__validation_variant",
        column_index=0,
    )
    radius = _call_priority_function(priority_function, dummy_data, dummy_coordinate, dummy_variant)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("Priority function must return a finite non-negative radius.")


def _build_smoke_test_training_data(
    *,
    smoke_test_subject_count: int,
    ancestry_dimension: int,
) -> PriorityTrainingData:
    """Create a tiny normalized training payload for sampler-side validation.

    Input:
        smoke_test_subject_count: Number of dummy records to construct.
        ancestry_dimension: Dimensionality of each dummy ancestry vector.

    Output:
        `PriorityTrainingData` containing alternating labels and simple variant
        dosage values.
    """

    records: list[PriorityTrainingRecord] = []
    for index in range(smoke_test_subject_count):
        records.append(
            PriorityTrainingRecord(
                label=float(index % 2),
                ancestry_coordinate=tuple(float(index) for _ in range(ancestry_dimension)),
                variant_dosages={"validation_variant": float(index % 3)},
                covariates=None,
            )
        )
    return PriorityTrainingData(
        records=tuple(records),
        variant_names=("validation_variant",),
        variant_dosage_fields=("dosage__validation_variant",),
        covariate_names=(),
        sample_count=smoke_test_subject_count,
        ancestry_dimension=ancestry_dimension,
        has_additional_covariates=False,
    )
