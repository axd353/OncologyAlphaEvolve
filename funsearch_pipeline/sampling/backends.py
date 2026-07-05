from __future__ import annotations

from funsearch_pipeline.sampling.deterministic import generate_deterministic_completions
from funsearch_pipeline.sampling.interfaces import GeneratedCompletion
from funsearch_pipeline.sampling.interfaces import SamplerRequest
from funsearch_pipeline.sampling.openai_backend import generate_openai_completions


def run_sampling_request(request: SamplerRequest) -> list[GeneratedCompletion]:
    """Run one sampler backend request.

    Input:
        request: Backend-agnostic sampler request containing the prompt code,
            system prompt, model settings, output directory, and sampler log
            path. In the island sampler this request usually asks for exactly
            one LLM completion so the island can be updated before the next
            interaction.

    Output:
        List of generated completions. Backends return an empty list when a
        request fails in a recoverable way and raise `ValueError` for unknown
        backend names.
    """

    if request.backend == "deterministic":
        return generate_deterministic_completions(request)
    if request.backend == "openai":
        return generate_openai_completions(request)
    raise ValueError(f"Unsupported sampler backend: {request.backend}")
