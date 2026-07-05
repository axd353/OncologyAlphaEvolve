from __future__ import annotations

from pathlib import Path
import json

from openai import OpenAI

from funsearch_pipeline.sampling.interfaces import GeneratedCompletion
from funsearch_pipeline.sampling.interfaces import SamplerRequest
from funsearch_pipeline.sampling.interfaces import append_sampler_log


def _completion_text_path(output_dir: Path, sample_index: int) -> Path:
    """Return the raw OpenAI completion artifact path.

    Input:
        output_dir: Directory for this backend interaction.
        sample_index: Backend-local completion index.

    Output:
        Path where the raw completion text is written.
    """

    return output_dir / f"candidate_{sample_index:03d}.pyfrag"


def _completion_metadata_path(output_dir: Path, sample_index: int) -> Path:
    """Return the OpenAI response metadata artifact path.

    Input:
        output_dir: Directory for this backend interaction.
        sample_index: Backend-local completion index.

    Output:
        Path where response metadata JSON is written.
    """

    return output_dir / f"candidate_{sample_index:03d}.json"


def generate_openai_completions(request: SamplerRequest) -> list[GeneratedCompletion]:
    """Call the OpenAI Responses API for priority-function completions.

    Input:
        request: Sampler request containing system instructions, island prompt
            code, model settings, output directory, and log path.

    Output:
        List of raw completions returned by the API. Recoverable API failures
        are logged and produce no completion for that request.
    """

    request.output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    append_sampler_log(
        request.log_path,
        (
            f"cycle={request.cycle_index} island={request.island_id} backend=openai "
            f"model={request.model} candidates={request.candidates_per_island_per_cycle}"
        ),
    )

    completions: list[GeneratedCompletion] = []
    for sample_index in range(request.candidates_per_island_per_cycle):
        request_kwargs: dict[str, object] = {
            "model": request.model,
            "instructions": request.system_prompt,
            "input": request.prompt_code,
        }
        if request.temperature is not None:
            request_kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = request.max_output_tokens

        try:
            response = client.responses.create(**request_kwargs)
        except Exception as exc:
            append_sampler_log(
                request.log_path,
                f"sample_index={sample_index} openai_error={type(exc).__name__}: {exc}",
            )
            continue

        raw_completion = response.output_text or ""
        metadata = {
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", request.model),
        }
        _completion_text_path(request.output_dir, sample_index).write_text(raw_completion)
        _completion_metadata_path(request.output_dir, sample_index).write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
        append_sampler_log(
            request.log_path,
            f"sample_index={sample_index} response_id={metadata['response_id']}",
        )
        completions.append(
            GeneratedCompletion(
                cycle_index=request.cycle_index,
                island_id=request.island_id,
                version_generated=request.version_generated,
                sample_index=sample_index,
                raw_completion=raw_completion,
                model=request.model,
                metadata=metadata,
            )
        )
    return completions
