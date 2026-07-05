from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SamplerRequest:
    """Backend-agnostic request for one LLM sampling interaction.

    Input fields:
        backend: Backend name such as `openai` or `deterministic`.
        cycle_index: Current outer evolution cycle.
        island_id: Island that produced `prompt_code`.
        version_generated: Version number expected by upstream FunSearch.
        prompt_code: Python prompt containing prior priority functions and the
            next function header.
        system_prompt: LLM instructions loaded from the configured text file.
        model: External model name.
        candidates_per_island_per_cycle: Number of completions requested from
            this backend call. Island workers set this to 1 so registration can
            happen before the next prompt.
        output_dir: Directory for raw completion artifacts.
        log_path: Per-island sampler log file.
        temperature: Optional model sampling temperature.
        max_output_tokens: Optional model output token cap.

    Output use:
        Passed to `run_sampling_request`, which returns generated completions.
    """

    backend: str
    cycle_index: int
    island_id: int
    version_generated: int
    prompt_code: str
    system_prompt: str
    model: str
    candidates_per_island_per_cycle: int
    output_dir: Path
    log_path: Path
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class GeneratedCompletion:
    """One raw completion returned by a sampler backend.

    Input fields:
        cycle_index: Cycle associated with the request.
        island_id: Island associated with the request.
        version_generated: Version number associated with the request prompt.
        sample_index: Backend-local completion index.
        raw_completion: Raw function-body text returned by the model/backend.
        model: Model name used for generation.
        metadata: Optional backend-specific metadata such as response ids.

    Output use:
        Island samplers materialize this text into a candidate program, then
        evaluate and register it locally.
    """

    cycle_index: int
    island_id: int
    version_generated: int
    sample_index: int
    raw_completion: str
    model: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def append_sampler_log(log_path: Path, message: str) -> None:
    """Append one line to a sampler log file.

    Input:
        log_path: Destination log file path.
        message: Text line to append; trailing newlines are normalized.

    Output:
        Creates parent directories if needed and appends the message to disk.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")
