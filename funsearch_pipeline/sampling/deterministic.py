from __future__ import annotations

from pathlib import Path

from funsearch_pipeline.sampling.interfaces import GeneratedCompletion
from funsearch_pipeline.sampling.interfaces import SamplerRequest
from funsearch_pipeline.sampling.interfaces import append_sampler_log


def _candidate_file(output_dir: Path, sample_index: int) -> Path:
    """Return the deterministic completion artifact path.

    Input:
        output_dir: Directory for this backend interaction.
        sample_index: Backend-local completion index.

    Output:
        Path where the synthetic Python fragment is written.
    """

    return output_dir / f"candidate_{sample_index:03d}.pyfrag"


def generate_deterministic_completions(request: SamplerRequest) -> list[GeneratedCompletion]:
    """Generate synthetic completions for orchestration tests.

    Input:
        request: Sampler request containing cycle/island ids, requested count,
            and output/log paths.

    Output:
        List of deterministic `return <float>` completions, with matching files
        and sampler log entries written to disk.
    """

    request.output_dir.mkdir(parents=True, exist_ok=True)
    append_sampler_log(
        request.log_path,
        (
            f"cycle={request.cycle_index} island={request.island_id} backend=deterministic "
            f"candidates={request.candidates_per_island_per_cycle}"
        ),
    )

    completions: list[GeneratedCompletion] = []
    base_value = request.cycle_index * 1000 + request.island_id * 100
    for sample_index in range(request.candidates_per_island_per_cycle):
        candidate_value = (base_value + sample_index + 1) / 100.0
        raw_completion = f"  return {candidate_value:.6f}\n"
        _candidate_file(request.output_dir, sample_index).write_text(raw_completion)
        append_sampler_log(
            request.log_path,
            f"sample_index={sample_index} synthetic_return_value={candidate_value:.6f}",
        )
        completions.append(
            GeneratedCompletion(
                cycle_index=request.cycle_index,
                island_id=request.island_id,
                version_generated=request.version_generated,
                sample_index=sample_index,
                raw_completion=raw_completion,
                model=request.model,
                metadata={"synthetic_return_value": candidate_value},
            )
        )
    return completions
