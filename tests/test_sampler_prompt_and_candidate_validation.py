from __future__ import annotations

from pathlib import Path

from funsearch_pipeline.config import ProgramDatabaseSettings
from funsearch_pipeline.program_database import CycleProgramsDatabase
from funsearch_pipeline.program_database import IslandPrompt
from funsearch_pipeline.sampling.island_sampler import _compose_sampler_input
from funsearch_pipeline.sampling.priority_candidate_validation import build_candidate_program


def test_compose_sampler_input_for_single_prior_function() -> None:
    prompt = IslandPrompt(
        island_id=0,
        version_generated=1,
        code="def priority_v0(training_data, ancestry_coordinate, target_variant) -> float:\n    return 0.5\n",
    )

    sampler_input = _compose_sampler_input(prompt)

    assert "one prior priority function, priority_v0" in sampler_input
    assert "complete definition of priority_v1" in sampler_input
    assert sampler_input.endswith(prompt.code)


def test_compose_sampler_input_for_two_prior_functions() -> None:
    prompt = IslandPrompt(
        island_id=0,
        version_generated=2,
        code=(
            "def priority_v0(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.4\n\n"
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.5\n"
        ),
    )

    sampler_input = _compose_sampler_input(prompt)

    assert "priority_v1 is a higher-scored improvement over priority_v0" in sampler_input
    assert "identify what made priority_v1 better than priority_v0" in sampler_input
    assert "You may also try a completely novel direction of improvement." in sampler_input
    assert "complete definition of priority_v2" in sampler_input
    assert sampler_input.endswith(prompt.code)


def test_build_candidate_program_accepts_full_function_definition() -> None:
    seed_text = Path("Collaterals/Run1/funsearch_priority_seed_example.py").read_text()
    database = CycleProgramsDatabase.from_seed_program_text(
        settings=ProgramDatabaseSettings(
            functions_per_prompt=2,
            num_islands=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30000,
        ),
        seed_program_text=seed_text,
        function_to_evolve="priority",
    )

    candidate_program = build_candidate_program(
        template=database.template,
        function_to_evolve="priority",
        island_id=0,
        version_generated=1,
        raw_completion=(
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.75\n"
        ),
        sample_index=0,
    )

    assert "def priority(" in candidate_program.program_source
    assert "return 0.75" in candidate_program.program_source
    assert candidate_program.raw_completion.strip().startswith("return 0.75")


def test_island_prompt_generation_handles_four_space_candidate_bodies() -> None:
    seed_text = (
        "def priority(training_data, ancestry_coordinate, target_variant) -> float:\n"
        "    return 0.5\n"
    )
    database = CycleProgramsDatabase.from_seed_program_text(
        settings=ProgramDatabaseSettings(
            functions_per_prompt=2,
            num_islands=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30000,
        ),
        seed_program_text=seed_text,
        function_to_evolve="priority",
    )
    database.register_seed({"mean": 0.5})

    candidate_program = build_candidate_program(
        template=database.template,
        function_to_evolve="priority",
        island_id=0,
        version_generated=1,
        raw_completion=(
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    try:\n"
            "        return 0.75\n"
            "    except Exception:\n"
            "        return 0.25\n"
        ),
        sample_index=0,
    )

    shard = database.export_island_shard(0)
    improved = shard.register_candidate(candidate_program, {"mean": 0.75})
    prompt = shard.get_prompt()

    assert improved is True
    assert "def priority_v0" in prompt.code
    assert "def priority_v1" in prompt.code
    assert '  """Improved version of `priority_v0`."""' in prompt.code
    assert "  try:" in prompt.code