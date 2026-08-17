from __future__ import annotations

from pathlib import Path
import numpy as np

from funsearch_pipeline.config import ProgramDatabaseSettings
from funsearch_pipeline.program_database import CycleProgramsDatabase
from funsearch_pipeline.program_database import IslandPrompt
from funsearch_pipeline.program_database import PromptPriorSummary
from funsearch_pipeline.sampling.island_sampler import _compose_sampler_input
from funsearch_pipeline.sampling.island_sampler import _log_prompt_prior_summaries
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


def test_compose_sampler_input_for_two_prior_functions_with_override() -> None:
    prompt = IslandPrompt(
        island_id=0,
        version_generated=2,
        code=(
            "def priority_v0(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.4\n\n"
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.5\n"
        ),
        preferred_second_due_to_simple_fold_win=True,
    )

    sampler_input = _compose_sampler_input(prompt)

    assert "preferred mutation over priority_v0 because it is simpler and wins at least one fold score" in sampler_input
    assert "higher-scored improvement over priority_v0" not in sampler_input
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


def test_build_candidate_program_accepts_unexpected_versioned_function_name() -> None:
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
        version_generated=2,
        raw_completion=(
            "def priority_v7(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.85\n"
        ),
        sample_index=0,
    )

    assert "def priority(" in candidate_program.program_source
    assert "return 0.85" in candidate_program.program_source
    assert candidate_program.raw_completion.strip().startswith("return 0.85")


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
    assert prompt.version_generated == 2
    assert "def priority_v0" in prompt.code
    assert "def priority_v1" in prompt.code
    assert '  """Improved version of `priority_v0`."""' in prompt.code
    assert "  try:" in prompt.code


def test_seed_only_island_prompt_uses_header_version_suffix() -> None:
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

    prompt = database.export_island_shard(0).get_prompt()

    assert prompt.version_generated == 1
    assert "def priority_v0" in prompt.code
    assert "def priority_v1" in prompt.code


def test_registration_keeps_fold_scores_and_mean_without_bonus_scores() -> None:
    seed_text = (
        "def priority(training_data, ancestry_coordinate, target_variant) -> float:\n"
        "    radius = 1.0\n"
        "    if training_data is None:\n"
        "        return radius\n"
        "    return radius\n"
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
    database.register_seed(
        {
            "pair_fold_1": 0.45,
            "pair_fold_2": 0.55,
            "simplicity": -999.0,
            "mean": 0.5,
        }
    )

    shard = database.export_island_shard(0)
    assert shard.best_scores_per_test is not None
    assert list(shard.best_scores_per_test) == ["pair_fold_1", "pair_fold_2", "simplicity", "mean"]
    assert shard.best_scores_per_test["mean"] == 0.5
    assert shard.best_score == 0.5

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

    improved = shard.register_candidate(
        candidate_program,
        {
            "pair_fold_1": 0.50,
            "pair_fold_2": 0.52,
            "simplicity": -1.0,
            "mean": 0.51,
        },
    )

    assert improved is True
    assert shard.best_scores_per_test is not None
    assert list(shard.best_scores_per_test) == ["pair_fold_1", "pair_fold_2", "simplicity", "mean"]
    assert shard.best_scores_per_test["pair_fold_1"] == 0.50
    assert shard.best_scores_per_test["pair_fold_2"] == 0.52
    assert shard.best_scores_per_test["mean"] == 0.51
    assert shard.best_score == 0.51


def test_reset_weak_islands_copies_founder_scores_without_recomputing() -> None:
    seed_text = (
        "def priority(training_data, ancestry_coordinate, target_variant) -> float:\n"
        "    radius = 1.0\n"
        "    if training_data is None:\n"
        "        return radius\n"
        "    return radius\n"
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
    database.register_seed(
        {
            "pair_fold_1": 0.45,
            "pair_fold_2": 0.55,
            "simplicity": -999.0,
            "mean": 0.5,
        }
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
    founder_scores = {
        "pair_fold_1": 0.60,
        "pair_fold_2": 0.56,
        "simplicity": -1.0,
        "mean": 0.58,
    }
    database.register_candidate(candidate_program, founder_scores)

    database.reset_weak_islands()

    reset_shard = database.export_island_shard(1)
    assert reset_shard.best_scores_per_test is not None
    assert reset_shard.best_scores_per_test == founder_scores
    assert reset_shard.best_score == founder_scores["mean"]


def test_prompt_order_prefers_simpler_lower_mean_when_it_wins_a_fold(monkeypatch) -> None:
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
    database.register_seed(
        {
            "pair_fold_1": 0.45,
            "pair_fold_2": 0.55,
            "simplicity": -999.0,
            "mean": 0.5,
        }
    )

    higher_mean_program = build_candidate_program(
        template=database.template,
        function_to_evolve="priority",
        island_id=0,
        version_generated=1,
        raw_completion=(
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    score = 0.7\n"
            "    score += 0.1\n"
            "    return score\n"
        ),
        sample_index=0,
    )
    database.register_candidate(
        higher_mean_program,
        {
            "pair_fold_1": 0.58,
            "pair_fold_2": 0.64,
            "simplicity": -20.0,
            "mean": 0.61,
        },
    )

    lower_mean_program = build_candidate_program(
        template=database.template,
        function_to_evolve="priority",
        island_id=0,
        version_generated=1,
        raw_completion=(
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.65\n"
        ),
        sample_index=1,
    )
    database.register_candidate(
        lower_mean_program,
        {
            "pair_fold_1": 0.63,
            "pair_fold_2": 0.50,
            "simplicity": -5.0,
            "mean": 0.565,
        },
    )

    def fake_choice(values, size=None, replace=True, p=None):
        if size is not None:
            return np.array([1, 2], dtype=int)
        return values[0]

    monkeypatch.setattr(
        "funsearch.implementation.programs_database.np.random.choice",
        fake_choice,
    )

    prompt = database.export_island_shard(0).get_prompt()

    assert "score = 0.7" in prompt.code
    assert "return 0.65" in prompt.code
    assert prompt.code.index("score = 0.7") < prompt.code.index("return 0.65")
    assert tuple(summary.version_name for summary in prompt.prior_summaries) == (
        "priority_v0",
        "priority_v1",
    )
    assert prompt.prior_summaries[0].mean_score == 0.61
    assert prompt.prior_summaries[1].mean_score == 0.565
    assert dict(prompt.prior_summaries[1].fold_scores)["pair_fold_1"] == 0.63
    assert prompt.shown_second_reason is not None
    assert "simplicity=-5.0 > -20.0" in prompt.shown_second_reason
    assert "wins_folds=[pair_fold_1 (0.63 > 0.58)]" in prompt.shown_second_reason


def test_prompt_order_keeps_higher_mean_second_without_override(monkeypatch) -> None:
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
    database.register_seed(
        {
            "pair_fold_1": 0.45,
            "pair_fold_2": 0.55,
            "simplicity": -999.0,
            "mean": 0.5,
        }
    )

    higher_mean_program = build_candidate_program(
        template=database.template,
        function_to_evolve="priority",
        island_id=0,
        version_generated=1,
        raw_completion=(
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    score = 0.7\n"
            "    score += 0.1\n"
            "    return score\n"
        ),
        sample_index=0,
    )
    database.register_candidate(
        higher_mean_program,
        {
            "pair_fold_1": 0.58,
            "pair_fold_2": 0.64,
            "simplicity": -20.0,
            "mean": 0.61,
        },
    )

    lower_mean_program = build_candidate_program(
        template=database.template,
        function_to_evolve="priority",
        island_id=0,
        version_generated=1,
        raw_completion=(
            "def priority_v1(training_data, ancestry_coordinate, target_variant) -> float:\n"
            "    return 0.65\n"
        ),
        sample_index=1,
    )
    database.register_candidate(
        lower_mean_program,
        {
            "pair_fold_1": 0.54,
            "pair_fold_2": 0.55,
            "simplicity": -5.0,
            "mean": 0.545,
        },
    )

    def fake_choice(values, size=None, replace=True, p=None):
        if size is not None:
            return np.array([1, 2], dtype=int)
        return values[0]

    monkeypatch.setattr(
        "funsearch.implementation.programs_database.np.random.choice",
        fake_choice,
    )

    prompt = database.export_island_shard(0).get_prompt()

    assert "return 0.65" in prompt.code
    assert "score = 0.7" in prompt.code
    assert prompt.code.index("return 0.65") < prompt.code.index("score = 0.7")
    assert prompt.shown_second_reason is not None
    assert "higher mean score (0.61 >= 0.545)" in prompt.shown_second_reason


def test_log_prompt_prior_summaries_writes_scores_and_reason(tmp_path: Path) -> None:
    log_path = tmp_path / "sampler.log"
    prompt = IslandPrompt(
        island_id=3,
        version_generated=2,
        code="def priority_v0(...):\n  return 0.1\n",
        prior_summaries=(
            PromptPriorSummary(
                version_name="priority_v0",
                mean_score=0.61,
                simplicity_score=-20.0,
                fold_scores=(("pair_fold_1", 0.58), ("pair_fold_2", 0.64)),
            ),
            PromptPriorSummary(
                version_name="priority_v1",
                mean_score=0.565,
                simplicity_score=-5.0,
                fold_scores=(("pair_fold_1", 0.63), ("pair_fold_2", 0.50)),
            ),
        ),
        shown_second_reason=(
            "priority_v1 was shown second because it is the preferred mutation: "
            "mean=0.565 < 0.61, simplicity=-5.0 > -20.0, "
            "wins_folds=[pair_fold_1 (0.63 > 0.58)]"
        ),
    )

    _log_prompt_prior_summaries(log_path, prompt, sample_index=2)

    log_text = log_path.read_text()
    assert "sample_index=2 prompt_prior_summary_begin island=3" in log_text
    assert "version=priority_v0 mean=0.61 simplicity=-20.0 fold_scores=[pair_fold_1=0.58, pair_fold_2=0.64]" in log_text
    assert "version=priority_v1 mean=0.565 simplicity=-5.0 fold_scores=[pair_fold_1=0.63, pair_fold_2=0.5]" in log_text
    assert "prompt_order_reason priority_v1 was shown second because it is the preferred mutation" in log_text
    assert "sample_index=2 prompt_prior_summary_end island=3" in log_text