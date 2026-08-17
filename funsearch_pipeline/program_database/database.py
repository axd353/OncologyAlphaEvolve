from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
import numpy as np
import pickle
import re
import textwrap
from typing import Any

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as upstream_config
from funsearch.implementation import evaluator as upstream_evaluator
from funsearch.implementation import programs_database as upstream_programs_database
from funsearch_pipeline.config import ProgramDatabaseSettings


def _normalize_function_body_indentation(body: str) -> str:
    """Normalize a function body to the 2-space base indent upstream expects."""

    dedented_body = textwrap.dedent(body).strip("\n")
    if not dedented_body:
        return dedented_body
    return "\n".join(
        f"  {line}" if line.strip() else ""
        for line in dedented_body.splitlines()
    )


def _expected_prompt_version(prompt_code: str, function_to_evolve: str) -> int:
    """Return the version suffix of the final function header in `prompt_code`."""

    pattern = re.compile(rf"^def {re.escape(function_to_evolve)}_v(\d+)\(", re.MULTILINE)
    matches = list(pattern.finditer(prompt_code))
    if not matches:
        raise ValueError(
            f"Prompt code does not contain a versioned {function_to_evolve!r} header."
        )
    return int(matches[-1].group(1))


def _prepare_scores_for_registration(scores_per_test: dict[str, float]) -> dict[str, float]:
    """Keep evaluator fold scores intact and ensure `mean` remains last."""

    if "mean" not in scores_per_test:
        raise ValueError("scores_per_test must include a 'mean' entry.")

    registered_scores: OrderedDict[str, float] = OrderedDict()
    for score_name, score_value in scores_per_test.items():
        if score_name in {"mean", "simplicity_bonus", "combined"}:
            continue
        registered_scores[score_name] = score_value
    registered_scores["mean"] = scores_per_test["mean"]
    return dict(registered_scores)


def _attach_scores_to_cluster(
    island: upstream_programs_database.Island,
    scores_per_test: dict[str, float],
) -> None:
    """Store the named score vector on the matching upstream cluster."""

    signature = upstream_programs_database._get_signature(scores_per_test)
    cluster = island._clusters.get(signature)
    if cluster is None:
        raise KeyError(f"Cluster for signature {signature!r} was not found during registration.")
    cluster._scores_per_test = dict(scores_per_test)


def _ordered_fold_score_names(scores_per_test: dict[str, float]) -> tuple[str, ...]:
    """Return fold-score keys in the stored order used to compute `mean`."""

    fold_score_names = tuple(
        score_name for score_name in scores_per_test if "_fold_" in score_name
    )
    if fold_score_names:
        return fold_score_names
    return tuple(
        score_name
        for score_name in scores_per_test
        if score_name not in {"simplicity", "mean", "simplicity_bonus", "combined"}
    )


def _should_swap_prompt_order(
    lower_mean_scores: dict[str, float],
    higher_mean_scores: dict[str, float],
) -> bool:
    """Prefer the lower-mean program when it is simpler and wins a fold."""

    lower_mean = float(lower_mean_scores["mean"])
    higher_mean = float(higher_mean_scores["mean"])
    if not lower_mean < higher_mean - 1e-12:
        return False

    lower_simplicity = lower_mean_scores.get("simplicity")
    higher_simplicity = higher_mean_scores.get("simplicity")
    if lower_simplicity is None or higher_simplicity is None:
        return False
    if not float(lower_simplicity) > float(higher_simplicity):
        return False

    lower_fold_names = _ordered_fold_score_names(lower_mean_scores)
    higher_fold_names = _ordered_fold_score_names(higher_mean_scores)
    if lower_fold_names != higher_fold_names or not lower_fold_names:
        return False

    return any(
        float(lower_mean_scores[fold_name]) > float(higher_mean_scores[fold_name])
        for fold_name in lower_fold_names
    )


def _build_prompt_prior_summary(
    version_name: str,
    scores_per_test: dict[str, float] | None,
) -> "PromptPriorSummary":
    """Summarize one prior function for sampler-side logging."""

    if scores_per_test is None:
        return PromptPriorSummary(version_name=version_name)

    return PromptPriorSummary(
        version_name=version_name,
        mean_score=(float(scores_per_test["mean"]) if "mean" in scores_per_test else None),
        simplicity_score=(
            float(scores_per_test["simplicity"])
            if "simplicity" in scores_per_test
            else None
        ),
        fold_scores=tuple(
            (fold_name, float(scores_per_test[fold_name]))
            for fold_name in _ordered_fold_score_names(scores_per_test)
        ),
    )


def _build_prompt_order_explanation(
    prior_summaries: tuple["PromptPriorSummary", ...],
    *,
    preferred_second_due_to_simple_fold_win: bool,
) -> str | None:
    """Explain why the visible `priority_v1` was shown second."""

    if len(prior_summaries) == 0:
        return None
    if len(prior_summaries) == 1:
        return "Only one prior function was available, so only priority_v0 was shown."

    v0_summary = prior_summaries[0]
    v1_summary = prior_summaries[1]
    if preferred_second_due_to_simple_fold_win:
        v0_fold_scores = dict(v0_summary.fold_scores)
        winning_fold_details = ", ".join(
            (
                f"{fold_name} ({v1_score} > {v0_fold_scores[fold_name]})"
                if fold_name in v0_fold_scores
                else f"{fold_name} ({v1_score})"
            )
            for fold_name, v1_score in v1_summary.fold_scores
            if fold_name not in v0_fold_scores or v1_score > v0_fold_scores[fold_name]
        )
        return (
            "priority_v1 was shown second because it is the preferred mutation: "
            f"mean={v1_summary.mean_score} < {v0_summary.mean_score}, "
            f"simplicity={v1_summary.simplicity_score} > {v0_summary.simplicity_score}, "
            f"wins_folds=[{winning_fold_details}]"
        )

    return (
        "priority_v1 was shown second because it has the higher mean score "
        f"({v1_summary.mean_score} >= {v0_summary.mean_score}) and the "
        "simplicity-plus-fold override did not apply."
    )


def _build_island_prompt(
    island: upstream_programs_database.Island,
    *,
    island_id: int,
) -> "IslandPrompt":
    """Build an island prompt with the prompt-order override applied."""

    signatures = list(island._clusters.keys())
    if not signatures:
        raise ValueError("Cannot build a prompt from an empty island.")

    cluster_scores = np.array([island._clusters[signature].score for signature in signatures])
    period = island._cluster_sampling_temperature_period
    temperature = island._cluster_sampling_temperature_init * (
        1 - (island._num_programs % period) / period
    )
    probabilities = upstream_programs_database._softmax(cluster_scores, temperature)
    functions_per_prompt = min(len(island._clusters), island._functions_per_prompt)
    chosen_indices = np.random.choice(
        len(signatures),
        size=functions_per_prompt,
        p=probabilities,
    )

    prompt_entries: list[tuple[code_manipulation.Function, float, dict[str, float] | None]] = []
    for chosen_index in np.atleast_1d(chosen_indices):
        signature = signatures[int(chosen_index)]
        cluster = island._clusters[signature]
        prompt_entries.append(
            (
                cluster.sample_program(),
                float(cluster.score),
                getattr(cluster, "_scores_per_test", None),
            )
        )

    sorted_entries = [
        prompt_entries[int(index)]
        for index in np.argsort([entry[1] for entry in prompt_entries])
    ]
    preferred_second_due_to_simple_fold_win = False
    if len(sorted_entries) == 2:
        lower_mean_scores = sorted_entries[0][2]
        higher_mean_scores = sorted_entries[1][2]
        if (
            lower_mean_scores is not None
            and higher_mean_scores is not None
            and _should_swap_prompt_order(lower_mean_scores, higher_mean_scores)
        ):
            sorted_entries = [sorted_entries[1], sorted_entries[0]]
            preferred_second_due_to_simple_fold_win = True

    prior_summaries = tuple(
        _build_prompt_prior_summary(f"priority_v{index}", entry[2])
        for index, entry in enumerate(sorted_entries)
    )

    code = island._generate_prompt([entry[0] for entry in sorted_entries])
    return IslandPrompt(
        island_id=island_id,
        version_generated=_expected_prompt_version(code, island._function_to_evolve),
        code=code,
        preferred_second_due_to_simple_fold_win=preferred_second_due_to_simple_fold_win,
        prior_summaries=prior_summaries,
        shown_second_reason=_build_prompt_order_explanation(
            prior_summaries,
            preferred_second_due_to_simple_fold_win=preferred_second_due_to_simple_fold_win,
        ),
    )


@dataclass(frozen=True)
class PromptPriorSummary:
    """Lightweight sampler-log summary for one prior function in a prompt."""

    version_name: str
    mean_score: float | None = None
    simplicity_score: float | None = None
    fold_scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class IslandPrompt:
    """Prompt generated from the current state of one island.

    Input fields:
        island_id: Island that produced the prompt.
        version_generated: Version suffix expected for the next generated body.
        code: Python prompt containing one or more previous priority functions
            and the header of the next function to complete.

    Output use:
        Passed to the sampler backend and later used to materialize the LLM
        completion into the unversioned priority function.
    """

    island_id: int
    version_generated: int
    code: str
    preferred_second_due_to_simple_fold_win: bool = False
    prior_summaries: tuple[PromptPriorSummary, ...] = ()
    shown_second_reason: str | None = None


@dataclass(frozen=True)
class CandidateProgram:
    """Runnable program assembled from one LLM completion.

    Input fields:
        island_id: Island that requested this candidate.
        version_generated: Prompt version that produced the candidate.
        sample_index: Candidate number within the island/cycle worker.
        raw_completion: Raw text returned by the sampler backend.
        evolved_function: Parsed upstream FunSearch function object.
        program_source: Full runnable Python source used by the evaluator.
        function_name: Unversioned function name to evaluate.

    Output use:
        The evaluator scores this object, then the island shard registers the
        `evolved_function` if evaluation succeeds.
    """

    island_id: int
    version_generated: int | None
    sample_index: int
    raw_completion: str
    evolved_function: code_manipulation.Function
    program_source: str
    function_name: str


@dataclass(frozen=True)
class IslandSummary:
    island_id: int
    best_score: float | None
    num_clusters: int
    num_programs: int


@dataclass(frozen=True)
class DatabaseSummary:
    num_islands: int
    global_best_score: float | None
    islands: tuple[IslandSummary, ...]


@dataclass(frozen=True)
class BestProgramArtifact:
    island_id: int
    reduced_score: float
    scores_per_test: dict[str, float] | None
    program_source: str


@dataclass
class IslandShard:
    """Process-local mutable copy of one upstream FunSearch island.

    Input fields:
        island_id: Original island id in the parent program database.
        island: Deep-copied upstream `Island` instance.
        best_score: Best reduced score known for this island.
        best_program: Best parsed function known for this island, if any.
        best_scores_per_test: Per-test score signature for `best_program`.

    Output use:
        A sampler worker mutates this object during a cycle. The parent runner
        combines completed shards back into the main database at cycle end.
    """

    island_id: int
    island: upstream_programs_database.Island
    best_score: float
    best_program: code_manipulation.Function | None
    best_scores_per_test: dict[str, float] | None

    def get_prompt(self) -> IslandPrompt:
        """Return a prompt from the shard's current island state.

        Input:
            The shard's mutable upstream island.

        Output:
            `IslandPrompt` containing one or two sampled priority functions when
            available, plus the header for the next generated version.
        """

        return _build_island_prompt(self.island, island_id=self.island_id)

    def materialize_candidate(
        self,
        prompt: IslandPrompt,
        raw_completion: str,
        sample_index: int,
        *,
        template: code_manipulation.Program,
        function_to_evolve: str,
    ) -> CandidateProgram:
        """Convert one LLM completion into a runnable candidate program.

        Input:
            prompt: Prompt metadata that generated the completion.
            raw_completion: Candidate function body returned by the LLM.
            sample_index: Candidate number inside this shard worker.
            template: Parsed seed program used by upstream FunSearch.
            function_to_evolve: Unversioned priority function name.

        Output:
            `CandidateProgram` ready for evaluator execution.
        """

        normalized_completion = _normalize_function_body_indentation(raw_completion)
        evolved_function, program_source = upstream_evaluator._sample_to_program(
            normalized_completion,
            prompt.version_generated,
            template,
            function_to_evolve,
        )
        return CandidateProgram(
            island_id=self.island_id,
            version_generated=prompt.version_generated,
            sample_index=sample_index,
            raw_completion=normalized_completion,
            evolved_function=evolved_function,
            program_source=program_source,
            function_name=function_to_evolve,
        )

    def register_candidate(
        self,
        candidate_program: CandidateProgram,
        scores_per_test: dict[str, float],
    ) -> bool:
        """Register an evaluated candidate into this shard's island.

        Input:
            candidate_program: Evaluated candidate to add to the island.
            scores_per_test: Evaluator scores including the base `mean` score.

        Output:
            `True` when this candidate improves the shard's best reduced score;
            otherwise `False`.
        """

        registered_scores = _prepare_scores_for_registration(scores_per_test)
        self.island.register_program(candidate_program.evolved_function, registered_scores)
        _attach_scores_to_cluster(self.island, registered_scores)
        reduced_score = upstream_programs_database._reduce_score(registered_scores)
        if reduced_score > self.best_score:
            self.best_score = reduced_score
            self.best_program = candidate_program.evolved_function
            self.best_scores_per_test = dict(registered_scores)
            return True
        return False


class CycleProgramsDatabase:
    """Cycle-aware wrapper around DeepMind's upstream ProgramsDatabase."""

    def __init__(
        self,
        settings: ProgramDatabaseSettings,
        template: code_manipulation.Program,
        function_to_evolve: str,
    ) -> None:
        upstream_settings = upstream_config.ProgramsDatabaseConfig(
            functions_per_prompt=settings.functions_per_prompt,
            num_islands=settings.num_islands,
            reset_period=10**18,
            cluster_sampling_temperature_init=settings.cluster_sampling_temperature_init,
            cluster_sampling_temperature_period=settings.cluster_sampling_temperature_period,
        )
        self._settings = settings
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._database = upstream_programs_database.ProgramsDatabase(
            upstream_settings,
            template,
            function_to_evolve,
        )

    @classmethod
    def from_seed_program_text(
        cls,
        settings: ProgramDatabaseSettings,
        seed_program_text: str,
        function_to_evolve: str,
    ) -> "CycleProgramsDatabase":
        """Build a database from a seed program file body.

        Input:
            settings: Program database hyperparameters from the config file.
            seed_program_text: Python source containing `function_to_evolve`.
            function_to_evolve: Name of the priority function to evolve.

        Output:
            Empty multi-island database ready for seed registration.
        """

        template = code_manipulation.text_to_program(seed_program_text)
        for function in template.functions:
            function.body = _normalize_function_body_indentation(function.body)
        template.get_function(function_to_evolve)
        return cls(settings=settings, template=template, function_to_evolve=function_to_evolve)

    @property
    def num_islands(self) -> int:
        """Return the configured number of islands.

        Input:
            No arguments; reads immutable database settings.

        Output:
            Integer island count used by the runner to create one shard per
            sampler process.
        """

        return self._settings.num_islands

    @property
    def function_to_evolve(self) -> str:
        """Return the unversioned priority function name.

        Input:
            No arguments; reads the template contract.

        Output:
            Function name passed to samplers and evaluators.
        """

        return self._function_to_evolve

    @property
    def template(self) -> code_manipulation.Program:
        """Return the parsed seed program template.

        Input:
            No arguments; reads the parsed upstream FunSearch template.

        Output:
            Template used by shard workers to materialize generated function
            bodies into runnable programs.
        """

        return self._template

    def build_seed_candidate(self) -> CandidateProgram:
        """Return the seed priority function as an evaluable candidate.

        Input:
            No arguments; uses the configured template function.

        Output:
            `CandidateProgram` for bootstrap evaluation before any sampling.
        """

        evolved_function = copy.deepcopy(self._template.get_function(self._function_to_evolve))
        return CandidateProgram(
            island_id=-1,
            version_generated=None,
            sample_index=-1,
            raw_completion=evolved_function.body,
            evolved_function=evolved_function,
            program_source=str(self._template),
            function_name=self._function_to_evolve,
        )

    def register_seed(self, scores_per_test: dict[str, float]) -> None:
        """Register the seed function into every island.

        Input:
            scores_per_test: Bootstrap evaluator scores containing base `mean`.

        Output:
            Mutates all upstream islands so every island starts from the same
            seed priority function.
        """

        seed_function = copy.deepcopy(self._template.get_function(self._function_to_evolve))
        for island_id in range(self.num_islands):
            registered_scores = _prepare_scores_for_registration(scores_per_test)
            self._database._register_program_in_island(
                copy.deepcopy(seed_function),
                island_id,
                registered_scores,
            )
            _attach_scores_to_cluster(self._database._islands[island_id], registered_scores)

    def get_prompt_for_island(self, island_id: int) -> IslandPrompt:
        """Return a prompt from the main database island state.

        Input:
            island_id: Island in the parent database.

        Output:
            `IslandPrompt` from that island. Cycle workers should prefer
            `IslandShard.get_prompt()` so prompts reflect local registrations.
        """

        return _build_island_prompt(self._database._islands[island_id], island_id=island_id)

    def materialize_candidate(
        self,
        prompt: IslandPrompt,
        raw_completion: str,
        sample_index: int,
    ) -> CandidateProgram:
        """Convert a completion using the parent database template.

        Input:
            prompt: Prompt metadata from the parent database.
            raw_completion: Candidate body returned by a sampler backend.
            sample_index: Candidate number in the current island/cycle.

        Output:
            `CandidateProgram` ready for evaluator execution.
        """

        normalized_completion = _normalize_function_body_indentation(raw_completion)
        evolved_function, program_source = upstream_evaluator._sample_to_program(
            normalized_completion,
            prompt.version_generated,
            self._template,
            self._function_to_evolve,
        )
        return CandidateProgram(
            island_id=prompt.island_id,
            version_generated=prompt.version_generated,
            sample_index=sample_index,
            raw_completion=normalized_completion,
            evolved_function=evolved_function,
            program_source=program_source,
            function_name=self._function_to_evolve,
        )

    def register_candidate(
        self,
        candidate_program: CandidateProgram,
        scores_per_test: dict[str, float],
    ) -> None:
        """Register an evaluated candidate into the parent database.

        Input:
            candidate_program: Candidate to add.
            scores_per_test: Evaluator scores including the base `mean` score.

        Output:
            Mutates the corresponding parent island.
        """

        registered_scores = _prepare_scores_for_registration(scores_per_test)
        self._database._register_program_in_island(
            candidate_program.evolved_function,
            candidate_program.island_id,
            registered_scores,
        )
        _attach_scores_to_cluster(
            self._database._islands[candidate_program.island_id],
            registered_scores,
        )

    def reset_weak_islands(self) -> None:
        """Reset the weaker half of islands using upstream FunSearch logic.

        Input:
            No arguments; uses current island best scores.

        Output:
            Mutates weak islands in the parent database and reseeds each from a
            stronger island founder.
        """

        indices_sorted_by_score = np.argsort(
            self._database._best_score_per_island
            + np.random.randn(len(self._database._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self.num_islands // 2
        reset_island_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_island_ids = indices_sorted_by_score[num_islands_to_reset:]

        for island_id in reset_island_ids:
            self._database._islands[island_id] = upstream_programs_database.Island(
                self._template,
                self._function_to_evolve,
                self._settings.functions_per_prompt,
                self._settings.cluster_sampling_temperature_init,
                self._settings.cluster_sampling_temperature_period,
            )
            self._database._best_score_per_island[island_id] = -float("inf")
            founder_island_id = int(np.random.choice(keep_island_ids))
            founder = copy.deepcopy(self._database._best_program_per_island[founder_island_id])
            founder_scores = self._database._best_scores_per_test_per_island[founder_island_id]
            if founder is None or founder_scores is None:
                continue
            reset_scores = _prepare_scores_for_registration(dict(founder_scores))
            self._database._register_program_in_island(founder, island_id, reset_scores)
            _attach_scores_to_cluster(self._database._islands[island_id], reset_scores)

    def export_island_shards(self) -> list[IslandShard]:
        """Copy every island into an independent shard.

        Input:
            No arguments; reads the parent database's current island state.

        Output:
            One deep-copied `IslandShard` per island. Each shard can be sent to
            a sampler process and mutated without touching the parent database.
        """

        return [self.export_island_shard(island_id) for island_id in range(self.num_islands)]

    def export_island_shard(self, island_id: int) -> IslandShard:
        """Copy one island and its best-score metadata.

        Input:
            island_id: Parent database island to copy.

        Output:
            Independent `IslandShard` for one sampler process.
        """

        best_scores = self._database._best_scores_per_test_per_island[island_id]
        return IslandShard(
            island_id=island_id,
            island=copy.deepcopy(self._database._islands[island_id]),
            best_score=self._database._best_score_per_island[island_id],
            best_program=copy.deepcopy(self._database._best_program_per_island[island_id]),
            best_scores_per_test=(dict(best_scores) if best_scores is not None else None),
        )

    def combine_island_shards(self, shards: list[IslandShard]) -> None:
        """Stitch completed island shards back into the parent database.

        Input:
            shards: Completed sampler shards, one per island id.

        Output:
            Replaces each parent island and best-score metadata with the shard's
            final state at the end of the cycle.
        """

        shard_by_id = {shard.island_id: shard for shard in shards}
        expected_ids = set(range(self.num_islands))
        if set(shard_by_id) != expected_ids:
            raise ValueError(
                "Completed island shards must contain exactly one shard for every island. "
                f"Expected {sorted(expected_ids)}, got {sorted(shard_by_id)}."
            )

        for island_id in range(self.num_islands):
            shard = shard_by_id[island_id]
            self._database._islands[island_id] = shard.island
            self._database._best_score_per_island[island_id] = shard.best_score
            self._database._best_program_per_island[island_id] = shard.best_program
            self._database._best_scores_per_test_per_island[island_id] = shard.best_scores_per_test

    def global_best_score(self) -> float | None:
        """Return the best finite reduced score in the database.

        Input:
            No arguments; reads current island best scores.

        Output:
            Maximum finite score, or `None` before any program has been
            registered.
        """

        finite_scores = [
            score for score in self._database._best_score_per_island if math.isfinite(score)
        ]
        if not finite_scores:
            return None
        return max(finite_scores)

    def build_global_best_program_artifact(self) -> BestProgramArtifact | None:
        """Return the best program across islands as runnable source.

        Input:
            No arguments; reads the current best program and score for each
            island.

        Output:
            `BestProgramArtifact` for the highest-scoring island, or `None`
            when no best program has been registered yet.
        """

        best_island_id: int | None = None
        best_score = -float("inf")
        for island_id, score in enumerate(self._database._best_score_per_island):
            if not math.isfinite(score):
                continue
            if best_island_id is None or score > best_score:
                best_island_id = island_id
                best_score = float(score)

        if best_island_id is None:
            return None

        best_program = self._database._best_program_per_island[best_island_id]
        if best_program is None:
            return None

        program = copy.deepcopy(self._template)
        target_function = program.get_function(self._function_to_evolve)
        target_function.args = best_program.args
        target_function.body = best_program.body
        target_function.return_type = best_program.return_type
        target_function.docstring = best_program.docstring
        best_scores = self._database._best_scores_per_test_per_island[best_island_id]
        return BestProgramArtifact(
            island_id=best_island_id,
            reduced_score=best_score,
            scores_per_test=dict(best_scores) if best_scores is not None else None,
            program_source=str(program),
        )

    def build_summary(self) -> DatabaseSummary:
        """Build a JSON-serializable summary of the current database state.

        Input:
            No arguments; reads island cluster counts and best scores.

        Output:
            `DatabaseSummary` used by experiment snapshots.
        """

        islands: list[IslandSummary] = []
        for island_id, island in enumerate(self._database._islands):
            raw_best_score = self._database._best_score_per_island[island_id]
            best_score = raw_best_score if math.isfinite(raw_best_score) else None
            islands.append(
                IslandSummary(
                    island_id=island_id,
                    best_score=best_score,
                    num_clusters=len(island._clusters),
                    num_programs=island._num_programs,
                )
            )
        return DatabaseSummary(
            num_islands=self.num_islands,
            global_best_score=self.global_best_score(),
            islands=tuple(islands),
        )

    def save_snapshot(
        self,
        pickle_path: Path,
        summary_path: Path,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist the complete database plus a compact JSON summary.

        Input:
            pickle_path: Destination for the full pickled database wrapper.
            summary_path: Destination for the human-readable JSON summary.
            extra_metadata: Optional snapshot metadata such as cycle and stage.

        Output:
            Writes both files to disk for reproducibility and restart analysis.
        """

        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with pickle_path.open("wb") as handle:
            pickle.dump(self, handle)

        summary = asdict(self.build_summary())
        if extra_metadata:
            summary["metadata"] = extra_metadata
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
