from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
import pickle
from typing import Any

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as upstream_config
from funsearch.implementation import evaluator as upstream_evaluator
from funsearch.implementation import programs_database as upstream_programs_database
from funsearch_pipeline.config import ProgramDatabaseSettings


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

        code, version_generated = self.island.get_prompt()
        return IslandPrompt(
            island_id=self.island_id,
            version_generated=version_generated,
            code=code,
        )

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

        evolved_function, program_source = upstream_evaluator._sample_to_program(
            raw_completion,
            prompt.version_generated,
            template,
            function_to_evolve,
        )
        return CandidateProgram(
            island_id=self.island_id,
            version_generated=prompt.version_generated,
            sample_index=sample_index,
            raw_completion=raw_completion,
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
            scores_per_test: Pair-level scores plus final reduced score under
                the `mean` key.

        Output:
            `True` when this candidate improves the shard's best reduced score;
            otherwise `False`.
        """

        self.island.register_program(candidate_program.evolved_function, scores_per_test)
        reduced_score = upstream_programs_database._reduce_score(scores_per_test)
        if reduced_score > self.best_score:
            self.best_score = reduced_score
            self.best_program = candidate_program.evolved_function
            self.best_scores_per_test = dict(scores_per_test)
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
            scores_per_test: Bootstrap scores returned by the evaluator.

        Output:
            Mutates all upstream islands so every island starts from the same
            seed priority function.
        """

        seed_function = copy.deepcopy(self._template.get_function(self._function_to_evolve))
        for island_id in range(self.num_islands):
            self._database._register_program_in_island(
                copy.deepcopy(seed_function),
                island_id,
                scores_per_test,
            )

    def get_prompt_for_island(self, island_id: int) -> IslandPrompt:
        """Return a prompt from the main database island state.

        Input:
            island_id: Island in the parent database.

        Output:
            `IslandPrompt` from that island. Cycle workers should prefer
            `IslandShard.get_prompt()` so prompts reflect local registrations.
        """

        code, version_generated = self._database._islands[island_id].get_prompt()
        return IslandPrompt(island_id=island_id, version_generated=version_generated, code=code)

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

        evolved_function, program_source = upstream_evaluator._sample_to_program(
            raw_completion,
            prompt.version_generated,
            self._template,
            self._function_to_evolve,
        )
        return CandidateProgram(
            island_id=prompt.island_id,
            version_generated=prompt.version_generated,
            sample_index=sample_index,
            raw_completion=raw_completion,
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
            scores_per_test: Pair-level scores plus reduced mean score.

        Output:
            Mutates the corresponding parent island.
        """

        self._database._register_program_in_island(
            candidate_program.evolved_function,
            candidate_program.island_id,
            scores_per_test,
        )

    def reset_weak_islands(self) -> None:
        """Reset the weaker half of islands using upstream FunSearch logic.

        Input:
            No arguments; uses current island best scores.

        Output:
            Mutates weak islands in the parent database and reseeds each from a
            stronger island founder.
        """

        self._database.reset_islands()

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
