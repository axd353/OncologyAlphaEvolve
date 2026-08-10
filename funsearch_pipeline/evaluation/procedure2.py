from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import ast
import hashlib
import inspect
import json
import logging
import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from GenomicsHelpers.effect_size_calculator import effect_size_calculator
from GenomicsHelpers.oracle_data_adapter import DEFAULT_COVARIATE_FIELDS
from GenomicsHelpers.oracle_data_adapter import DEFAULT_LABEL_FIELD
from GenomicsHelpers.oracle_data_adapter import DOSAGE_COLUMN_PREFIX
from GenomicsHelpers.oracle_data_adapter import iter_training_records
from GenomicsHelpers.oracle_data_adapter import list_record_field_names
from GenomicsHelpers.oracle_data_adapter import read_ancestry_coordinate
from GenomicsHelpers.oracle_data_adapter import read_label
from GenomicsHelpers.oracle_data_adapter import read_optional_covariates
from GenomicsHelpers.oracle_data_adapter import read_variant_dosage

from funsearch_pipeline.config import DatasetPairConfig
from funsearch_pipeline.config import EvaluatorSettings
from funsearch_pipeline.evaluation.interfaces import EvaluatedCandidate
from funsearch_pipeline.evaluation.interfaces import PairScore
from funsearch_pipeline.priority_tools.contracts import PriorityAncestryCoordinate
from funsearch_pipeline.priority_tools.contracts import PriorityTargetVariant
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingData
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingRecord
from funsearch_pipeline.program_database.database import CandidateProgram


PriorityFunction = Callable[
    [PriorityTrainingData, PriorityAncestryCoordinate, PriorityTargetVariant],
    float,
]

_PRIORITY_TOOLS_IMPORTS = (
    "from __future__ import annotations\n"
    "from funsearch_pipeline.priority_tools import PriorityAncestryCoordinate\n"
    "from funsearch_pipeline.priority_tools import PriorityTargetVariant\n"
    "from funsearch_pipeline.priority_tools import PriorityTrainingData\n"
    "from funsearch_pipeline.priority_tools import ancestry_novelty_score\n"
    "from funsearch_pipeline.priority_tools import dosage_entropy_by_cumulative_radius\n"
    "from funsearch_pipeline.priority_tools import dosage_entropy_by_interval\n"
    "from funsearch_pipeline.priority_tools import equal_count_interval_densities\n"
    "from funsearch_pipeline.priority_tools import equal_count_intervals\n"
    "from funsearch_pipeline.priority_tools import effect_size_by_cumulative_radius\n"
    "from funsearch_pipeline.priority_tools import effect_size_by_interval\n"
    "from funsearch_pipeline.priority_tools import effect_size_standard_error_by_cumulative_radius\n"
    "from funsearch_pipeline.priority_tools import effect_size_standard_error_by_interval\n"
    "from funsearch_pipeline.priority_tools import label_entropy_by_cumulative_radius\n"
    "from funsearch_pipeline.priority_tools import minimum_radius_for_training_percentage\n"
    "from funsearch_pipeline.priority_tools import radius_for_percentage\n\n"
    "from funsearch_pipeline.priority_tools import standardized_effect_change_by_interval\n"
    "from funsearch_pipeline.priority_tools import target_ld_similarity_by_cumulative_radius\n\n"
)

_EXPECTED_PRIORITY_PARAMETERS = (
    "training_data",
    "ancestry_coordinate",
    "target_variant",
)
_DEFAULT_BOOTSTRAP_ITERATIONS = 200
_DEFAULT_CALIBRATION_PARTITIONS = 1
_DEFAULT_SCORING_PARTITIONS = 1


@dataclass(frozen=True)
class PreparedFoldArtifacts:
    fold_index: int
    calibration_pickle: str
    scoring_pickle: str


@dataclass(frozen=True)
class PreparedDatasetPair:
    name: str
    raw_training_pickles: tuple[str, ...]
    raw_testing_pickles: tuple[str, ...]
    has_additional_covariates: bool
    oracle_train_pickle: str
    fold_artifacts: tuple[PreparedFoldArtifacts, ...]


@dataclass(frozen=True)
class CalibrationModel:
    intercept: float
    covariate_coefficients: np.ndarray
    oracle_coefficients: np.ndarray
    penalty: float
    calibration_log_loss: float


@dataclass(frozen=True)
class FoldEvaluationResult:
    name: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PairEvaluationResult:
    fold_results: tuple[FoldEvaluationResult, ...]
    metadata: dict[str, Any]


class Procedure2PriorityEvaluator:
    """Oracle Procedure 2 evaluator for FunSearch priority functions."""

    def __init__(
        self,
        *,
        settings: EvaluatorSettings,
        function_name: str,
        random_seed: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create the Procedure 2 evaluator.

        Input:
            settings: Evaluator subsection from the JSON config.
            function_name: Unversioned priority function to execute.

        Output:
            Evaluator instance that can prepare and score configured data pairs.
        """

        self._settings = settings
        self._function_name = function_name
        self._random_seed = random_seed
        self._logger = logger
        self._prepared_pairs: tuple[PreparedDatasetPair, ...] = ()

    @property
    def prepared_pairs(self) -> tuple[PreparedDatasetPair, ...]:
        """Return prepared dataset-pair metadata.

        Input:
            No arguments; reads the last `prepare` result.

        Output:
            Tuple describing oracle-train and per-fold calibration/scoring artifact
            paths for each configured pair.
        """

        return self._prepared_pairs

    def prepare(self, experiment_dir: Path) -> None:
        """Create or load the Procedure 2 preprocessing artifacts.

        Input:
            experiment_dir: Root experiment directory.

        Output:
            Creates or loads per-pair oracle-train plus fold-specific calibration and scoring
            pickles, plus `procedure2_layout.json` describing their paths.
        """

        preprocessed_root = experiment_dir / self._settings.preprocessed_dirname
        preprocessed_root.mkdir(parents=True, exist_ok=True)
        manifest_path = preprocessed_root / "procedure2_layout.json"

        if manifest_path.exists():
            manifest_num_folds, prepared_pairs = _load_prepared_pairs_manifest(manifest_path)
            if manifest_num_folds != self._settings.num_folds:
                raise ValueError(
                    "Prepared evaluator manifest num_folds does not match current config: "
                    f"manifest={manifest_num_folds} config={self._settings.num_folds}."
                )
            self._prepared_pairs = prepared_pairs
            return

        prepared_pairs: list[PreparedDatasetPair] = []
        for dataset_pair in self._settings.dataset_pairs:
            prepared_pairs.append(self._prepare_pair(preprocessed_root, dataset_pair))

        self._prepared_pairs = tuple(prepared_pairs)
        manifest_path.write_text(
            json.dumps(
                {
                    "backend": "procedure2",
                    "function_name": self._function_name,
                    "oracle_train_fraction": self._settings.oracle_train_fraction,
                    "metric": self._settings.metric,
                    "calibration_penalties": self._settings.calibration_penalties,
                    "num_folds": self._settings.num_folds,
                    "dataset_pairs": [asdict(pair) for pair in self._prepared_pairs],
                },
                indent=2,
                sort_keys=True,
            )
        )

    def _prepare_pair(
        self,
        preprocessed_root: Path,
        dataset_pair: DatasetPairConfig,
    ) -> PreparedDatasetPair:
        """Prepare persisted oracle-train plus fold-specific calibration/scoring pickles.

        Input:
            preprocessed_root: Root preprocessing directory.
            dataset_pair: Pair config from the JSON file.

        Output:
            `PreparedDatasetPair` containing artifact paths for evaluation.
        """

        pair_root = preprocessed_root / dataset_pair.name
        pair_root.mkdir(parents=True, exist_ok=True)
        oracle_train_pickle = dataset_pair.oracle_train_pickle or pair_root / "oracle_train.pkl"
        fold_artifacts = _build_fold_artifacts(
            pair_root=pair_root,
            calibration_base_path=dataset_pair.calibration_pickle,
            scoring_base_path=dataset_pair.scoring_pickle,
            num_folds=self._settings.num_folds,
        )
        prepared_pair = PreparedDatasetPair(
            name=dataset_pair.name,
            raw_training_pickles=tuple(str(path) for path in dataset_pair.training_pickles),
            raw_testing_pickles=tuple(str(path) for path in dataset_pair.testing_pickles),
            has_additional_covariates=dataset_pair.has_additional_covariates,
            oracle_train_pickle=str(oracle_train_pickle),
            fold_artifacts=fold_artifacts,
        )

        if all(
            path is not None
            for path in (
                dataset_pair.oracle_train_pickle,
                dataset_pair.calibration_pickle,
                dataset_pair.scoring_pickle,
            )
        ):
            missing_paths = [
                path
                for path in _iter_prepared_artifact_paths(prepared_pair)
                if not Path(path).exists()
            ]
            if missing_paths:
                raise FileNotFoundError(f"Prepared evaluator pickle(s) not found: {missing_paths}")
            return prepared_pair

        if all(Path(path).exists() for path in _iter_prepared_artifact_paths(prepared_pair)):
            return prepared_pair

        combined_training, training_imputed_counts = _impute_missing_feature_columns(
            _load_and_combine_pickles(dataset_pair.training_pickles)
        )
        combined_training_sample_count = _dataset_length(combined_training)
        oracle_train = combined_training
        combined_scoring, scoring_imputed_counts = _impute_missing_feature_columns(
            _load_and_combine_pickles(dataset_pair.testing_pickles)
        )
        scoring_sample_count = _dataset_length(combined_scoring)
        calibration_and_scoring_folds = _split_scoring_data_into_folds(
            combined_scoring,
            num_folds=self._settings.num_folds,
            random_seed=_stable_prepare_seed(self._random_seed, dataset_pair.name),
        )

        _write_pickle(prepared_pair.oracle_train_pickle, oracle_train)
        fold_sizes: list[int] = []
        for fold_artifact, (calibration, scoring) in zip(
            prepared_pair.fold_artifacts,
            calibration_and_scoring_folds,
        ):
            _write_pickle(fold_artifact.calibration_pickle, calibration)
            _write_pickle(fold_artifact.scoring_pickle, scoring)
            fold_sizes.append(_dataset_length(scoring))
        if self._logger is not None:
            self._logger.info(
                "Imputed missing values for dataset pair %s: training=%s (rows=%d) scoring=%s (rows=%d)",
                dataset_pair.name,
                _format_imputation_summary(
                    training_imputed_counts,
                    combined_training_sample_count,
                ),
                combined_training_sample_count,
                _format_imputation_summary(
                    scoring_imputed_counts,
                    scoring_sample_count,
                ),
                scoring_sample_count,
            )
            self._logger.info(
                "Prepared Procedure 2 dataset pair %s from training_sources=%s testing_sources=%s "
                "oracle_train_pickle=%s oracle_train_samples=%d num_folds=%d scoring_fold_sizes=%s",
                dataset_pair.name,
                [str(path) for path in dataset_pair.training_pickles],
                [str(path) for path in dataset_pair.testing_pickles],
                prepared_pair.oracle_train_pickle,
                _dataset_length(oracle_train),
                self._settings.num_folds,
                fold_sizes,
            )
            for fold_artifact, (calibration, scoring) in zip(
                prepared_pair.fold_artifacts,
                calibration_and_scoring_folds,
            ):
                self._logger.info(
                    "Prepared Procedure 2 dataset pair %s fold=%d calibration_pickle=%s calibration_samples=%d "
                    "scoring_pickle=%s scoring_samples=%d",
                    dataset_pair.name,
                    fold_artifact.fold_index,
                    fold_artifact.calibration_pickle,
                    _dataset_length(calibration),
                    fold_artifact.scoring_pickle,
                    _dataset_length(scoring),
                )
        return prepared_pair

    def evaluate_candidate(self, candidate: CandidateProgram) -> EvaluatedCandidate | None:
        """Evaluate one candidate with Procedure 2 oracle calibration.

        Input:
            candidate: Runnable priority-function candidate.

        Output:
            `EvaluatedCandidate` when all configured pairs score successfully;
            `None` when the priority function is invalid or cannot be scored.
        """

        if not self._prepared_pairs:
            return None

        try:
            priority_function = _load_priority_function(candidate.program_source, self._function_name)
            _validate_priority_signature(priority_function)
            pair_results = self._evaluate_pairs(candidate)
        except Exception:
            if self._logger is not None:
                self._logger.exception(
                    "Procedure2 evaluation failed for candidate island=%s sample=%s",
                    candidate.island_id,
                    candidate.sample_index,
                )
            return None

        pair_scores = tuple(
            PairScore(name=fold_result.name, score=fold_result.score)
            for result in pair_results.values()
            for fold_result in result.fold_results
        )
        reduced_score = float(np.mean([pair_score.score for pair_score in pair_scores]))
        return EvaluatedCandidate(
            candidate=candidate,
            pair_scores=pair_scores,
            reduced_score=reduced_score,
            auxiliary_scores={
                "simplicity": _score_priority_simplicity(
                    candidate.program_source,
                    self._function_name,
                ),
            },
            metadata={
                "procedure2_pairs": {
                    pair_name: result.metadata for pair_name, result in pair_results.items()
                },
            },
        )

    def _evaluate_pairs(
        self,
        candidate: CandidateProgram,
    ) -> dict[str, "PairEvaluationResult"]:
        """Score all prepared pairs, one worker process per pair when >1.

        A single pair runs inline to avoid process overhead. Multiple pairs are
        each scored in their own worker process so they run concurrently; the OS
        schedules the processes onto available cores, so no CPU pinning is
        needed.
        """

        if len(self._prepared_pairs) <= 1:
            return {
                pair.name: _evaluate_prepared_pair(
                    self._settings,
                    self._function_name,
                    pair,
                    candidate.program_source,
                    self._logger.name if self._logger is not None else None,
                )
                for pair in self._prepared_pairs
            }

        pair_results: dict[str, PairEvaluationResult] = {}
        with ProcessPoolExecutor(max_workers=len(self._prepared_pairs)) as executor:
            future_to_name = {
                executor.submit(
                    _evaluate_prepared_pair,
                    self._settings,
                    self._function_name,
                    pair,
                    candidate.program_source,
                    self._logger.name if self._logger is not None else None,
                ): pair.name
                for pair in self._prepared_pairs
            }
            for future in as_completed(future_to_name):
                pair_results[future_to_name[future]] = future.result()
        # Preserve the configured pair order regardless of completion order so
        # the program-database score signature stays deterministic.
        return {pair.name: pair_results[pair.name] for pair in self._prepared_pairs}


def _evaluate_prepared_pair(
    settings: EvaluatorSettings,
    function_name: str,
    prepared_pair: PreparedDatasetPair,
    candidate_source: str,
    logger_name: str | None = None,
) -> "PairEvaluationResult":
    """Score one prepared training/testing pair for one priority function.

    This is a module-level function so it can run inside a worker process. It
    reloads the priority function from ``candidate_source`` because compiled
    functions are not picklable across processes.
    """

    _log_worker_cpu_binding(
        logger_name,
        role="procedure2_pair_worker",
        pair_name=prepared_pair.name,
    )

    priority_function = _load_priority_function(candidate_source, function_name)
    _validate_priority_signature(priority_function)

    oracle_train = _read_pickle(prepared_pair.oracle_train_pickle)
    variant_names = _list_variant_names(oracle_train)
    if not variant_names:
        raise ValueError(f"No dosage columns found for pair {prepared_pair.name!r}.")

    fold_results: list[FoldEvaluationResult] = []
    for fold_artifact in prepared_pair.fold_artifacts:
        calibration = _read_pickle(fold_artifact.calibration_pickle)
        scoring = _read_pickle(fold_artifact.scoring_pickle)

        calibration_labels = _extract_labels(calibration)
        calibration_oracle_features = _build_calibration_oracle_feature_matrix(
            training_data=oracle_train,
            calibration_data=calibration,
            variant_names=variant_names,
            candidate_source=candidate_source,
            function_name=function_name,
            calibration_partitions=int(
                getattr(settings, "calibration_partitions", _DEFAULT_CALIBRATION_PARTITIONS)
            ),
        )
        calibration_covariates = _extract_covariates(
            calibration,
            include_covariates=prepared_pair.has_additional_covariates,
        )
        calibration_model = _fit_best_calibration_model(
            oracle_features=calibration_oracle_features,
            covariates=calibration_covariates,
            labels=calibration_labels,
            penalties=settings.calibration_penalties,
        )

        oracle_training_for_scoring = _combine_data_objects([oracle_train, calibration])
        scoring_labels = _extract_labels(scoring)
        scoring_oracle_features = _build_scoring_oracle_feature_matrix(
            training_data=oracle_training_for_scoring,
            scoring_data=scoring,
            variant_names=variant_names,
            candidate_source=candidate_source,
            function_name=function_name,
            scoring_partitions=int(
                getattr(settings, "scoring_partitions", _DEFAULT_SCORING_PARTITIONS)
            ),
        )
        scoring_covariates = _extract_covariates(
            scoring,
            include_covariates=prepared_pair.has_additional_covariates,
        )
        risk_scores = _predict_linear_score(
            calibration_model,
            oracle_features=scoring_oracle_features,
            covariates=scoring_covariates,
        )
        direct_auc = _safe_roc_auc(scoring_labels, risk_scores)
        bootstrap_auc_median = _bootstrap_auc_median(
            labels=scoring_labels,
            risk_scores=risk_scores,
            bootstrap_iterations=int(
                getattr(settings, "bootstrap_iterations", _DEFAULT_BOOTSTRAP_ITERATIONS)
            ),
            random_seed=_stable_bootstrap_seed(
                candidate_source,
                f"{prepared_pair.name}_fold_{fold_artifact.fold_index}",
            ),
        )
        fold_name = f"{prepared_pair.name}_fold_{fold_artifact.fold_index}"
        fold_results.append(
            FoldEvaluationResult(
                name=fold_name,
                score=bootstrap_auc_median,
                metadata={
                    "fold_index": fold_artifact.fold_index,
                    "direct_auc": direct_auc,
                    "bootstrap_auc_median": bootstrap_auc_median,
                    "bootstrap_iterations": int(
                        getattr(settings, "bootstrap_iterations", _DEFAULT_BOOTSTRAP_ITERATIONS)
                    ),
                    "selected_calibration_penalty": calibration_model.penalty,
                    "calibration_log_loss": calibration_model.calibration_log_loss,
                    "num_variants": len(variant_names),
                    "num_oracle_train_samples": _dataset_length(oracle_train),
                    "num_calibration_samples": _dataset_length(calibration),
                    "num_scoring_samples": _dataset_length(scoring),
                    "used_additional_covariates": prepared_pair.has_additional_covariates,
                    "calibration_pickle": fold_artifact.calibration_pickle,
                    "scoring_pickle": fold_artifact.scoring_pickle,
                },
            )
        )

    return PairEvaluationResult(
        fold_results=tuple(fold_results),
        metadata={
            "num_folds": len(fold_results),
            "num_variants": len(variant_names),
            "num_oracle_train_samples": _dataset_length(oracle_train),
            "used_additional_covariates": prepared_pair.has_additional_covariates,
            "mean_score": float(np.mean([result.score for result in fold_results])),
            "folds": {result.name: result.metadata for result in fold_results},
        },
    )


def _load_prepared_pairs_manifest(manifest_path: Path) -> tuple[int, tuple[PreparedDatasetPair, ...]]:
    manifest = json.loads(manifest_path.read_text())
    prepared_pairs = tuple(
        PreparedDatasetPair(
            name=str(raw_pair["name"]),
            raw_training_pickles=tuple(raw_pair.get("raw_training_pickles", ())),
            raw_testing_pickles=tuple(raw_pair.get("raw_testing_pickles", ())),
            has_additional_covariates=bool(raw_pair["has_additional_covariates"]),
            oracle_train_pickle=str(raw_pair["oracle_train_pickle"]),
            fold_artifacts=tuple(
                PreparedFoldArtifacts(
                    fold_index=int(raw_fold["fold_index"]),
                    calibration_pickle=str(raw_fold["calibration_pickle"]),
                    scoring_pickle=str(raw_fold["scoring_pickle"]),
                )
                for raw_fold in (
                    raw_pair["fold_artifacts"]
                    if "fold_artifacts" in raw_pair
                    else (
                        {
                            "fold_index": 1,
                            "calibration_pickle": raw_pair["calibration_pickle"],
                            "scoring_pickle": raw_pair["scoring_pickle"],
                        },
                    )
                )
            ),
        )
        for raw_pair in manifest.get("dataset_pairs", [])
    )
    return int(manifest.get("num_folds", 1)), prepared_pairs


def _read_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _load_and_combine_pickles(paths: Sequence[str | Path]) -> Any:
    if not paths:
        raise ValueError("At least one pickle path is required.")
    return _combine_data_objects([_read_pickle(path) for path in paths])


def _impute_missing_feature_columns(data: Any) -> tuple[Any, dict[str, int]]:
    """Impute missing dosage and covariate values before persisting artifacts.

    Dosage columns are mean-imputed (continuous expected allele dosages) and the
    categorical covariate columns are mode-imputed. Covariate columns are cast to
    float so pandas nullable ``NA`` never reaches the downstream feature builders.
    Returns the imputed data plus a per-column count of imputed values.
    """

    if not isinstance(data, pd.DataFrame):
        return data, {}

    imputed = data.copy()
    imputed_counts: dict[str, int] = {}

    dosage_columns = [
        column
        for column in imputed.columns
        if str(column).startswith(DOSAGE_COLUMN_PREFIX)
    ]
    for column in dosage_columns:
        numeric_column = imputed[column].astype("float64")
        missing_count = int(numeric_column.isna().sum())
        if missing_count:
            numeric_column = numeric_column.fillna(numeric_column.mean())
            imputed_counts[column] = missing_count
        imputed[column] = numeric_column

    covariate_columns = [
        column for column in DEFAULT_COVARIATE_FIELDS if column in imputed.columns
    ]
    for column in covariate_columns:
        numeric_column = imputed[column].astype("float64")
        missing_count = int(numeric_column.isna().sum())
        if missing_count:
            mode_values = numeric_column.mode(dropna=True)
            fill_value = float(mode_values.iloc[0]) if not mode_values.empty else 0.0
            numeric_column = numeric_column.fillna(fill_value)
            imputed_counts[column] = missing_count
        imputed[column] = numeric_column

    return imputed, imputed_counts


def _format_imputation_summary(imputed_counts: dict[str, int], sample_count: int) -> dict[str, str] | str:
    if not imputed_counts:
        return "none"
    if sample_count <= 0:
        return {column: "n/a" for column in imputed_counts}
    return {
        column: f"{(count / sample_count) * 100:.3g}%"
        for column, count in imputed_counts.items()
    }


def _combine_data_objects(data_objects: Sequence[Any]) -> Any:
    if not data_objects:
        raise ValueError("At least one data object is required.")
    if all(isinstance(data_object, pd.DataFrame) for data_object in data_objects):
        return pd.concat(data_objects, ignore_index=True)
    if all(isinstance(data_object, dict) and "records" in data_object for data_object in data_objects):
        records: list[Any] = []
        for data_object in data_objects:
            records.extend(data_object["records"])
        return {"records": records}
    if all(isinstance(data_object, list) for data_object in data_objects):
        combined: list[Any] = []
        for data_object in data_objects:
            combined.extend(data_object)
        return combined
    if len(data_objects) == 1:
        return data_objects[0]
    raise TypeError("Cannot combine mixed training data container types.")


def _build_fold_artifacts(
    *,
    pair_root: Path,
    calibration_base_path: Path | None,
    scoring_base_path: Path | None,
    num_folds: int,
) -> tuple[PreparedFoldArtifacts, ...]:
    return tuple(
        PreparedFoldArtifacts(
            fold_index=fold_index,
            calibration_pickle=str(
                _numbered_pickle_path(
                    calibration_base_path or pair_root / "calibration.pkl",
                    fold_index,
                )
            ),
            scoring_pickle=str(
                _numbered_pickle_path(
                    scoring_base_path or pair_root / "scoring.pkl",
                    fold_index,
                )
            ),
        )
        for fold_index in range(1, num_folds + 1)
    )


def _numbered_pickle_path(base_path: Path, fold_index: int) -> Path:
    suffix = base_path.suffix or ".pkl"
    stem = base_path.stem if base_path.suffix else base_path.name
    return base_path.with_name(f"{stem}_{fold_index}{suffix}")


def _iter_prepared_artifact_paths(prepared_pair: PreparedDatasetPair) -> tuple[str, ...]:
    return (prepared_pair.oracle_train_pickle,) + tuple(
        path
        for fold_artifact in prepared_pair.fold_artifacts
        for path in (fold_artifact.calibration_pickle, fold_artifact.scoring_pickle)
    )


def _split_scoring_data_into_folds(
    scoring_data: Any,
    *,
    num_folds: int,
    random_seed: int,
) -> tuple[tuple[Any, Any], ...]:
    sample_count = _dataset_length(scoring_data)
    if sample_count < num_folds:
        raise ValueError(
            "At least as many testing samples as evaluator.num_folds are required to build scoring folds. "
            f"Got samples={sample_count} num_folds={num_folds}."
        )

    rng = np.random.default_rng(random_seed)
    shuffled_indices = rng.permutation(sample_count)
    fold_indices = tuple(np.array(indices, dtype=int) for indices in np.array_split(shuffled_indices, num_folds))
    return tuple(
        (
            _select_rows(
                scoring_data,
                np.concatenate(
                    [indices for other_index, indices in enumerate(fold_indices) if other_index != fold_position]
                ).astype(int),
            ),
            _select_rows(scoring_data, fold_indices[fold_position]),
        )
        for fold_position in range(num_folds)
    )


def _stable_prepare_seed(base_seed: int, pair_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{pair_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _dataset_length(data: Any) -> int:
    if isinstance(data, pd.DataFrame):
        return int(data.shape[0])
    if isinstance(data, dict) and "records" in data:
        return len(data["records"])
    return len(list(iter_training_records(data)))


def _select_rows(data: Any, indices: np.ndarray) -> Any:
    if isinstance(data, pd.DataFrame):
        return data.iloc[indices].reset_index(drop=True)
    if isinstance(data, dict) and "records" in data:
        records = list(data["records"])
        return {"records": [records[int(index)] for index in indices]}
    records = list(iter_training_records(data))
    return [records[int(index)] for index in indices]


def _load_priority_function(program_source: str, function_name: str) -> PriorityFunction:
    namespace: dict[str, Any] = {}
    exec(_compose_candidate_module(program_source), namespace)
    priority_function = namespace[function_name]
    if not callable(priority_function):
        raise TypeError(f"{function_name!r} is not callable.")
    return priority_function


def _compose_candidate_module(program_source: str) -> str:
    """Prepend priority-tool imports while keeping ``from __future__`` first.

    Any ``from __future__`` line in the candidate is dropped so the single
    canonical future import in ``_PRIORITY_TOOLS_IMPORTS`` stays at the top of
    the executed module and never triggers a SyntaxError.
    """

    body_lines = [
        line
        for line in program_source.splitlines()
        if not line.lstrip().startswith("from __future__ import")
    ]
    return _PRIORITY_TOOLS_IMPORTS + "\n".join(body_lines) + "\n"


def _validate_priority_signature(priority_function: PriorityFunction) -> None:
    signature = inspect.signature(priority_function)
    parameters = tuple(signature.parameters.values())
    if len(parameters) != len(_EXPECTED_PRIORITY_PARAMETERS):
        raise TypeError(
            "Priority function must accept exactly "
            f"{len(_EXPECTED_PRIORITY_PARAMETERS)} parameters."
        )
    for parameter, expected_name in zip(parameters, _EXPECTED_PRIORITY_PARAMETERS):
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise TypeError("Priority function parameters must be positional.")
        if parameter.name != expected_name:
            raise TypeError(
                "Priority function parameters must be named "
                f"{_EXPECTED_PRIORITY_PARAMETERS!r}."
            )


def _list_variant_names(training_data: Any) -> tuple[str, ...]:
    if isinstance(training_data, pd.DataFrame):
        return tuple(
            str(column_name)
            for column_name in training_data.columns
            if str(column_name).startswith(DOSAGE_COLUMN_PREFIX)
        )

    first_record = next(iter(iter_training_records(training_data)), None)
    if first_record is None:
        return ()
    return tuple(
        field_name
        for field_name in list_record_field_names(first_record)
        if str(field_name).startswith(DOSAGE_COLUMN_PREFIX)
    )


def _extract_labels(data: Any) -> np.ndarray:
    if isinstance(data, pd.DataFrame) and DEFAULT_LABEL_FIELD in data.columns:
        labels = data[DEFAULT_LABEL_FIELD].to_numpy(dtype=float)
    else:
        labels = np.asarray([read_label(record) for record in iter_training_records(data)], dtype=float)
    if labels.size == 0:
        raise ValueError("Cannot score an empty labelled dataset.")
    if np.unique(labels).size < 2:
        raise ValueError("Labels must contain both classes for ROC AUC scoring.")
    return labels


def _extract_covariates(data: Any, *, include_covariates: bool) -> np.ndarray:
    sample_count = _dataset_length(data)
    if not include_covariates:
        return np.zeros((sample_count, 0), dtype=float)
    if isinstance(data, pd.DataFrame):
        missing = [field for field in DEFAULT_COVARIATE_FIELDS if field not in data.columns]
        if missing:
            raise ValueError(f"Missing expected covariate columns: {missing}.")
        return data.loc[:, DEFAULT_COVARIATE_FIELDS].to_numpy(dtype=float)

    covariates = []
    for record in iter_training_records(data):
        row = read_optional_covariates(record)
        if row is None:
            raise ValueError("Covariates were requested but are absent from a record.")
        covariates.append(row)
    return np.vstack(covariates) if covariates else np.zeros((0, len(DEFAULT_COVARIATE_FIELDS)))


def _build_oracle_feature_matrix(
    *,
    training_data: Any,
    subject_data: Any,
    variant_names: Sequence[str],
    priority_function: PriorityFunction,
) -> np.ndarray:
    records = list(iter_training_records(subject_data))
    feature_matrix = np.zeros((len(records), len(variant_names)), dtype=float)
    priority_training_data = _build_priority_training_data_contract(
        training_data,
        variant_names,
    )
    priority_target_variants = _build_priority_target_variants(variant_names)

    for row_index, record in enumerate(records):
        raw_ancestry_coordinate = read_ancestry_coordinate(record)
        priority_ancestry_coordinate = _build_priority_ancestry_coordinate(
            raw_ancestry_coordinate,
        )
        for column_index, variant_name in enumerate(variant_names):
            radius = _call_priority_function(
                priority_function,
                priority_training_data,
                priority_ancestry_coordinate,
                priority_target_variants[column_index],
            )
            effect_size = effect_size_calculator(
                training_data,
                raw_ancestry_coordinate,
                variant_name,
                radius,
            )
            dosage = read_variant_dosage(record, variant_name)
            contribution = float(dosage) * float(effect_size)
            if not math.isfinite(contribution):
                raise ValueError("Oracle contribution must be finite.")
            feature_matrix[row_index, column_index] = contribution
    return feature_matrix


def _build_calibration_oracle_feature_matrix(
    *,
    training_data: Any,
    calibration_data: Any,
    variant_names: Sequence[str],
    candidate_source: str,
    function_name: str,
    calibration_partitions: int,
) -> np.ndarray:
    return _build_partitioned_oracle_feature_matrix(
        training_data=training_data,
        subject_data=calibration_data,
        variant_names=variant_names,
        candidate_source=candidate_source,
        function_name=function_name,
        partitions=calibration_partitions,
    )


def _call_priority_function(
    priority_function: PriorityFunction,
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
) -> float:
    radius = float(priority_function(training_data, ancestry_coordinate, target_variant))
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("Priority function must return a finite non-negative radius.")
    return radius


def _build_priority_training_data_contract(
    training_data: Any,
    variant_names: Sequence[str],
) -> PriorityTrainingData:
    dosage_fields = tuple(str(variant_name) for variant_name in variant_names)
    logical_variant_names = tuple(_logical_variant_name(variant_name) for variant_name in dosage_fields)
    records: list[PriorityTrainingRecord] = []
    covariate_names: tuple[str, ...] = ()

    for record in iter_training_records(training_data):
        ancestry_coordinate = tuple(float(value) for value in read_ancestry_coordinate(record))
        optional_covariates = read_optional_covariates(record)
        covariates = None
        if optional_covariates is not None:
            covariates = {
                covariate_name: float(covariate_value)
                for covariate_name, covariate_value in zip(
                    DEFAULT_COVARIATE_FIELDS,
                    optional_covariates,
                )
            }
            if not covariate_names:
                covariate_names = tuple(covariates.keys())

        variant_dosages = {
            logical_variant_name: float(read_variant_dosage(record, dosage_field))
            for logical_variant_name, dosage_field in zip(logical_variant_names, dosage_fields)
        }
        records.append(
            PriorityTrainingRecord(
                label=float(read_label(record)),
                ancestry_coordinate=ancestry_coordinate,
                variant_dosages=variant_dosages,
                covariates=covariates,
            )
        )

    ancestry_dimension = len(records[0].ancestry_coordinate) if records else 0
    return PriorityTrainingData(
        records=tuple(records),
        variant_names=logical_variant_names,
        variant_dosage_fields=dosage_fields,
        covariate_names=covariate_names,
        sample_count=len(records),
        ancestry_dimension=ancestry_dimension,
        has_additional_covariates=bool(covariate_names),
    )


def _build_priority_ancestry_coordinate(
    ancestry_coordinate: Sequence[float],
) -> PriorityAncestryCoordinate:
    values = tuple(float(value) for value in ancestry_coordinate)
    return PriorityAncestryCoordinate(values=values, dimension=len(values))


def _build_priority_target_variants(
    variant_names: Sequence[str],
) -> tuple[PriorityTargetVariant, ...]:
    return tuple(
        PriorityTargetVariant(
            name=_logical_variant_name(str(variant_name)),
            dosage_field=str(variant_name),
            column_index=column_index,
        )
        for column_index, variant_name in enumerate(variant_names)
    )


def _logical_variant_name(variant_name: str) -> str:
    if variant_name.startswith(DOSAGE_COLUMN_PREFIX):
        return variant_name[len(DOSAGE_COLUMN_PREFIX) :]
    return variant_name


def _build_partitioned_oracle_feature_matrix(
    *,
    training_data: Any,
    subject_data: Any,
    variant_names: Sequence[str],
    candidate_source: str,
    function_name: str,
    partitions: int,
) -> np.ndarray:
    partitions = max(1, partitions)
    if partitions == 1 or _dataset_length(subject_data) <= 1:
        priority_function = _load_priority_function(candidate_source, function_name)
        _validate_priority_signature(priority_function)
        return _build_oracle_feature_matrix(
            training_data=training_data,
            subject_data=subject_data,
            variant_names=variant_names,
            priority_function=priority_function,
        )

    chunks = _split_data_into_chunks(subject_data, partitions)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                _build_partitioned_oracle_feature_matrix_worker,
                candidate_source,
                function_name,
                training_data,
                chunk,
                tuple(variant_names),
            )
            for chunk in chunks
        ]
        return np.vstack([future.result() for future in futures])


def _build_scoring_oracle_feature_matrix(
    *,
    training_data: Any,
    scoring_data: Any,
    variant_names: Sequence[str],
    candidate_source: str,
    function_name: str,
    scoring_partitions: int,
) -> np.ndarray:
    return _build_partitioned_oracle_feature_matrix(
        training_data=training_data,
        subject_data=scoring_data,
        variant_names=variant_names,
        candidate_source=candidate_source,
        function_name=function_name,
        partitions=scoring_partitions,
    )


def _build_partitioned_oracle_feature_matrix_worker(
    candidate_source: str,
    function_name: str,
    training_data: Any,
    subject_chunk: Any,
    variant_names: tuple[str, ...],
) -> np.ndarray:
    priority_function = _load_priority_function(candidate_source, function_name)
    _validate_priority_signature(priority_function)
    return _build_oracle_feature_matrix(
        training_data=training_data,
        subject_data=subject_chunk,
        variant_names=variant_names,
        priority_function=priority_function,
    )


def _split_data_into_chunks(data: Any, requested_chunks: int) -> list[Any]:
    sample_count = _dataset_length(data)
    chunk_count = min(max(1, requested_chunks), sample_count)
    index_chunks = [chunk for chunk in np.array_split(np.arange(sample_count), chunk_count) if chunk.size]
    return [_select_rows(data, chunk) for chunk in index_chunks]


def _current_cpu_index() -> int | None:
    if not hasattr(os, "sched_getcpu"):
        return None
    try:
        return int(os.sched_getcpu())
    except OSError:
        return None


def _allowed_cpu_indices() -> tuple[int, ...] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    except OSError:
        return None


def _log_worker_cpu_binding(
    logger_name: str | None,
    *,
    role: str,
    pair_name: str,
) -> None:
    if logger_name is None:
        return

    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        return

    current_cpu = _current_cpu_index()
    allowed_cpus = _allowed_cpu_indices()
    logger.info(
        "%s initialized pid=%d pair=%s current_cpu=%s allowed_cpus=%s",
        role,
        os.getpid(),
        pair_name,
        current_cpu if current_cpu is not None else "unknown",
        list(allowed_cpus) if allowed_cpus is not None else "unknown",
    )


def _fit_best_calibration_model(
    *,
    oracle_features: np.ndarray,
    covariates: np.ndarray,
    labels: np.ndarray,
    penalties: Sequence[float],
) -> CalibrationModel:
    if oracle_features.shape[0] != labels.shape[0]:
        raise ValueError("Oracle feature rows must match calibration labels.")
    if covariates.shape[0] != labels.shape[0]:
        raise ValueError("Covariate rows must match calibration labels.")
    if not penalties:
        penalties = (1.0,)

    design_matrix = _build_calibration_design_matrix(oracle_features, covariates)
    covariate_count = covariates.shape[1]
    oracle_start = 1 + covariate_count
    best_model: CalibrationModel | None = None

    for penalty in penalties:
        if penalty < 0.0:
            raise ValueError("Calibration penalties must be non-negative.")
        penalty_vector = np.zeros(design_matrix.shape[1], dtype=float)
        penalty_vector[oracle_start:] = float(penalty)
        coefficients = _fit_logistic_ridge_newton(
            design_matrix=design_matrix,
            labels=labels,
            penalty_vector=penalty_vector,
        )
        log_loss = _binary_log_loss(labels, design_matrix @ coefficients)
        model = CalibrationModel(
            intercept=float(coefficients[0]),
            covariate_coefficients=coefficients[1:oracle_start].copy(),
            oracle_coefficients=coefficients[oracle_start:].copy(),
            penalty=float(penalty),
            calibration_log_loss=log_loss,
        )
        if best_model is None or model.calibration_log_loss < best_model.calibration_log_loss:
            best_model = model

    if best_model is None:
        raise ValueError("No calibration model could be fit.")
    return best_model


def _build_calibration_design_matrix(
    oracle_features: np.ndarray,
    covariates: np.ndarray,
) -> np.ndarray:
    intercept = np.ones((oracle_features.shape[0], 1), dtype=float)
    return np.hstack([intercept, covariates, oracle_features])


def _fit_logistic_ridge_newton(
    *,
    design_matrix: np.ndarray,
    labels: np.ndarray,
    penalty_vector: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-7,
) -> np.ndarray:
    coefficients = np.zeros(design_matrix.shape[1], dtype=float)
    penalty_matrix = np.diag(penalty_vector)

    for _ in range(max_iter):
        linear_predictor = design_matrix @ coefficients
        probabilities = _sigmoid(linear_predictor)
        weights = probabilities * (1.0 - probabilities)
        weighted_design = design_matrix * weights[:, None]
        hessian = design_matrix.T @ weighted_design + penalty_matrix
        gradient = design_matrix.T @ (labels - probabilities) - penalty_matrix @ coefficients
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        updated = coefficients + step
        if np.linalg.norm(updated - coefficients) <= tolerance:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def _predict_linear_score(
    model: CalibrationModel,
    *,
    oracle_features: np.ndarray,
    covariates: np.ndarray,
) -> np.ndarray:
    if oracle_features.shape[1] != model.oracle_coefficients.shape[0]:
        raise ValueError("Scoring oracle feature width does not match calibration model.")
    if covariates.shape[1] != model.covariate_coefficients.shape[0]:
        raise ValueError("Scoring covariate width does not match calibration model.")
    return (
        model.intercept
        + covariates @ model.covariate_coefficients
        + oracle_features @ model.oracle_coefficients
    )


def _safe_roc_auc(labels: np.ndarray, risk_scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        raise ValueError("ROC AUC requires both classes in the scoring labels.")
    auc = float(roc_auc_score(labels, risk_scores))
    if not math.isfinite(auc):
        raise ValueError("ROC AUC must be finite.")
    return auc


def _bootstrap_auc_median(
    *,
    labels: np.ndarray,
    risk_scores: np.ndarray,
    bootstrap_iterations: int,
    random_seed: int,
) -> float:
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1.")
    rng = np.random.default_rng(random_seed)
    sample_count = labels.shape[0]
    bootstrap_scores: list[float] = []
    for _ in range(bootstrap_iterations):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        sampled_labels = labels[sample_indices]
        if np.unique(sampled_labels).size < 2:
            continue
        bootstrap_scores.append(
            _safe_roc_auc(sampled_labels, risk_scores[sample_indices])
        )
    if not bootstrap_scores:
        return _safe_roc_auc(labels, risk_scores)
    return float(np.median(np.asarray(bootstrap_scores, dtype=float)))


def _stable_bootstrap_seed(program_source: str, pair_name: str) -> int:
    digest = hashlib.sha256(f"{pair_name}\n{program_source}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def _binary_log_loss(labels: np.ndarray, linear_predictor: np.ndarray) -> float:
    probabilities = np.clip(_sigmoid(linear_predictor), 1e-12, 1.0 - 1e-12)
    losses = -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
    return float(np.mean(losses))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _score_priority_simplicity(program_source: str, function_name: str) -> float:
    module = ast.parse(program_source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return -float(sum(1 for _ in ast.walk(node)))
    return -float(sum(1 for _ in ast.walk(module)))
