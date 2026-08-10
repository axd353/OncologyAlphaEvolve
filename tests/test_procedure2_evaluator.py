from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from funsearch_pipeline.config import DatasetPairConfig
from funsearch_pipeline.config import EvaluatorSettings
from funsearch_pipeline.config import ProgramDatabaseSettings
from funsearch_pipeline.evaluation import build_evaluator
from funsearch_pipeline.evaluation.procedure2 import Procedure2PriorityEvaluator
from funsearch_pipeline.evaluation.procedure2 import _impute_missing_feature_columns
from funsearch_pipeline.logging_utils import configure_file_logger
from funsearch_pipeline.program_database.database import CycleProgramsDatabase


def _make_synthetic_oracle_frame(sample_count: int, *, offset: int = 0) -> pd.DataFrame:
    labels = ((np.arange(sample_count) + offset) % 2).astype(float)
    data = {
        "phenotype": labels,
        "dosage__risk": labels * 2.0,
        "dosage__noise": 1.0 - labels,
    }
    for coordinate_index in range(1, 17):
        data[f"PC{coordinate_index}"] = np.zeros(sample_count, dtype=float)
    return pd.DataFrame(data)


def _write_pickle(path: Path, data_frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(path)
    return path


def test_procedure2_evaluator_scores_seed_priority_function(tmp_path: Path) -> None:
    train_a = _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(40))
    train_b = _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(40, offset=40),
    )
    test_path = _write_pickle(
        tmp_path / "test.pkl",
        _make_synthetic_oracle_frame(30, offset=80),
    )
    seed_source = (
        "from __future__ import annotations\n"
        "from funsearch_pipeline.priority_tools.contracts import PriorityAncestryCoordinate\n"
        "from funsearch_pipeline.priority_tools.contracts import PriorityTargetVariant\n"
        "from funsearch_pipeline.priority_tools.contracts import PriorityTrainingData\n\n"
        "def priority(\n"
        "    training_data: PriorityTrainingData,\n"
        "    ancestry_coordinate: PriorityAncestryCoordinate,\n"
        "    target_variant: PriorityTargetVariant,\n"
        ") -> float:\n"
        "    if hasattr(training_data, 'columns'):\n"
        "        raise AssertionError('raw training dataframe leaked into priority')\n"
        "    if training_data.sample_count < 1:\n"
        "        raise AssertionError('missing oracle-train samples')\n"
        "    if ancestry_coordinate.dimension != len(ancestry_coordinate.values):\n"
        "        raise AssertionError('ancestry contract malformed')\n"
        "    if target_variant.name not in training_data.variant_names:\n"
        "        raise AssertionError('variant contract malformed')\n"
        "    return 10.0\n"
    )
    database = CycleProgramsDatabase.from_seed_program_text(
        settings=ProgramDatabaseSettings(
            functions_per_prompt=2,
            num_islands=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=100,
        ),
        seed_program_text=seed_source,
        function_to_evolve="priority",
    )
    candidate = database.build_seed_candidate()
    evaluator = Procedure2PriorityEvaluator(
        settings=EvaluatorSettings(
            backend="procedure2",
            metric="roc_auc",
            oracle_train_fraction=0.8,
            preprocessed_dirname="preprocessed",
            calibration_penalties=(0.1, 1.0),
            calibration_partitions=2,
            scoring_partitions=2,
            bootstrap_iterations=10,
            dataset_pairs=(
                DatasetPairConfig(
                    name="no_covariates",
                    has_additional_covariates=False,
                    training_pickles=(train_a, train_b),
                    testing_pickles=(test_path,),
                ),
            ),
        ),
        function_name="priority",
    )

    assert isinstance(
        build_evaluator(
            evaluator._settings,
            function_name="priority",
        ),
        Procedure2PriorityEvaluator,
    )
    evaluator.prepare(tmp_path)
    result = evaluator.evaluate_candidate(candidate)

    assert result is not None
    assert (tmp_path / "preprocessed" / "no_covariates" / "oracle_train.pkl").exists()
    scores = result.scores_per_test()
    assert list(scores) == ["no_covariates_fold_1", "no_covariates_fold_2", "simplicity", "mean"]
    assert scores["no_covariates_fold_1"] > 0.9
    assert scores["no_covariates_fold_2"] > 0.9
    assert scores["mean"] == np.mean([
        scores["no_covariates_fold_1"],
        scores["no_covariates_fold_2"],
    ])
    assert scores["simplicity"] < 0.0
    assert result.metadata["procedure2_pairs"]["no_covariates"]["num_variants"] == 2
    assert result.metadata["procedure2_pairs"]["no_covariates"]["num_folds"] == 2


def test_procedure2_scores_two_pairs_in_parallel(tmp_path: Path) -> None:
    train = _write_pickle(tmp_path / "train.pkl", _make_synthetic_oracle_frame(60))
    test = _write_pickle(tmp_path / "test.pkl", _make_synthetic_oracle_frame(40, offset=60))
    log_path = tmp_path / "parallel_pairs.log"
    logger = configure_file_logger(
        log_path,
        "INFO",
        logger_name=f"test.procedure2.parallel_pairs.{tmp_path.name}",
    )
    seed_source = (
        "def priority(training_data, ancestry_coordinate, target_variant):\n"
        "    return 10.0\n"
    )
    database = CycleProgramsDatabase.from_seed_program_text(
        settings=ProgramDatabaseSettings(
            functions_per_prompt=2,
            num_islands=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=100,
        ),
        seed_program_text=seed_source,
        function_to_evolve="priority",
    )
    candidate = database.build_seed_candidate()
    evaluator = Procedure2PriorityEvaluator(
        settings=EvaluatorSettings(
            backend="procedure2",
            metric="roc_auc",
            oracle_train_fraction=0.8,
            preprocessed_dirname="preprocessed",
            calibration_penalties=(1.0,),
            calibration_partitions=1,
            scoring_partitions=1,
            bootstrap_iterations=10,
            dataset_pairs=(
                DatasetPairConfig(
                    name="pair_one",
                    has_additional_covariates=False,
                    training_pickles=(train,),
                    testing_pickles=(test,),
                ),
                DatasetPairConfig(
                    name="pair_two",
                    has_additional_covariates=False,
                    training_pickles=(train,),
                    testing_pickles=(test,),
                ),
            ),
        ),
        function_name="priority",
        logger=logger,
    )

    evaluator.prepare(tmp_path)
    result = evaluator.evaluate_candidate(candidate)

    assert result is not None
    scores = result.scores_per_test()
    assert list(scores) == [
        "pair_one_fold_1",
        "pair_one_fold_2",
        "pair_two_fold_1",
        "pair_two_fold_2",
        "simplicity",
        "mean",
    ]
    assert set(result.metadata["procedure2_pairs"]) == {"pair_one", "pair_two"}
    log_text = log_path.read_text()
    assert "procedure2_pair_worker initialized" in log_text
    assert "pair=pair_one" in log_text or "pair=pair_two" in log_text


def test_procedure2_prepare_logs_first_materialization(tmp_path: Path) -> None:
    train_a = _write_pickle(tmp_path / "train_a.pkl", _make_synthetic_oracle_frame(40))
    train_b = _write_pickle(
        tmp_path / "train_b.pkl",
        _make_synthetic_oracle_frame(40, offset=40),
    )
    test_path = _write_pickle(
        tmp_path / "test.pkl",
        _make_synthetic_oracle_frame(30, offset=80),
    )
    log_path = tmp_path / "prepare.log"
    logger = configure_file_logger(
        log_path,
        "INFO",
        logger_name=f"test.procedure2.prepare.{tmp_path.name}",
    )
    settings = EvaluatorSettings(
        backend="procedure2",
        metric="roc_auc",
        oracle_train_fraction=0.8,
        preprocessed_dirname="preprocessed",
        calibration_penalties=(0.1, 1.0),
        calibration_partitions=1,
        scoring_partitions=1,
        bootstrap_iterations=10,
        dataset_pairs=(
            DatasetPairConfig(
                name="no_covariates",
                has_additional_covariates=False,
                training_pickles=(train_a, train_b),
                testing_pickles=(test_path,),
            ),
        ),
    )

    evaluator = build_evaluator(
        settings,
        function_name="priority",
        logger=logger,
    )

    assert isinstance(evaluator, Procedure2PriorityEvaluator)

    experiment_dir = tmp_path / "experiment"
    evaluator.prepare(experiment_dir)
    evaluator.prepare(experiment_dir)

    log_text = log_path.read_text()
    assert log_text.count("Prepared Procedure 2 dataset pair no_covariates from") == 1
    assert str(train_a) in log_text
    assert str(train_b) in log_text
    assert str(test_path) in log_text
    assert "oracle_train_samples=80" in log_text
    assert "num_folds=2" in log_text
    assert "scoring_fold_sizes=[15, 15]" in log_text
    assert "calibration_samples=15" in log_text
    assert "scoring_samples=15" in log_text


def test_procedure2_imputes_missing_dosages_before_scoring(tmp_path: Path) -> None:
    train_frame = _make_synthetic_oracle_frame(40)
    train_frame.loc[0, "dosage__risk"] = np.nan
    test_frame = _make_synthetic_oracle_frame(30, offset=80)
    test_frame.loc[0, "dosage__risk"] = np.nan
    train_path = _write_pickle(tmp_path / "train.pkl", train_frame)
    test_path = _write_pickle(tmp_path / "test.pkl", test_frame)

    seed_source = (
        "from __future__ import annotations\n"
        "from funsearch_pipeline.priority_tools.contracts import PriorityAncestryCoordinate\n"
        "from funsearch_pipeline.priority_tools.contracts import PriorityTargetVariant\n"
        "from funsearch_pipeline.priority_tools.contracts import PriorityTrainingData\n\n"
        "def priority(\n"
        "    training_data: PriorityTrainingData,\n"
        "    ancestry_coordinate: PriorityAncestryCoordinate,\n"
        "    target_variant: PriorityTargetVariant,\n"
        ") -> float:\n"
        "    return 10.0\n"
    )
    database = CycleProgramsDatabase.from_seed_program_text(
        settings=ProgramDatabaseSettings(
            functions_per_prompt=2,
            num_islands=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=100,
        ),
        seed_program_text=seed_source,
        function_to_evolve="priority",
    )
    candidate = database.build_seed_candidate()
    evaluator = Procedure2PriorityEvaluator(
        settings=EvaluatorSettings(
            backend="procedure2",
            metric="roc_auc",
            oracle_train_fraction=0.8,
            preprocessed_dirname="preprocessed",
            calibration_penalties=(0.1, 1.0),
            calibration_partitions=1,
            scoring_partitions=1,
            bootstrap_iterations=10,
            dataset_pairs=(
                DatasetPairConfig(
                    name="no_covariates",
                    has_additional_covariates=False,
                    training_pickles=(train_path,),
                    testing_pickles=(test_path,),
                ),
            ),
        ),
        function_name="priority",
    )

    evaluator.prepare(tmp_path)

    prepared_scoring = pd.read_pickle(
        tmp_path / "preprocessed" / "no_covariates" / "scoring_1.pkl"
    )
    assert not prepared_scoring["dosage__risk"].isna().any()

    result = evaluator.evaluate_candidate(candidate)
    assert result is not None


def test_impute_missing_feature_columns_handles_nullable_covariates() -> None:
    frame = pd.DataFrame(
        {
            "phenotype": [0, 1, 0, 1],
            "dosage__risk": [0.0, np.nan, 2.0, 2.0],
            "bmi_cat": pd.array([1, 1, pd.NA, 2], dtype="Int64"),
            "current_smoking": pd.array([pd.NA, 0, 0, 1], dtype="Int64"),
        }
    )

    imputed, counts = _impute_missing_feature_columns(frame)

    assert counts == {"dosage__risk": 1, "bmi_cat": 1, "current_smoking": 1}
    assert not imputed[["dosage__risk", "bmi_cat", "current_smoking"]].isna().any().any()
    # dosage uses the column mean of the observed values (0, 2, 2).
    assert np.isclose(imputed.loc[1, "dosage__risk"], 4.0 / 3.0)
    # categorical covariates use the mode of the observed codes.
    assert imputed.loc[2, "bmi_cat"] == 1.0
    assert imputed.loc[0, "current_smoking"] == 0.0