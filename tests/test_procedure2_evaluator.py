from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from funsearch_pipeline.config import DatasetPairConfig
from funsearch_pipeline.config import EvaluatorSettings
from funsearch_pipeline.config import ProgramDatabaseSettings
from funsearch_pipeline.evaluation import build_evaluator
from funsearch_pipeline.evaluation.procedure2 import Procedure2PriorityEvaluator
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
    assert list(scores) == ["no_covariates", "simplicity", "mean"]
    assert scores["no_covariates"] > 0.9
    assert scores["mean"] == scores["no_covariates"]
    assert scores["simplicity"] < 0.0
    assert result.metadata["procedure2_pairs"]["no_covariates"]["num_variants"] == 2


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
    assert log_text.count("Prepared Procedure 2 dataset pair no_covariates") == 1
    assert str(train_a) in log_text
    assert str(train_b) in log_text
    assert str(test_path) in log_text
    assert "oracle_train_samples=64" in log_text
    assert "calibration_samples=16" in log_text
    assert "scoring_samples=30" in log_text