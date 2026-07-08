from __future__ import annotations

import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from funsearch_pipeline.config import DatasetPairConfig
from funsearch_pipeline.config import EvaluatorSettings
from funsearch_pipeline.config import ProgramDatabaseSettings
from funsearch_pipeline.evaluation import build_evaluator
from funsearch_pipeline.logging_utils import configure_file_logger
from funsearch_pipeline.program_database.database import CycleProgramsDatabase

"""
PYTHONDONTWRITEBYTECODE=1 pytest -s -q -p no:cacheprovider test_evaluator2.py
"""

NO_COVARIATES_TRAINING = (
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/train_AA.pkl",
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/train_LA.pkl",
)
NO_COVARIATES_TESTING = (
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/test_AA.pkl",
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/test_LA.pkl",
)
WITH_COVARIATES_TRAINING = (
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/train_AA_add_covs.pkl",
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/train_LA_add_covs.pkl",
)
WITH_COVARIATES_TESTING = (
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/test_AA_add_covs.pkl",
    "/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/"
    "MultiStagePythonCodeOutput/20260703_193925/stage4/test_LA_add_covs.pkl",
)


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


PRIORITY_FUNCTION_SOURCE = (
    "def priority(\n"
    "    training_data: PriorityTrainingData,\n"
    "    ancestry_coordinate: PriorityAncestryCoordinate,\n"
    "    target_variant: PriorityTargetVariant,\n"
    ") -> float:\n"
    "    radii_and_effects = effect_size_by_cumulative_radius(\n"
    "        training_data,\n"
    "        ancestry_coordinate,\n"
    "        target_variant,\n"
    "        6,\n"
    "    )\n"
    "    if not radii_and_effects:\n"
    "        return 0.0\n"
    "    if len(radii_and_effects) == 1:\n"
    "        return radii_and_effects[0][0]\n"
    "    drop_index = max(\n"
    "        range(1, len(radii_and_effects)),\n"
    "        key=lambda index: abs(radii_and_effects[index - 1][1] - radii_and_effects[index][1]),\n"
    "    )\n"
    "    return radii_and_effects[drop_index][0]\n"
)


def test_manual_procedure2_evaluation() -> None:
    run_root = Path(__file__).resolve().parent / "test_evaluator2"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    seed_path = _write_text(
        run_root / "seed_priority.py",
        PRIORITY_FUNCTION_SOURCE,
    )
    log_path = run_root / "test_evaluator2.log"
    logger = configure_file_logger(
        log_path,
        "INFO",
        logger_name="anirban.manual.test_evaluator2",
    )
    experiment_dir = run_root / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    evaluator = build_evaluator(
        EvaluatorSettings(
            backend="procedure2",
            metric="roc_auc",
            oracle_train_fraction=0.8,
            preprocessed_dirname="preprocessed",
            calibration_penalties=(0.1, 1.0, 10.0),
            calibration_partitions=10,
            scoring_partitions=10,
            bootstrap_iterations=50,
            dataset_pairs=(
                DatasetPairConfig(
                    name="no_covariates",
                    has_additional_covariates=False,
                    training_pickles=tuple(Path(path) for path in NO_COVARIATES_TRAINING),
                    testing_pickles=tuple(Path(path) for path in NO_COVARIATES_TESTING),
                ),
                DatasetPairConfig(
                    name="with_covariates",
                    has_additional_covariates=True,
                    training_pickles=tuple(Path(path) for path in WITH_COVARIATES_TRAINING),
                    testing_pickles=tuple(Path(path) for path in WITH_COVARIATES_TESTING),
                ),
            ),
        ),
        function_name="priority",
        logger=logger,
    )

    evaluator.prepare(experiment_dir)

    database = CycleProgramsDatabase.from_seed_program_text(
        settings=ProgramDatabaseSettings(
            functions_per_prompt=2,
            num_islands=2,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=100,
        ),
        seed_program_text=seed_path.read_text(),
        function_to_evolve="priority",
    )
    seed_candidate = database.build_seed_candidate()
    evaluation = evaluator.evaluate_candidate(seed_candidate)
    assert evaluation is not None
    database.register_seed(dict(evaluation.scores_per_test()))

    database.save_snapshot(
        experiment_dir / "program_db" / "manual_bootstrap.pkl",
        experiment_dir / "program_db" / "manual_bootstrap_summary.json",
        extra_metadata={"stage": "manual_bootstrap"},
    )
    logger.info("Manual evaluation scores: %s", evaluation.scores_per_test())
    logger.info("Manual evaluation log path: %s", log_path)
    print(f"manual evaluation log: {log_path}")
    print(f"manual evaluation scores: {evaluation.scores_per_test()}")