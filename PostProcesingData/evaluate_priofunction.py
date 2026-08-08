from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import argparse
import json
import os
import time

import pandas as pd

from funsearch_pipeline.evaluation.procedure2 import _build_calibration_oracle_feature_matrix
from funsearch_pipeline.evaluation.procedure2 import _build_scoring_oracle_feature_matrix
from funsearch_pipeline.evaluation.procedure2 import _combine_data_objects
from funsearch_pipeline.evaluation.procedure2 import _extract_covariates
from funsearch_pipeline.evaluation.procedure2 import _extract_labels
from funsearch_pipeline.evaluation.procedure2 import _fit_best_calibration_model
from funsearch_pipeline.evaluation.procedure2 import _impute_missing_feature_columns
from funsearch_pipeline.evaluation.procedure2 import _list_variant_names
from funsearch_pipeline.evaluation.procedure2 import _load_priority_function
from funsearch_pipeline.evaluation.procedure2 import _predict_linear_score
from funsearch_pipeline.evaluation.procedure2 import _safe_roc_auc
from funsearch_pipeline.evaluation.procedure2 import _validate_priority_signature

from PostProcesingData.prio_func_eval_baselines import evaluate_baseline
from PostProcesingData.prio_func_eval_baselines import normalize_baseline_name


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "Data" / "FunsearchEvaluatorData"
DEFAULT_HELDOUT_PICKLE_PATH = _DEFAULT_DATA_DIR / "no_covariates_heldout.pkl"
DEFAULT_CALIBRATING_PICKLE_PATH = _DEFAULT_DATA_DIR / "no_covariates_test.pkl"
DEFAULT_TRAINING_PICKLE_PATH = _DEFAULT_DATA_DIR / "no_covariates_train.pkl"
DEFAULT_FUNCTION_NAME = "priority"
DEFAULT_CALIBRATION_PENALTIES = (0.1, 1.0, 10.0)


"""
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
PYTHONPATH=$PWD python -m PostProcesingData.evaluate_priofunction PostProcesingData/evaluate_priofunction.oracle_priority_20260717_141059.cycle_0006.best_prio.json
"""

@dataclass(frozen=True)
class BaselineSpec:
    name: str
    enabled: bool
    options: dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    prio_function_path: Path
    heldout_pickle_path: Path
    calibrating_pickle_path: Path
    training_pickle_path: Path
    function_name: str
    calibration_penalties: tuple[float, ...]
    calibration_partitions: int | None
    scoring_partitions: int | None
    baselines: tuple[BaselineSpec, ...]


@dataclass(frozen=True)
class BaselineEvaluation:
    name: str
    auc_roc: float


@dataclass(frozen=True)
class EvaluationReport:
    prio_function_path: Path
    heldout_auc_roc: float
    baseline_evaluations: tuple[BaselineEvaluation, ...]


def _report_progress(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _visible_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = os.sched_getaffinity(0)
        except OSError:
            affinity = None
        if affinity:
            return max(1, len(affinity))

    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task is not None:
        try:
            parsed = int(slurm_cpus_per_task)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    detected = os.cpu_count()
    return max(1, int(detected) if detected is not None else 1)


def _parse_partition_count(raw_value: Any, *, field_name: str) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized == "auto":
            return None
        try:
            parsed = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be a positive integer or 'auto'."
            ) from exc
    else:
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be a positive integer or 'auto'."
            ) from exc

    if parsed <= 0:
        return None
    return parsed


def _resolve_partition_count(requested: int | None) -> int:
    if requested is not None:
        return requested
    return _visible_cpu_count()


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_baselines(raw_baselines: Any) -> tuple[BaselineSpec, ...]:
    if raw_baselines is None:
        return ()
    if not isinstance(raw_baselines, list):
        raise ValueError("baselines must be a JSON array.")

    baselines: list[BaselineSpec] = []
    for raw_baseline in raw_baselines:
        if isinstance(raw_baseline, str):
            baselines.append(
                BaselineSpec(
                    name=raw_baseline,
                    enabled=True,
                    options={},
                )
            )
            continue
        if not isinstance(raw_baseline, dict):
            raise ValueError("Each baseline entry must be a string or JSON object.")
        if "name" not in raw_baseline:
            raise ValueError("Each baseline object must include a 'name'.")
        options = {
            key: value
            for key, value in raw_baseline.items()
            if key not in {"name", "enabled"}
        }
        baselines.append(
            BaselineSpec(
                name=str(raw_baseline["name"]),
                enabled=bool(raw_baseline.get("enabled", True)),
                options=options,
            )
        )
    return tuple(baselines)


def load_evaluation_config(config_path: str | Path) -> EvaluationConfig:
    config_path = Path(config_path).expanduser().resolve()
    raw_config = json.loads(config_path.read_text())
    if not isinstance(raw_config, dict):
        raise ValueError("The evaluation config must be a JSON object.")

    base_dir = config_path.parent
    if "prio_function_path" not in raw_config:
        raise ValueError("Missing required config field 'prio_function_path'.")

    calibration_penalties = raw_config.get(
        "calibration_penalties",
        list(DEFAULT_CALIBRATION_PENALTIES),
    )
    if not isinstance(calibration_penalties, list) or not calibration_penalties:
        raise ValueError("calibration_penalties must be a non-empty JSON array.")

    baselines = tuple(
        baseline
        for baseline in _parse_baselines(raw_config.get("baselines", []))
        if baseline.enabled
    )
    return EvaluationConfig(
        prio_function_path=_resolve_path(base_dir, str(raw_config["prio_function_path"])),
        heldout_pickle_path=_resolve_path(
            base_dir,
            str(raw_config.get("heldout_pickle_path", DEFAULT_HELDOUT_PICKLE_PATH)),
        ),
        calibrating_pickle_path=_resolve_path(
            base_dir,
            str(raw_config.get("calibrating_pickle_path", DEFAULT_CALIBRATING_PICKLE_PATH)),
        ),
        training_pickle_path=_resolve_path(
            base_dir,
            str(raw_config.get("training_pickle_path", DEFAULT_TRAINING_PICKLE_PATH)),
        ),
        function_name=str(raw_config.get("function_name", DEFAULT_FUNCTION_NAME)),
        calibration_penalties=tuple(float(value) for value in calibration_penalties),
        calibration_partitions=_parse_partition_count(
            raw_config.get("calibration_partitions", "auto"),
            field_name="calibration_partitions",
        ),
        scoring_partitions=_parse_partition_count(
            raw_config.get("scoring_partitions", "auto"),
            field_name="scoring_partitions",
        ),
        baselines=baselines,
    )


def _read_and_impute_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame pickle at {path}.")
    imputed_frame, _ = _impute_missing_feature_columns(frame)
    if not isinstance(imputed_frame, pd.DataFrame):
        raise TypeError(f"Expected imputed data at {path} to remain a DataFrame.")
    return imputed_frame


def evaluate_priority_function(
    config: EvaluationConfig,
    *,
    progress_reporter: Callable[[str], None] | None = None,
) -> EvaluationReport:
    report = progress_reporter or (lambda message: None)
    calibration_partitions = _resolve_partition_count(config.calibration_partitions)
    scoring_partitions = _resolve_partition_count(config.scoring_partitions)
    visible_cpu_count = _visible_cpu_count()
    report(f"Loading priority function from {config.prio_function_path}")
    report(
        "Runtime CPU detection: "
        f"visible_cpus={visible_cpu_count} calibration_workers={calibration_partitions} "
        f"scoring_workers={scoring_partitions}"
    )
    program_source = config.prio_function_path.read_text()
    priority_function = _load_priority_function(program_source, config.function_name)
    _validate_priority_signature(priority_function)

    report(
        "Loading datasets: "
        f"training={config.training_pickle_path} calibration={config.calibrating_pickle_path} "
        f"heldout={config.heldout_pickle_path}"
    )
    training_data = _read_and_impute_frame(config.training_pickle_path)
    calibration_data = _read_and_impute_frame(config.calibrating_pickle_path)
    heldout_data = _read_and_impute_frame(config.heldout_pickle_path)
    report(
        "Loaded datasets with rows: "
        f"training={len(training_data)} calibration={len(calibration_data)} heldout={len(heldout_data)}"
    )
    variant_names = _list_variant_names(training_data)
    if not variant_names:
        raise ValueError("No dosage columns were found in the training pickle.")
    report(f"Found {len(variant_names)} dosage variants in training data")

    calibration_labels = _extract_labels(calibration_data)
    report(
        "Building calibration oracle features; this is often the first long-running step"
    )
    calibration_oracle_features = _build_calibration_oracle_feature_matrix(
        training_data=training_data,
        calibration_data=calibration_data,
        variant_names=variant_names,
        candidate_source=program_source,
        function_name=config.function_name,
        calibration_partitions=calibration_partitions,
    )
    report(
        "Finished calibration oracle features with shape "
        f"{calibration_oracle_features.shape}"
    )
    calibration_covariates = _extract_covariates(
        calibration_data,
        include_covariates=False,
    )
    report(
        "Fitting calibration model across penalties "
        f"{list(config.calibration_penalties)}"
    )
    calibration_model = _fit_best_calibration_model(
        oracle_features=calibration_oracle_features,
        covariates=calibration_covariates,
        labels=calibration_labels,
        penalties=config.calibration_penalties,
    )
    report(
        "Finished calibration model fit with selected penalty "
        f"{calibration_model.penalty}"
    )

    oracle_training_for_heldout = _combine_data_objects([training_data, calibration_data])
    heldout_labels = _extract_labels(heldout_data)
    report(
        "Building heldout oracle features using training + calibration data; "
        "this is often the slowest step"
    )
    heldout_oracle_features = _build_scoring_oracle_feature_matrix(
        training_data=oracle_training_for_heldout,
        scoring_data=heldout_data,
        variant_names=variant_names,
        candidate_source=program_source,
        function_name=config.function_name,
        scoring_partitions=scoring_partitions,
    )
    report(
        "Finished heldout oracle features with shape "
        f"{heldout_oracle_features.shape}"
    )
    heldout_covariates = _extract_covariates(
        heldout_data,
        include_covariates=False,
    )
    heldout_risk_scores = _predict_linear_score(
        calibration_model,
        oracle_features=heldout_oracle_features,
        covariates=heldout_covariates,
    )
    heldout_auc_roc = _safe_roc_auc(heldout_labels, heldout_risk_scores)
    report(f"Priority-function heldout ROC AUC={heldout_auc_roc:.6f}")

    baseline_evaluations: list[BaselineEvaluation] = []
    for baseline in config.baselines:
        baseline_name = normalize_baseline_name(baseline.name)
        report(f"Running baseline {baseline_name}")
        baseline_auc = evaluate_baseline(
            baseline.name,
            training_data=training_data,
            calibration_data=calibration_data,
            heldout_data=heldout_data,
            options=baseline.options,
        )
        report(f"Finished baseline {baseline_name} with heldout ROC AUC={baseline_auc:.6f}")
        baseline_evaluations.append(
            BaselineEvaluation(
                name=baseline_name,
                auc_roc=baseline_auc,
            )
        )

    return EvaluationReport(
        prio_function_path=config.prio_function_path,
        heldout_auc_roc=heldout_auc_roc,
        baseline_evaluations=tuple(baseline_evaluations),
    )


def evaluate_from_config_path(
    config_path: str | Path,
    *,
    progress_reporter: Callable[[str], None] | None = None,
) -> EvaluationReport:
    return evaluate_priority_function(
        load_evaluation_config(config_path),
        progress_reporter=progress_reporter,
    )


def format_report(report: EvaluationReport) -> str:
    lines = [
        f"prio_function_path={report.prio_function_path}",
        f"heldout_auc_roc={report.heldout_auc_roc:.6f}",
    ]
    lines.extend(
        f"baseline_auc_roc[{baseline.name}]={baseline.auc_roc:.6f}"
        for baseline in report.baseline_evaluations
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one priority-function file on heldout subjects using the "
            "Procedure 2 calibration/scoring flow and optional baseline models."
        )
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to a JSON config file for heldout priority-function evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_from_config_path(
        args.config_path,
        progress_reporter=_report_progress,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())