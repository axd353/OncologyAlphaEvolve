from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import argparse
import json
import os
import re
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
from GenomicsHelpers.ancestry_distance_cache import ensure_distance_cache

from PostProcesingData.prio_func_eval_baselines import evaluate_baseline
from PostProcesingData.prio_func_eval_baselines import BaselineResult
from PostProcesingData.prio_func_eval_baselines import normalize_baseline_name


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "Data" / "FunsearchEvaluatorData"
DEFAULT_HELDOUT_PICKLE_PATH = _DEFAULT_DATA_DIR / "no_covariates_heldout.pkl"
DEFAULT_CALIBRATING_PICKLE_PATH = _DEFAULT_DATA_DIR / "no_covariates_test.pkl"
DEFAULT_TRAINING_PICKLE_PATH = _DEFAULT_DATA_DIR / "no_covariates_train.pkl"
DEFAULT_OUTPUT_ROW_TRACKING_PATH = _DEFAULT_DATA_DIR / "output_row_tracking.pkl"
DEFAULT_FUNCTION_NAME = "priority"
DEFAULT_CALIBRATION_PENALTIES = (0.1, 1.0, 10.0)
DEFAULT_SUPPORTED_ANCESTRY_GROUPS = ("AA", "JA", "LA")


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
    report_file_name: str
    heldout_pickle_paths: tuple[Path, ...]
    calibrating_pickle_paths: tuple[Path, ...]
    training_pickle_paths: tuple[Path, ...]
    output_row_tracking_path: Path
    supported_ancestry_groups: tuple[str, ...]
    should_overwrite: bool
    function_name: str
    calibration_penalties: tuple[float, ...]
    calibration_partitions: int | None
    scoring_partitions: int | None
    distance_cache_enabled: bool
    distance_cache_dir: Path | None
    baselines: tuple[BaselineSpec, ...]


@dataclass(frozen=True)
class BaselineEvaluation:
    name: str
    auc_roc: float
    heldout_ancestry_evaluations: tuple[HeldoutAncestryEvaluation, ...]


@dataclass(frozen=True)
class HeldoutAncestryEvaluation:
    ancestry_group: str
    subject_count: int
    auc_roc: float


@dataclass(frozen=True)
class EvaluationReport:
    prio_function_path: Path
    heldout_auc_roc: float
    heldout_ancestry_evaluations: tuple[HeldoutAncestryEvaluation, ...]
    baseline_evaluations: tuple[BaselineEvaluation, ...]


def _config_to_payload(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "prio_function_path": str(config.prio_function_path),
        "report_file_name": config.report_file_name,
        "training_pickle_path": [str(path) for path in config.training_pickle_paths],
        "calibrating_pickle_path": [str(path) for path in config.calibrating_pickle_paths],
        "heldout_pickle_path": [str(path) for path in config.heldout_pickle_paths],
        "output_row_tracking_path": str(config.output_row_tracking_path),
        "supported_ancestry_groups": list(config.supported_ancestry_groups),
        "should_overwrite": config.should_overwrite,
        "function_name": config.function_name,
        "calibration_penalties": list(config.calibration_penalties),
        "calibration_partitions": config.calibration_partitions,
        "scoring_partitions": config.scoring_partitions,
        "distance_cache_enabled": config.distance_cache_enabled,
        "distance_cache_dir": str(config.distance_cache_dir) if config.distance_cache_dir is not None else None,
        "baselines": [
            {
                "name": baseline.name,
                "enabled": baseline.enabled,
                **baseline.options,
            }
            for baseline in config.baselines
        ],
    }


def _report_to_payload(report: EvaluationReport) -> dict[str, Any]:
    return {
        "prio_function_path": str(report.prio_function_path),
        "heldout_auc_roc": report.heldout_auc_roc,
        "heldout_ancestry_evaluations": [
            {
                "ancestry_group": evaluation.ancestry_group,
                "subject_count": evaluation.subject_count,
                "auc_roc": evaluation.auc_roc,
            }
            for evaluation in report.heldout_ancestry_evaluations
        ],
        "baseline_evaluations": [
            {
                "name": baseline.name,
                "auc_roc": baseline.auc_roc,
                "heldout_ancestry_evaluations": [
                    {
                        "ancestry_group": evaluation.ancestry_group,
                        "subject_count": evaluation.subject_count,
                        "auc_roc": evaluation.auc_roc,
                    }
                    for evaluation in baseline.heldout_ancestry_evaluations
                ],
            }
            for baseline in report.baseline_evaluations
        ],
        "summary_text": format_report(report),
    }


def _default_report_file_name_for_priority_function(prio_function_path: Path) -> str:
    return prio_function_path.with_suffix(".evaluation_report.json").name


def report_output_path_for_priority_function(
    prio_function_path: Path,
    report_file_name: str | None = None,
) -> Path:
    resolved_report_file_name = report_file_name or _default_report_file_name_for_priority_function(
        prio_function_path
    )
    return prio_function_path.parent / resolved_report_file_name


def _default_distance_cache_dir_for_priority_function(prio_function_path: Path) -> Path:
    return prio_function_path.parent / "distance_cache"


def write_evaluation_report_file(
    *,
    config_path: str | Path,
    config: EvaluationConfig,
    report: EvaluationReport,
) -> Path:
    output_path = report_output_path_for_priority_function(
        report.prio_function_path,
        config.report_file_name,
    )
    payload = {
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config": _config_to_payload(config),
        "results": _report_to_payload(report),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


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


def _resolve_path_list(
    base_dir: Path,
    raw_value: Any,
    *,
    field_name: str,
    default_path: Path,
) -> tuple[Path, ...]:
    if raw_value is None:
        raw_items: list[Any] = [default_path]
    elif isinstance(raw_value, list):
        raw_items = raw_value
    else:
        raw_items = [raw_value]

    if not raw_items:
        raise ValueError(f"{field_name} must contain at least one path.")

    resolved_paths: list[Path] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, (str, Path)):
            raise ValueError(f"{field_name} entries must be paths encoded as strings.")
        resolved_paths.append(_resolve_path(base_dir, str(raw_item)))
    return tuple(resolved_paths)


def _parse_supported_ancestry_groups(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return DEFAULT_SUPPORTED_ANCESTRY_GROUPS
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("supported_ancestry_groups must be a non-empty JSON array.")

    ancestry_groups: list[str] = []
    for raw_group in raw_value:
        if not isinstance(raw_group, str):
            raise ValueError("supported_ancestry_groups entries must be strings.")
        normalized_group = raw_group.strip().upper()
        if not normalized_group:
            raise ValueError("supported_ancestry_groups entries must not be empty.")
        if normalized_group not in ancestry_groups:
            ancestry_groups.append(normalized_group)
    return tuple(ancestry_groups)


def _parse_report_file_name(raw_value: Any, *, prio_function_path: Path) -> str:
    if raw_value is None:
        return _default_report_file_name_for_priority_function(prio_function_path)
    if not isinstance(raw_value, str):
        raise ValueError("report_file_name must be a string.")
    report_file_name = raw_value.strip()
    if not report_file_name:
        raise ValueError("report_file_name must not be empty.")
    report_path = Path(report_file_name)
    if report_path.is_absolute() or report_path.name != report_file_name or report_file_name in {".", ".."}:
        raise ValueError("report_file_name must be a file name, not a path.")
    return report_file_name


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
    prio_function_path = _resolve_path(base_dir, str(raw_config["prio_function_path"]))
    return EvaluationConfig(
        prio_function_path=prio_function_path,
        report_file_name=_parse_report_file_name(
            raw_config.get("report_file_name"),
            prio_function_path=prio_function_path,
        ),
        heldout_pickle_paths=_resolve_path_list(
            base_dir,
            raw_config.get("heldout_pickle_path", DEFAULT_HELDOUT_PICKLE_PATH),
            field_name="heldout_pickle_path",
            default_path=DEFAULT_HELDOUT_PICKLE_PATH,
        ),
        calibrating_pickle_paths=_resolve_path_list(
            base_dir,
            raw_config.get("calibrating_pickle_path", DEFAULT_CALIBRATING_PICKLE_PATH),
            field_name="calibrating_pickle_path",
            default_path=DEFAULT_CALIBRATING_PICKLE_PATH,
        ),
        training_pickle_paths=_resolve_path_list(
            base_dir,
            raw_config.get("training_pickle_path", DEFAULT_TRAINING_PICKLE_PATH),
            field_name="training_pickle_path",
            default_path=DEFAULT_TRAINING_PICKLE_PATH,
        ),
        output_row_tracking_path=_resolve_path(
            base_dir,
            str(raw_config.get("output_row_tracking_path", DEFAULT_OUTPUT_ROW_TRACKING_PATH)),
        ),
        supported_ancestry_groups=_parse_supported_ancestry_groups(
            raw_config.get("supported_ancestry_groups")
        ),
        should_overwrite=bool(raw_config.get("should_overwrite", True)),
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
        distance_cache_enabled=bool(raw_config.get("distance_cache_enabled", True)),
        distance_cache_dir=(
            _resolve_path(base_dir, str(raw_config["distance_cache_dir"]))
            if raw_config.get("distance_cache_dir")
            else None
        ),
        baselines=baselines,
    )


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame pickle at {path}.")
    return frame


def _read_and_impute_dataset(paths: tuple[Path, ...]) -> tuple[pd.DataFrame, tuple[int, ...]]:
    frames = tuple(_read_frame(path) for path in paths)
    combined_frame = _combine_data_objects(frames)
    if not isinstance(combined_frame, pd.DataFrame):
        raise TypeError("Expected combined input data to be a pandas DataFrame.")
    imputed_frame, _ = _impute_missing_feature_columns(combined_frame)
    if not isinstance(imputed_frame, pd.DataFrame):
        raise TypeError("Expected imputed combined input data to remain a DataFrame.")
    return imputed_frame, tuple(len(frame) for frame in frames)


def _format_paths_for_logging(paths: tuple[Path, ...]) -> str:
    return "[" + ", ".join(str(path) for path in paths) + "]"


def _extract_ancestry_group(
    source_pickle_name: str,
    *,
    supported_ancestry_groups: tuple[str, ...],
) -> str:
    normalized_source_name = Path(source_pickle_name).name
    stem = re.sub(r"(?:_add_covs)?\.pkl$", "", normalized_source_name, flags=re.IGNORECASE)
    normalized_stem = stem.upper()

    # Match against configured ancestry names rather than assuming the ancestry
    # is a single token after the final underscore. This keeps legacy AA/JA/LA
    # names working and also supports names like African_Ancestry.
    matching_groups = [
        ancestry_group
        for ancestry_group in supported_ancestry_groups
        if normalized_stem == ancestry_group
        or normalized_stem.endswith(f"_{ancestry_group}")
    ]
    if matching_groups:
        return max(matching_groups, key=len)

    raise ValueError(
        "Could not infer a supported ancestry group from source pickle name "
        f"{source_pickle_name!r}. Supported groups={list(supported_ancestry_groups)!r}."
    )


def _load_output_row_tracking(path: Path) -> pd.DataFrame:
    tracking_frame = pd.read_pickle(path)
    if not isinstance(tracking_frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame pickle at {path}.")
    required_columns = {
        "output_pickle_name",
        "output_row_number",
        "source_pickle_name",
    }
    missing_columns = sorted(required_columns - set(tracking_frame.columns))
    if missing_columns:
        raise ValueError(
            f"Tracking file {path} is missing required columns: {missing_columns}."
        )
    return tracking_frame


def _build_dataset_ancestry_groups(
    *,
    dataset_pickle_paths: tuple[Path, ...],
    dataset_lengths: tuple[int, ...],
    output_row_tracking_path: Path,
    supported_ancestry_groups: tuple[str, ...],
) -> tuple[str, ...]:
    tracking_frame = _load_output_row_tracking(output_row_tracking_path)
    ancestry_groups: list[str] = []

    for dataset_path, expected_length in zip(dataset_pickle_paths, dataset_lengths):
        output_name = dataset_path.name
        output_tracking = tracking_frame.loc[
            tracking_frame["output_pickle_name"] == output_name,
            ["output_row_number", "source_pickle_name"],
        ].sort_values("output_row_number")
        if len(output_tracking) != expected_length:
            raise ValueError(
                f"Tracking file {output_row_tracking_path} has {len(output_tracking)} rows for "
                f"{output_name}, expected {expected_length}."
            )
        expected_row_numbers = list(range(expected_length))
        actual_row_numbers = output_tracking["output_row_number"].astype(int).tolist()
        if actual_row_numbers != expected_row_numbers:
            raise ValueError(
                f"Tracking rows for {output_name} must cover output_row_number 0..{expected_length - 1} in order."
            )
        ancestry_groups.extend(
            _extract_ancestry_group(
                str(source_pickle_name),
                supported_ancestry_groups=supported_ancestry_groups,
            )
            for source_pickle_name in output_tracking["source_pickle_name"].tolist()
        )

    return tuple(ancestry_groups)


def _safe_group_roc_auc(labels: pd.Series, scores: pd.Series, *, ancestry_group: str) -> float:
    try:
        return _safe_roc_auc(labels, scores)
    except ValueError as exc:
        raise ValueError(
            f"Could not compute heldout ROC AUC for ancestry group {ancestry_group}: {exc}"
        ) from exc


def _evaluate_heldout_ancestry_groups(
    *,
    heldout_labels: pd.Series,
    heldout_risk_scores: Sequence[float],
    heldout_ancestry_groups: tuple[str, ...],
    supported_ancestry_groups: tuple[str, ...],
) -> tuple[HeldoutAncestryEvaluation, ...]:
    if len(heldout_ancestry_groups) != len(heldout_labels):
        raise ValueError(
            "Heldout ancestry-group assignments did not match heldout row count: "
            f"groups={len(heldout_ancestry_groups)} rows={len(heldout_labels)}."
        )

    ancestry_series = pd.Series(heldout_ancestry_groups, name="ancestry_group")
    label_series = pd.Series(heldout_labels, copy=False).reset_index(drop=True)
    score_series = pd.Series(heldout_risk_scores, copy=False).reset_index(drop=True)
    evaluations: list[HeldoutAncestryEvaluation] = []

    for ancestry_group in supported_ancestry_groups:
        group_mask = ancestry_series == ancestry_group
        subject_count = int(group_mask.sum())
        if subject_count == 0:
            continue
        group_auc = _safe_group_roc_auc(
            label_series.loc[group_mask].reset_index(drop=True),
            score_series.loc[group_mask].reset_index(drop=True),
            ancestry_group=ancestry_group,
        )
        evaluations.append(
            HeldoutAncestryEvaluation(
                ancestry_group=ancestry_group,
                subject_count=subject_count,
                auc_roc=group_auc,
            )
        )
    return tuple(evaluations)


def evaluate_priority_function(
    config: EvaluationConfig,
    *,
    baselines_to_run: tuple[BaselineSpec, ...] | None = None,
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
    cache_root = config.distance_cache_dir or _default_distance_cache_dir_for_priority_function(
        config.prio_function_path
    )
    program_source = config.prio_function_path.read_text()
    priority_function = _load_priority_function(program_source, config.function_name)
    _validate_priority_signature(priority_function)

    report(
        "Loading datasets: "
        f"training={_format_paths_for_logging(config.training_pickle_paths)} "
        f"calibration={_format_paths_for_logging(config.calibrating_pickle_paths)} "
        f"heldout={_format_paths_for_logging(config.heldout_pickle_paths)} "
        f"tracking={config.output_row_tracking_path}"
    )
    training_data, training_lengths = _read_and_impute_dataset(config.training_pickle_paths)
    calibration_data, calibration_lengths = _read_and_impute_dataset(
        config.calibrating_pickle_paths
    )
    heldout_data, heldout_lengths = _read_and_impute_dataset(config.heldout_pickle_paths)
    training_ancestry_groups = _build_dataset_ancestry_groups(
        dataset_pickle_paths=config.training_pickle_paths,
        dataset_lengths=training_lengths,
        output_row_tracking_path=config.output_row_tracking_path,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    calibration_ancestry_groups = _build_dataset_ancestry_groups(
        dataset_pickle_paths=config.calibrating_pickle_paths,
        dataset_lengths=calibration_lengths,
        output_row_tracking_path=config.output_row_tracking_path,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    heldout_ancestry_groups = _build_dataset_ancestry_groups(
        dataset_pickle_paths=config.heldout_pickle_paths,
        dataset_lengths=heldout_lengths,
        output_row_tracking_path=config.output_row_tracking_path,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    report(
        "Loaded datasets with rows: "
        f"training={len(training_data)} from {list(training_lengths)} "
        f"calibration={len(calibration_data)} from {list(calibration_lengths)} "
        f"heldout={len(heldout_data)} from {list(heldout_lengths)}"
    )
    variant_names = _list_variant_names(training_data)
    if not variant_names:
        raise ValueError("No dosage columns were found in the training pickle set.")
    report(f"Found {len(variant_names)} dosage variants in training data")

    calibration_labels = _extract_labels(calibration_data)
    calibration_distance_cache_manifest = None
    if config.distance_cache_enabled:
        report(f"Preparing calibration distance cache under {cache_root}")
        calibration_cache = ensure_distance_cache(
            reference_data=training_data,
            target_data=calibration_data,
            cache_root=cache_root,
            cache_name="postprocessing.calibration",
            reference_source_paths=tuple(str(path) for path in config.training_pickle_paths),
            target_source_paths=tuple(str(path) for path in config.calibrating_pickle_paths),
        )
        calibration_distance_cache_manifest = (
            calibration_cache.manifest_path if calibration_cache is not None else None
        )
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
        distance_cache_manifest_path=calibration_distance_cache_manifest,
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
    heldout_distance_cache_manifest = None
    if config.distance_cache_enabled:
        report(f"Preparing heldout distance cache under {cache_root}")
        heldout_cache = ensure_distance_cache(
            reference_data=oracle_training_for_heldout,
            target_data=heldout_data,
            cache_root=cache_root,
            cache_name="postprocessing.heldout",
            reference_source_paths=tuple(
                str(path) for path in config.training_pickle_paths + config.calibrating_pickle_paths
            ),
            target_source_paths=tuple(str(path) for path in config.heldout_pickle_paths),
        )
        heldout_distance_cache_manifest = heldout_cache.manifest_path if heldout_cache is not None else None
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
        distance_cache_manifest_path=heldout_distance_cache_manifest,
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
    heldout_ancestry_evaluations = _evaluate_heldout_ancestry_groups(
        heldout_labels=heldout_labels,
        heldout_risk_scores=heldout_risk_scores,
        heldout_ancestry_groups=heldout_ancestry_groups,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    for evaluation in heldout_ancestry_evaluations:
        report(
            "Priority-function heldout ROC AUC "
            f"for ancestry={evaluation.ancestry_group} subjects={evaluation.subject_count} "
            f"is {evaluation.auc_roc:.6f}"
        )

    selected_baselines = config.baselines if baselines_to_run is None else baselines_to_run
    baseline_evaluations: list[BaselineEvaluation] = []
    for baseline in selected_baselines:
        baseline_name = normalize_baseline_name(baseline.name)
        report(f"Running baseline {baseline_name}")
        baseline_result = evaluate_baseline(
            baseline.name,
            training_data=training_data,
            calibration_data=calibration_data,
            heldout_data=heldout_data,
            training_ancestry_groups=training_ancestry_groups,
            calibration_ancestry_groups=calibration_ancestry_groups,
            heldout_ancestry_groups=heldout_ancestry_groups,
            options=baseline.options,
        )
        if not isinstance(baseline_result, BaselineResult):
            raise TypeError(
                f"Baseline {baseline_name} returned {type(baseline_result).__name__}, "
                "expected BaselineResult."
            )
        baseline_ancestry_evaluations = _evaluate_heldout_ancestry_groups(
            heldout_labels=heldout_labels,
            heldout_risk_scores=baseline_result.heldout_scores,
            heldout_ancestry_groups=heldout_ancestry_groups,
            supported_ancestry_groups=config.supported_ancestry_groups,
        )
        report(
            f"Finished baseline {baseline_name} with heldout ROC AUC={baseline_result.auc_roc:.6f}"
        )
        for evaluation in baseline_ancestry_evaluations:
            report(
                f"Baseline {baseline_name} heldout ROC AUC for ancestry={evaluation.ancestry_group} "
                f"subjects={evaluation.subject_count} is {evaluation.auc_roc:.6f}"
            )
        baseline_evaluations.append(
            BaselineEvaluation(
                name=baseline_name,
                auc_roc=baseline_result.auc_roc,
                heldout_ancestry_evaluations=baseline_ancestry_evaluations,
            )
        )

    return EvaluationReport(
        prio_function_path=config.prio_function_path,
        heldout_auc_roc=heldout_auc_roc,
        heldout_ancestry_evaluations=heldout_ancestry_evaluations,
        baseline_evaluations=tuple(baseline_evaluations),
    )


def evaluate_from_config_path(
    config_path: str | Path,
    *,
    baselines_to_run: tuple[BaselineSpec, ...] | None = None,
    progress_reporter: Callable[[str], None] | None = None,
) -> EvaluationReport:
    return evaluate_priority_function(
        load_evaluation_config(config_path),
        baselines_to_run=baselines_to_run,
        progress_reporter=progress_reporter,
    )


def _parse_heldout_ancestry_evaluations(
    raw_values: Any,
) -> tuple[HeldoutAncestryEvaluation, ...]:
    if raw_values is None:
        return ()
    if not isinstance(raw_values, list):
        raise ValueError("heldout_ancestry_evaluations must be a JSON array.")
    parsed: list[HeldoutAncestryEvaluation] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, dict):
            raise ValueError("Each heldout ancestry evaluation must be a JSON object.")
        parsed.append(
            HeldoutAncestryEvaluation(
                ancestry_group=str(raw_value["ancestry_group"]),
                subject_count=int(raw_value["subject_count"]),
                auc_roc=float(raw_value["auc_roc"]),
            )
        )
    return tuple(parsed)


def load_existing_evaluation_report(report_path: str | Path) -> EvaluationReport:
    report_path = Path(report_path).expanduser().resolve()
    raw_payload = json.loads(report_path.read_text())
    if not isinstance(raw_payload, dict) or "results" not in raw_payload:
        raise ValueError(f"Existing evaluation report at {report_path} is malformed.")
    raw_results = raw_payload["results"]
    if not isinstance(raw_results, dict):
        raise ValueError(f"Existing evaluation report at {report_path} has malformed results.")
    raw_baselines = raw_results.get("baseline_evaluations", [])
    if not isinstance(raw_baselines, list):
        raise ValueError("baseline_evaluations must be a JSON array.")
    baseline_evaluations: list[BaselineEvaluation] = []
    for raw_baseline in raw_baselines:
        if not isinstance(raw_baseline, dict):
            raise ValueError("Each baseline evaluation must be a JSON object.")
        baseline_evaluations.append(
            BaselineEvaluation(
                name=str(raw_baseline["name"]),
                auc_roc=float(raw_baseline["auc_roc"]),
                heldout_ancestry_evaluations=_parse_heldout_ancestry_evaluations(
                    raw_baseline.get("heldout_ancestry_evaluations", [])
                ),
            )
        )
    return EvaluationReport(
        prio_function_path=Path(str(raw_results["prio_function_path"])).expanduser().resolve(),
        heldout_auc_roc=float(raw_results["heldout_auc_roc"]),
        heldout_ancestry_evaluations=_parse_heldout_ancestry_evaluations(
            raw_results.get("heldout_ancestry_evaluations", [])
        ),
        baseline_evaluations=tuple(baseline_evaluations),
    )


def merge_evaluation_reports(
    existing_report: EvaluationReport,
    additional_report: EvaluationReport,
) -> EvaluationReport:
    if existing_report.prio_function_path != additional_report.prio_function_path:
        raise ValueError("Cannot merge reports for different priority-function paths.")
    existing_by_name = {
        normalize_baseline_name(baseline.name): baseline
        for baseline in existing_report.baseline_evaluations
    }
    for baseline in additional_report.baseline_evaluations:
        existing_by_name[normalize_baseline_name(baseline.name)] = baseline
    merged_baselines = tuple(existing_by_name[name] for name in sorted(existing_by_name))
    return EvaluationReport(
        prio_function_path=existing_report.prio_function_path,
        heldout_auc_roc=existing_report.heldout_auc_roc,
        heldout_ancestry_evaluations=existing_report.heldout_ancestry_evaluations,
        baseline_evaluations=merged_baselines,
    )


def format_report(report: EvaluationReport) -> str:
    lines = [
        f"prio_function_path={report.prio_function_path}",
        f"heldout_auc_roc={report.heldout_auc_roc:.6f}",
    ]
    lines.extend(
        f"heldout_subject_count[{evaluation.ancestry_group}]={evaluation.subject_count}"
        for evaluation in report.heldout_ancestry_evaluations
    )
    lines.extend(
        f"heldout_auc_roc[{evaluation.ancestry_group}]={evaluation.auc_roc:.6f}"
        for evaluation in report.heldout_ancestry_evaluations
    )
    lines.extend(
        f"baseline_auc_roc[{baseline.name}]={baseline.auc_roc:.6f}"
        for baseline in report.baseline_evaluations
    )
    lines.extend(
        f"baseline_auc_roc[{baseline.name}][{evaluation.ancestry_group}]={evaluation.auc_roc:.6f}"
        for baseline in report.baseline_evaluations
        for evaluation in baseline.heldout_ancestry_evaluations
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


def _missing_baselines(
    configured_baselines: tuple[BaselineSpec, ...],
    existing_report: EvaluationReport,
) -> tuple[BaselineSpec, ...]:
    existing_names = {
        normalize_baseline_name(baseline.name)
        for baseline in existing_report.baseline_evaluations
    }
    return tuple(
        baseline
        for baseline in configured_baselines
        if normalize_baseline_name(baseline.name) not in existing_names
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_evaluation_config(args.config_path)
    report_path = report_output_path_for_priority_function(
        config.prio_function_path,
        config.report_file_name,
    )
    if config.should_overwrite or not report_path.exists():
        report = evaluate_priority_function(
            config,
            progress_reporter=_report_progress,
        )
    else:
        existing_report = load_existing_evaluation_report(report_path)
        if existing_report.prio_function_path != config.prio_function_path:
            raise ValueError(
                f"Existing evaluation report {report_path} does not match current priority function "
                f"{config.prio_function_path}."
            )
        missing_baselines = _missing_baselines(config.baselines, existing_report)
        if missing_baselines:
            _report_progress(
                "Existing report found with missing baselines: "
                + ", ".join(normalize_baseline_name(baseline.name) for baseline in missing_baselines)
            )
            additional_report = evaluate_priority_function(
                config,
                baselines_to_run=missing_baselines,
                progress_reporter=_report_progress,
            )
            report = merge_evaluation_reports(existing_report, additional_report)
        else:
            _report_progress(
                f"Existing report already contains all configured baselines; reusing {report_path}"
            )
            report = existing_report
    report_path = write_evaluation_report_file(
        config_path=args.config_path,
        config=config,
        report=report,
    )
    print(format_report(report))
    print(f"evaluation_report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())