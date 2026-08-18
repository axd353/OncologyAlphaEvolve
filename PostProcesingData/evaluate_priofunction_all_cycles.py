from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import argparse
import hashlib
import json
import re

import pandas as pd

from funsearch_pipeline.evaluation.procedure2 import _build_calibration_oracle_feature_matrix
from funsearch_pipeline.evaluation.procedure2 import _build_scoring_oracle_feature_matrix
from funsearch_pipeline.evaluation.procedure2 import _combine_data_objects
from funsearch_pipeline.evaluation.procedure2 import _extract_covariates
from funsearch_pipeline.evaluation.procedure2 import _extract_labels
from funsearch_pipeline.evaluation.procedure2 import _fit_best_calibration_model
from funsearch_pipeline.evaluation.procedure2 import _list_variant_names
from funsearch_pipeline.evaluation.procedure2 import _load_priority_function
from funsearch_pipeline.evaluation.procedure2 import _predict_linear_score
from funsearch_pipeline.evaluation.procedure2 import _safe_roc_auc
from funsearch_pipeline.evaluation.procedure2 import _validate_priority_signature
from GenomicsHelpers.ancestry_distance_cache import ensure_distance_cache

from PostProcesingData.evaluate_priofunction import DEFAULT_CALIBRATING_PICKLE_PATH
from PostProcesingData.evaluate_priofunction import DEFAULT_FUNCTION_NAME
from PostProcesingData.evaluate_priofunction import DEFAULT_HELDOUT_PICKLE_PATH
from PostProcesingData.evaluate_priofunction import DEFAULT_OUTPUT_ROW_TRACKING_PATH
from PostProcesingData.evaluate_priofunction import DEFAULT_SUPPORTED_ANCESTRY_GROUPS
from PostProcesingData.evaluate_priofunction import DEFAULT_TRAINING_PICKLE_PATH
from PostProcesingData.evaluate_priofunction import BaselineEvaluation
from PostProcesingData.evaluate_priofunction import BaselineSpec
from PostProcesingData.evaluate_priofunction import HeldoutAncestryEvaluation
from PostProcesingData.evaluate_priofunction import _build_dataset_ancestry_groups
from PostProcesingData.evaluate_priofunction import _build_dataset_tracking_rows
from PostProcesingData.evaluate_priofunction import _build_progress_reporter
from PostProcesingData.evaluate_priofunction import _evaluate_heldout_ancestry_groups
from PostProcesingData.evaluate_priofunction import _format_paths_for_logging
from PostProcesingData.evaluate_priofunction import _initialize_progress_log
from PostProcesingData.evaluate_priofunction import _parse_baselines
from PostProcesingData.evaluate_priofunction import _parse_partition_count
from PostProcesingData.evaluate_priofunction import _parse_supported_ancestry_groups
from PostProcesingData.evaluate_priofunction import _read_and_impute_dataset
from PostProcesingData.evaluate_priofunction import _resolve_partition_count
from PostProcesingData.evaluate_priofunction import _resolve_path
from PostProcesingData.evaluate_priofunction import _resolve_path_list
from PostProcesingData.evaluate_priofunction import _visible_cpu_count
from PostProcesingData.prio_func_eval_baselines import BaselineResult
from PostProcesingData.prio_func_eval_baselines import evaluate_baseline
from PostProcesingData.prio_func_eval_baselines import normalize_baseline_name


_CYCLE_DIR_PATTERN = re.compile(r"cycle_(\d+)$")


@dataclass(frozen=True)
class AllCyclesEvaluationConfig:
    run_dir: Path
    report_file_name: str
    target_ancestry_group: str
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
class PreparedTargetAncestryDatasets:
    training_data: pd.DataFrame
    calibration_reference_data: pd.DataFrame
    heldout_reference_data: pd.DataFrame
    calibration_target_data: pd.DataFrame
    heldout_target_data: pd.DataFrame
    training_ancestry_groups: tuple[str, ...]
    calibration_target_ancestry_groups: tuple[str, ...]
    heldout_target_ancestry_groups: tuple[str, ...]
    heldout_target_tracking_rows: pd.DataFrame
    variant_names: tuple[str, ...]
    calibration_distance_cache_manifest: Path | None
    heldout_distance_cache_manifest: Path | None


@dataclass(frozen=True)
class CyclePriorityEvaluation:
    cycle_index: int
    cycle_dir: Path
    prio_function_path: Path
    priority_source_hash: str
    reused_from_cycle_index: int | None
    heldout_auc_roc: float
    heldout_ancestry_evaluations: tuple[HeldoutAncestryEvaluation, ...]


@dataclass(frozen=True)
class AllCyclesEvaluationReport:
    run_dir: Path
    target_ancestry_group: str
    calibration_subject_count: int
    heldout_subject_count: int
    baseline_evaluations: tuple[BaselineEvaluation, ...]
    cycle_evaluations: tuple[CyclePriorityEvaluation, ...]


def _default_report_file_name(target_ancestry_group: str) -> str:
    return f"best_prio.all_cycles.{target_ancestry_group}.evaluation_report.json"


def _default_distance_cache_dir(
    run_dir: Path,
    target_ancestry_group: str,
) -> Path:
    return run_dir / f"distance_cache_all_cycles_{target_ancestry_group}"


def _parse_file_name(raw_value: Any, *, default_value: str) -> str:
    if raw_value is None:
        return default_value
    if not isinstance(raw_value, str):
        raise ValueError("report_file_name must be a string.")
    report_file_name = raw_value.strip()
    if not report_file_name:
        raise ValueError("report_file_name must not be empty.")
    report_path = Path(report_file_name)
    if report_path.is_absolute() or report_path.name != report_file_name or report_file_name in {".", ".."}:
        raise ValueError("report_file_name must be a file name, not a path.")
    return report_file_name


def _parse_target_ancestry_group(
    raw_value: Any,
    *,
    supported_ancestry_groups: tuple[str, ...],
) -> str:
    if not isinstance(raw_value, str):
        raise ValueError("Missing required config field 'target_ancestry_group'.")
    target_ancestry_group = raw_value.strip().upper()
    if not target_ancestry_group:
        raise ValueError("target_ancestry_group must not be empty.")
    if target_ancestry_group not in supported_ancestry_groups:
        raise ValueError(
            "target_ancestry_group must be one of supported_ancestry_groups: "
            f"target={target_ancestry_group!r} supported={list(supported_ancestry_groups)!r}."
        )
    return target_ancestry_group


def _config_to_payload(config: AllCyclesEvaluationConfig) -> dict[str, Any]:
    return {
        "run_dir": str(config.run_dir),
        "report_file_name": config.report_file_name,
        "target_ancestry_group": config.target_ancestry_group,
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


def _heldout_ancestry_evaluations_to_payload(
    evaluations: tuple[HeldoutAncestryEvaluation, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "ancestry_group": evaluation.ancestry_group,
            "subject_count": evaluation.subject_count,
            "auc_roc": evaluation.auc_roc,
        }
        for evaluation in evaluations
    ]


def _baseline_evaluations_to_payload(
    baseline_evaluations: tuple[BaselineEvaluation, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "name": baseline.name,
            "auc_roc": baseline.auc_roc,
            "heldout_ancestry_evaluations": _heldout_ancestry_evaluations_to_payload(
                baseline.heldout_ancestry_evaluations
            ),
        }
        for baseline in baseline_evaluations
    ]


def _cycle_evaluations_to_payload(
    cycle_evaluations: tuple[CyclePriorityEvaluation, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "cycle_index": cycle_evaluation.cycle_index,
            "cycle_dir": str(cycle_evaluation.cycle_dir),
            "prio_function_path": str(cycle_evaluation.prio_function_path),
            "priority_source_hash": cycle_evaluation.priority_source_hash,
            "reused_from_cycle_index": cycle_evaluation.reused_from_cycle_index,
            "heldout_auc_roc": cycle_evaluation.heldout_auc_roc,
            "heldout_ancestry_evaluations": _heldout_ancestry_evaluations_to_payload(
                cycle_evaluation.heldout_ancestry_evaluations
            ),
        }
        for cycle_evaluation in cycle_evaluations
    ]


def format_all_cycles_report(report: AllCyclesEvaluationReport) -> str:
    unique_priority_function_count = len(
        {cycle.priority_source_hash for cycle in report.cycle_evaluations}
    )
    lines = [
        f"run_dir={report.run_dir}",
        f"target_ancestry_group={report.target_ancestry_group}",
        f"calibration_subject_count={report.calibration_subject_count}",
        f"heldout_subject_count={report.heldout_subject_count}",
        f"cycle_count={len(report.cycle_evaluations)}",
        f"unique_priority_function_count={unique_priority_function_count}",
    ]
    lines.extend(
        f"baseline_auc_roc[{baseline.name}]={baseline.auc_roc:.6f}"
        for baseline in report.baseline_evaluations
    )
    lines.extend(
        f"baseline_auc_roc[{baseline.name}][{evaluation.ancestry_group}]={evaluation.auc_roc:.6f}"
        for baseline in report.baseline_evaluations
        for evaluation in baseline.heldout_ancestry_evaluations
    )
    for cycle in report.cycle_evaluations:
        cycle_label = f"cycle_{cycle.cycle_index:04d}"
        lines.append(f"{cycle_label}.prio_function_path={cycle.prio_function_path}")
        lines.append(f"{cycle_label}.heldout_auc_roc={cycle.heldout_auc_roc:.6f}")
        if cycle.reused_from_cycle_index is not None:
            lines.append(
                f"{cycle_label}.reused_from_cycle=cycle_{cycle.reused_from_cycle_index:04d}"
            )
        lines.extend(
            f"{cycle_label}.heldout_subject_count[{evaluation.ancestry_group}]={evaluation.subject_count}"
            for evaluation in cycle.heldout_ancestry_evaluations
        )
        lines.extend(
            f"{cycle_label}.heldout_auc_roc[{evaluation.ancestry_group}]={evaluation.auc_roc:.6f}"
            for evaluation in cycle.heldout_ancestry_evaluations
        )
    return "\n".join(lines)


def _report_to_payload(report: AllCyclesEvaluationReport) -> dict[str, Any]:
    return {
        "run_dir": str(report.run_dir),
        "target_ancestry_group": report.target_ancestry_group,
        "calibration_subject_count": report.calibration_subject_count,
        "heldout_subject_count": report.heldout_subject_count,
        "baseline_evaluations": _baseline_evaluations_to_payload(
            report.baseline_evaluations
        ),
        "cycle_evaluations": _cycle_evaluations_to_payload(report.cycle_evaluations),
        "summary_text": format_all_cycles_report(report),
    }


def write_all_cycles_evaluation_report_file(
    *,
    config_path: str | Path,
    config: AllCyclesEvaluationConfig,
    report: AllCyclesEvaluationReport,
) -> Path:
    output_path = config.run_dir / config.report_file_name
    payload = {
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config": _config_to_payload(config),
        "results": _report_to_payload(report),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def load_all_cycles_evaluation_config(config_path: str | Path) -> AllCyclesEvaluationConfig:
    config_path = Path(config_path).expanduser().resolve()
    raw_config = json.loads(config_path.read_text())
    if not isinstance(raw_config, dict):
        raise ValueError("The evaluation config must be a JSON object.")

    base_dir = config_path.parent
    if "run_dir" not in raw_config:
        raise ValueError("Missing required config field 'run_dir'.")

    calibration_penalties = raw_config.get("calibration_penalties", [0.1, 1.0, 10.0])
    if not isinstance(calibration_penalties, list) or not calibration_penalties:
        raise ValueError("calibration_penalties must be a non-empty JSON array.")

    baselines = tuple(
        baseline
        for baseline in _parse_baselines(raw_config.get("baselines", []))
        if baseline.enabled
    )
    supported_ancestry_groups = _parse_supported_ancestry_groups(
        raw_config.get("supported_ancestry_groups", list(DEFAULT_SUPPORTED_ANCESTRY_GROUPS))
    )
    target_ancestry_group = _parse_target_ancestry_group(
        raw_config.get("target_ancestry_group"),
        supported_ancestry_groups=supported_ancestry_groups,
    )
    run_dir = _resolve_path(base_dir, str(raw_config["run_dir"]))
    return AllCyclesEvaluationConfig(
        run_dir=run_dir,
        report_file_name=_parse_file_name(
            raw_config.get("report_file_name"),
            default_value=_default_report_file_name(target_ancestry_group),
        ),
        target_ancestry_group=target_ancestry_group,
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
        supported_ancestry_groups=supported_ancestry_groups,
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


def _sorted_cycle_priority_paths(run_dir: Path) -> tuple[tuple[int, Path], ...]:
    cycle_priority_paths: list[tuple[int, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        match = _CYCLE_DIR_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        priority_path = child / "best_prio.py"
        if not priority_path.exists():
            continue
        cycle_priority_paths.append((int(match.group(1)), priority_path.resolve()))
    return tuple(sorted(cycle_priority_paths, key=lambda item: item[0]))


def _filter_dataset_to_target_ancestry(
    *,
    data_frame: pd.DataFrame,
    tracking_rows: pd.DataFrame,
    target_ancestry_group: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data_frame) != len(tracking_rows):
        raise ValueError(
            f"{dataset_name} row count did not match tracking rows: "
            f"data={len(data_frame)} tracking={len(tracking_rows)}."
        )
    ancestry_mask = tracking_rows["ancestry_group"] == target_ancestry_group
    selected_tracking_rows = tracking_rows.loc[ancestry_mask].reset_index(drop=True)
    selected_data_frame = data_frame.loc[ancestry_mask.to_numpy(copy=False)].reset_index(drop=True)
    if selected_data_frame.empty:
        raise ValueError(
            f"No {dataset_name} rows matched target ancestry {target_ancestry_group!r}."
        )
    return selected_data_frame, selected_tracking_rows


def _priority_source_hash(program_source: str) -> str:
    return hashlib.sha256(program_source.encode("utf-8")).hexdigest()


def _prepare_target_ancestry_datasets(
    config: AllCyclesEvaluationConfig,
    *,
    progress_reporter: Callable[[str], None],
) -> PreparedTargetAncestryDatasets:
    progress_reporter(
        "Loading datasets: "
        f"training={_format_paths_for_logging(config.training_pickle_paths)} "
        f"calibration={_format_paths_for_logging(config.calibrating_pickle_paths)} "
        f"heldout={_format_paths_for_logging(config.heldout_pickle_paths)} "
        f"tracking={config.output_row_tracking_path}"
    )
    training_data, training_lengths = _read_and_impute_dataset(config.training_pickle_paths)
    calibration_reference_data, calibration_lengths = _read_and_impute_dataset(
        config.calibrating_pickle_paths
    )
    heldout_data, heldout_lengths = _read_and_impute_dataset(config.heldout_pickle_paths)
    training_ancestry_groups = _build_dataset_ancestry_groups(
        dataset_pickle_paths=config.training_pickle_paths,
        dataset_lengths=training_lengths,
        output_row_tracking_path=config.output_row_tracking_path,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    calibration_tracking_rows = _build_dataset_tracking_rows(
        dataset_pickle_paths=config.calibrating_pickle_paths,
        dataset_lengths=calibration_lengths,
        output_row_tracking_path=config.output_row_tracking_path,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    heldout_tracking_rows = _build_dataset_tracking_rows(
        dataset_pickle_paths=config.heldout_pickle_paths,
        dataset_lengths=heldout_lengths,
        output_row_tracking_path=config.output_row_tracking_path,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    calibration_target_data, calibration_target_tracking_rows = _filter_dataset_to_target_ancestry(
        data_frame=calibration_reference_data,
        tracking_rows=calibration_tracking_rows,
        target_ancestry_group=config.target_ancestry_group,
        dataset_name="calibration",
    )
    heldout_target_data, heldout_target_tracking_rows = _filter_dataset_to_target_ancestry(
        data_frame=heldout_data,
        tracking_rows=heldout_tracking_rows,
        target_ancestry_group=config.target_ancestry_group,
        dataset_name="heldout",
    )
    calibration_target_ancestry_groups = tuple(
        calibration_target_tracking_rows["ancestry_group"].tolist()
    )
    heldout_target_ancestry_groups = tuple(
        heldout_target_tracking_rows["ancestry_group"].tolist()
    )

    variant_names = _list_variant_names(training_data)
    if not variant_names:
        raise ValueError("No dosage columns were found in the training pickle set.")

    progress_reporter(
        "Prepared target-ancestry datasets with rows: "
        f"training={len(training_data)} "
        f"calibration_full={len(calibration_reference_data)} "
        f"calibration_target={len(calibration_target_data)} "
        f"heldout_full={len(heldout_data)} "
        f"heldout_target={len(heldout_target_data)} "
        f"target_ancestry={config.target_ancestry_group}"
    )
    progress_reporter(f"Found {len(variant_names)} dosage variants in training data")

    cache_root = config.distance_cache_dir or _default_distance_cache_dir(
        config.run_dir,
        config.target_ancestry_group,
    )
    calibration_distance_cache_manifest = None
    heldout_distance_cache_manifest = None
    heldout_reference_data = _combine_data_objects(
        [training_data, calibration_reference_data]
    )
    if config.distance_cache_enabled:
        progress_reporter(
            f"Preparing calibration distance cache under {cache_root} for target ancestry {config.target_ancestry_group}"
        )
        calibration_cache = ensure_distance_cache(
            reference_data=training_data,
            target_data=calibration_target_data,
            cache_root=cache_root,
            cache_name=f"postprocessing.all_cycles.{config.target_ancestry_group}.calibration",
            reference_source_paths=tuple(str(path) for path in config.training_pickle_paths),
            target_source_paths=tuple(str(path) for path in config.calibrating_pickle_paths),
        )
        calibration_distance_cache_manifest = (
            calibration_cache.manifest_path if calibration_cache is not None else None
        )

        progress_reporter(
            f"Preparing heldout distance cache under {cache_root} for target ancestry {config.target_ancestry_group}"
        )
        heldout_cache = ensure_distance_cache(
            reference_data=heldout_reference_data,
            target_data=heldout_target_data,
            cache_root=cache_root,
            cache_name=f"postprocessing.all_cycles.{config.target_ancestry_group}.heldout",
            reference_source_paths=tuple(
                str(path) for path in config.training_pickle_paths + config.calibrating_pickle_paths
            ),
            target_source_paths=tuple(str(path) for path in config.heldout_pickle_paths),
        )
        heldout_distance_cache_manifest = (
            heldout_cache.manifest_path if heldout_cache is not None else None
        )

    return PreparedTargetAncestryDatasets(
        training_data=training_data,
        calibration_reference_data=calibration_reference_data,
        heldout_reference_data=heldout_reference_data,
        calibration_target_data=calibration_target_data,
        heldout_target_data=heldout_target_data,
        training_ancestry_groups=training_ancestry_groups,
        calibration_target_ancestry_groups=calibration_target_ancestry_groups,
        heldout_target_ancestry_groups=heldout_target_ancestry_groups,
        heldout_target_tracking_rows=heldout_target_tracking_rows,
        variant_names=tuple(variant_names),
        calibration_distance_cache_manifest=calibration_distance_cache_manifest,
        heldout_distance_cache_manifest=heldout_distance_cache_manifest,
    )


def _evaluate_baselines_once(
    config: AllCyclesEvaluationConfig,
    prepared_datasets: PreparedTargetAncestryDatasets,
    *,
    progress_reporter: Callable[[str], None],
) -> tuple[BaselineEvaluation, ...]:
    if not config.baselines:
        return ()

    heldout_labels = _extract_labels(prepared_datasets.heldout_target_data)
    baseline_evaluations: list[BaselineEvaluation] = []
    for baseline in config.baselines:
        baseline_name = normalize_baseline_name(baseline.name)
        progress_reporter(
            f"Running baseline {baseline_name} for target ancestry {config.target_ancestry_group}"
        )
        baseline_result = evaluate_baseline(
            baseline.name,
            training_data=prepared_datasets.training_data,
            calibration_data=prepared_datasets.calibration_target_data,
            heldout_data=prepared_datasets.heldout_target_data,
            training_ancestry_groups=prepared_datasets.training_ancestry_groups,
            calibration_ancestry_groups=prepared_datasets.calibration_target_ancestry_groups,
            heldout_ancestry_groups=prepared_datasets.heldout_target_ancestry_groups,
            options=baseline.options,
        )
        if not isinstance(baseline_result, BaselineResult):
            raise TypeError(
                f"Baseline {baseline_name} returned {type(baseline_result).__name__}, expected BaselineResult."
            )
        baseline_ancestry_evaluations = _evaluate_heldout_ancestry_groups(
            heldout_labels=heldout_labels,
            heldout_risk_scores=baseline_result.heldout_scores,
            heldout_ancestry_groups=prepared_datasets.heldout_target_ancestry_groups,
            supported_ancestry_groups=config.supported_ancestry_groups,
        )
        progress_reporter(
            f"Finished baseline {baseline_name} with heldout ROC AUC={baseline_result.auc_roc:.6f}"
        )
        baseline_evaluations.append(
            BaselineEvaluation(
                name=baseline_name,
                auc_roc=baseline_result.auc_roc,
                heldout_ancestry_evaluations=baseline_ancestry_evaluations,
            )
        )
    return tuple(baseline_evaluations)


def _evaluate_cycle_priority_function(
    config: AllCyclesEvaluationConfig,
    prepared_datasets: PreparedTargetAncestryDatasets,
    *,
    cycle_index: int,
    prio_function_path: Path,
    program_source: str,
    progress_reporter: Callable[[str], None],
    progress_log_path: Path | None,
) -> CyclePriorityEvaluation:
    calibration_partitions = _resolve_partition_count(config.calibration_partitions)
    scoring_partitions = _resolve_partition_count(config.scoring_partitions)
    priority_function = _load_priority_function(program_source, config.function_name)
    _validate_priority_signature(priority_function)

    calibration_labels = _extract_labels(prepared_datasets.calibration_target_data)
    progress_reporter(
        f"Cycle {cycle_index:04d}: building calibration oracle features on {len(prepared_datasets.calibration_target_data)} target-ancestry rows"
    )
    calibration_oracle_features = _build_calibration_oracle_feature_matrix(
        training_data=prepared_datasets.training_data,
        calibration_data=prepared_datasets.calibration_target_data,
        variant_names=prepared_datasets.variant_names,
        candidate_source=program_source,
        function_name=config.function_name,
        calibration_partitions=calibration_partitions,
        distance_cache_manifest_path=prepared_datasets.calibration_distance_cache_manifest,
        progress_reporter=progress_reporter,
        progress_log_path=progress_log_path,
        progress_label=f"cycle_{cycle_index:04d} calibration oracle features",
    )
    calibration_covariates = _extract_covariates(
        prepared_datasets.calibration_target_data,
        include_covariates=False,
    )
    calibration_model = _fit_best_calibration_model(
        oracle_features=calibration_oracle_features,
        covariates=calibration_covariates,
        labels=calibration_labels,
        penalties=config.calibration_penalties,
    )
    progress_reporter(
        f"Cycle {cycle_index:04d}: selected calibration penalty {calibration_model.penalty}"
    )

    heldout_labels = _extract_labels(prepared_datasets.heldout_target_data)
    progress_reporter(
        f"Cycle {cycle_index:04d}: building heldout oracle features on {len(prepared_datasets.heldout_target_data)} target-ancestry rows with full training+calibration reference data"
    )
    heldout_oracle_features = _build_scoring_oracle_feature_matrix(
        training_data=prepared_datasets.heldout_reference_data,
        scoring_data=prepared_datasets.heldout_target_data,
        variant_names=prepared_datasets.variant_names,
        candidate_source=program_source,
        function_name=config.function_name,
        scoring_partitions=scoring_partitions,
        distance_cache_manifest_path=prepared_datasets.heldout_distance_cache_manifest,
        progress_reporter=progress_reporter,
        progress_log_path=progress_log_path,
        progress_label=f"cycle_{cycle_index:04d} heldout oracle features",
    )
    heldout_covariates = _extract_covariates(
        prepared_datasets.heldout_target_data,
        include_covariates=False,
    )
    heldout_risk_scores = _predict_linear_score(
        calibration_model,
        oracle_features=heldout_oracle_features,
        covariates=heldout_covariates,
    )
    heldout_auc_roc = _safe_roc_auc(heldout_labels, heldout_risk_scores)
    heldout_ancestry_evaluations = _evaluate_heldout_ancestry_groups(
        heldout_labels=heldout_labels,
        heldout_risk_scores=heldout_risk_scores,
        heldout_ancestry_groups=prepared_datasets.heldout_target_ancestry_groups,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    progress_reporter(
        f"Cycle {cycle_index:04d}: heldout ROC AUC={heldout_auc_roc:.6f} for target ancestry {config.target_ancestry_group}"
    )
    return CyclePriorityEvaluation(
        cycle_index=cycle_index,
        cycle_dir=prio_function_path.parent,
        prio_function_path=prio_function_path,
        priority_source_hash=_priority_source_hash(program_source),
        reused_from_cycle_index=None,
        heldout_auc_roc=heldout_auc_roc,
        heldout_ancestry_evaluations=heldout_ancestry_evaluations,
    )


def evaluate_all_cycles(
    config: AllCyclesEvaluationConfig,
    *,
    progress_reporter: Callable[[str], None] | None = None,
    progress_log_path: Path | None = None,
) -> AllCyclesEvaluationReport:
    report = progress_reporter or (lambda message: None)
    calibration_partitions = _resolve_partition_count(config.calibration_partitions)
    scoring_partitions = _resolve_partition_count(config.scoring_partitions)
    visible_cpu_count = _visible_cpu_count()
    report(
        "Runtime CPU detection: "
        f"visible_cpus={visible_cpu_count} calibration_workers={calibration_partitions} "
        f"scoring_workers={scoring_partitions}"
    )
    report(
        f"Starting all-cycle priority-function evaluation for run={config.run_dir} target_ancestry={config.target_ancestry_group}"
    )

    cycle_priority_paths = _sorted_cycle_priority_paths(config.run_dir)
    if not cycle_priority_paths:
        raise ValueError(
            f"No completed cycle directories with best_prio.py were found under {config.run_dir}."
        )
    report(f"Discovered {len(cycle_priority_paths)} completed cycles with best_prio.py")

    prepared_datasets = _prepare_target_ancestry_datasets(
        config,
        progress_reporter=report,
    )
    baseline_evaluations = _evaluate_baselines_once(
        config,
        prepared_datasets,
        progress_reporter=report,
    )

    unique_cycle_results_by_hash: dict[str, CyclePriorityEvaluation] = {}
    cycle_evaluations: list[CyclePriorityEvaluation] = []
    for cycle_index, prio_function_path in cycle_priority_paths:
        program_source = prio_function_path.read_text(encoding="utf-8")
        source_hash = _priority_source_hash(program_source)
        existing_cycle = unique_cycle_results_by_hash.get(source_hash)
        if existing_cycle is not None:
            report(
                f"Skipping cycle {cycle_index:04d}; best_prio.py matches cycle {existing_cycle.cycle_index:04d}"
            )
            cycle_evaluations.append(
                CyclePriorityEvaluation(
                    cycle_index=cycle_index,
                    cycle_dir=prio_function_path.parent,
                    prio_function_path=prio_function_path,
                    priority_source_hash=source_hash,
                    reused_from_cycle_index=existing_cycle.cycle_index,
                    heldout_auc_roc=existing_cycle.heldout_auc_roc,
                    heldout_ancestry_evaluations=existing_cycle.heldout_ancestry_evaluations,
                )
            )
            continue

        report(f"Evaluating cycle {cycle_index:04d} from {prio_function_path}")
        cycle_evaluation = _evaluate_cycle_priority_function(
            config,
            prepared_datasets,
            cycle_index=cycle_index,
            prio_function_path=prio_function_path,
            program_source=program_source,
            progress_reporter=report,
            progress_log_path=progress_log_path,
        )
        unique_cycle_results_by_hash[source_hash] = cycle_evaluation
        cycle_evaluations.append(cycle_evaluation)

    return AllCyclesEvaluationReport(
        run_dir=config.run_dir,
        target_ancestry_group=config.target_ancestry_group,
        calibration_subject_count=len(prepared_datasets.calibration_target_data),
        heldout_subject_count=len(prepared_datasets.heldout_target_data),
        baseline_evaluations=baseline_evaluations,
        cycle_evaluations=tuple(cycle_evaluations),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate best_prio.py across all completed FunSearch cycles for one target ancestry, "
            "reusing results when consecutive or repeated cycle files have identical contents."
        )
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to a JSON config file for all-cycle heldout priority-function evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_all_cycles_evaluation_config(args.config_path)
    cache_root = config.distance_cache_dir or _default_distance_cache_dir(
        config.run_dir,
        config.target_ancestry_group,
    )
    progress_log_path = cache_root / "evaluate_priofunction_all_cycles.progress.log"
    _initialize_progress_log(progress_log_path)
    progress_reporter = _build_progress_reporter(progress_log_path)
    progress_reporter(
        "Starting evaluate_priofunction_all_cycles for "
        f"config={Path(args.config_path).expanduser().resolve()}"
    )
    progress_reporter(f"Progress log path={progress_log_path}")

    report_path = config.run_dir / config.report_file_name
    if not config.should_overwrite and report_path.exists():
        progress_reporter(f"Reusing existing report at {report_path}")
        raw_payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError(f"Existing report at {report_path} is malformed.")
        raw_results = raw_payload.get("results")
        if not isinstance(raw_results, dict) or not isinstance(raw_results.get("summary_text"), str):
            raise ValueError(f"Existing report at {report_path} is malformed.")
        print(raw_results["summary_text"])
        return 0

    report = evaluate_all_cycles(
        config,
        progress_reporter=progress_reporter,
        progress_log_path=progress_log_path,
    )
    output_path = write_all_cycles_evaluation_report_file(
        config_path=args.config_path,
        config=config,
        report=report,
    )
    progress_reporter(f"Saved all-cycle evaluation report to {output_path}")
    print(format_all_cycles_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())