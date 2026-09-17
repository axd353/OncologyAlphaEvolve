from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import argparse
import hashlib
import json
import pickle
import re
import time

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
from PostProcesingData.evaluate_priofunction import HeldoutAncestryEvaluation
from PostProcesingData.evaluate_priofunction import _build_dataset_tracking_rows
from PostProcesingData.evaluate_priofunction import _build_progress_reporter
from PostProcesingData.evaluate_priofunction import _evaluate_heldout_ancestry_groups
from PostProcesingData.evaluate_priofunction import _format_paths_for_logging
from PostProcesingData.evaluate_priofunction import _initialize_progress_log
from PostProcesingData.evaluate_priofunction import _parse_partition_count
from PostProcesingData.evaluate_priofunction import _parse_supported_ancestry_groups
from PostProcesingData.evaluate_priofunction import _read_and_impute_dataset
from PostProcesingData.evaluate_priofunction import _resolve_partition_count
from PostProcesingData.evaluate_priofunction import _resolve_path
from PostProcesingData.evaluate_priofunction import _resolve_path_list
from PostProcesingData.evaluate_priofunction import _visible_cpu_count
from PostProcesingData.evaluate_priofunction_all_cycles import _sorted_cycle_priority_paths


_RUN_CONFIG_FILE_NAME = "config.used.json"
_DEFAULT_OUTPUT_ROOT_DIR = Path(__file__).resolve().parent / "OutputDir"


@dataclass(frozen=True)
class DynamicsOfHeldOutConfig:
    run_dir: Path
    heldout_pickle_paths: tuple[Path, ...]
    calibrating_pickle_paths: tuple[Path, ...]
    training_pickle_paths: tuple[Path, ...]
    output_row_tracking_path: Path
    supported_ancestry_groups: tuple[str, ...]
    heldout_target_ancestry_group: str | None
    function_name: str
    calibration_penalties: tuple[float, ...]
    calibration_partitions: int | None
    scoring_partitions: int | None
    distance_cache_enabled: bool
    distance_cache_dir: Path | None
    output_root_dir: Path
    output_dir_tag: str | None
    seed_prio_function_path: Path | None


@dataclass(frozen=True)
class PreparedEvaluationDatasets:
    training_data: pd.DataFrame
    calibration_data: pd.DataFrame
    heldout_reference_data: pd.DataFrame
    heldout_target_data: pd.DataFrame
    heldout_target_tracking_rows: pd.DataFrame
    heldout_target_ancestry_groups: tuple[str, ...]
    variant_names: tuple[str, ...]
    calibration_distance_cache_manifest: Path | None
    heldout_distance_cache_manifest: Path | None


@dataclass(frozen=True)
class PriorityFunctionReference:
    label: str
    source_kind: str
    cycle_index: int | None
    prio_function_path: Path
    discovery_snapshot_path: Path
    discovery_source_island_id: int | None
    discovery_reduced_score: float | None
    discovery_scores_per_test: dict[str, float]
    program_source: str
    priority_source_hash: str


@dataclass(frozen=True)
class PriorityFunctionEvaluation:
    label: str
    source_kind: str
    cycle_index: int | None
    prio_function_path: Path
    discovery_snapshot_path: Path
    discovery_source_island_id: int | None
    discovery_reduced_score: float | None
    discovery_scores_per_test: dict[str, float]
    priority_source_hash: str
    program_source: str
    reused_from_label: str | None
    selected_calibration_penalty: float
    heldout_subject_count: int
    heldout_auc_roc: float
    heldout_ancestry_evaluations: tuple[HeldoutAncestryEvaluation, ...]


@dataclass(frozen=True)
class DynamicsOfHeldOutReport:
    config: DynamicsOfHeldOutConfig
    output_dir: Path
    evaluations: tuple[PriorityFunctionEvaluation, ...]


def _default_distance_cache_dir(
    run_dir: Path,
    heldout_target_ancestry_group: str | None,
) -> Path:
    scope = heldout_target_ancestry_group.lower() if heldout_target_ancestry_group else "all_heldout"
    return run_dir / f"dynamics_of_heldout_distance_cache_{scope}"


def _parse_calibration_penalties(raw_value: Any) -> tuple[float, ...]:
    if raw_value is None:
        return (0.1, 1.0, 10.0)
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("calibration_penalties must be a non-empty JSON array.")
    return tuple(float(value) for value in raw_value)


def _parse_optional_ancestry_group(
    raw_value: Any,
    *,
    supported_ancestry_groups: tuple[str, ...],
) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError("heldout_target_ancestry_group must be a string or null.")
    ancestry_group = raw_value.strip().upper()
    if not ancestry_group:
        raise ValueError("heldout_target_ancestry_group must not be empty when provided.")
    if ancestry_group not in supported_ancestry_groups:
        raise ValueError(
            "heldout_target_ancestry_group must be one of supported_ancestry_groups: "
            f"target={ancestry_group!r} supported={list(supported_ancestry_groups)!r}."
        )
    return ancestry_group


def _parse_output_dir_tag(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError("output_dir_tag must be a string or null.")
    tag = raw_value.strip()
    if not tag:
        return None
    return tag


def _sanitize_output_dir_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-")
    return sanitized or None


def _config_to_payload(config: DynamicsOfHeldOutConfig) -> dict[str, Any]:
    return {
        "run_dir": str(config.run_dir),
        "training_pickle_path": [str(path) for path in config.training_pickle_paths],
        "calibrating_pickle_path": [str(path) for path in config.calibrating_pickle_paths],
        "heldout_pickle_path": [str(path) for path in config.heldout_pickle_paths],
        "output_row_tracking_path": str(config.output_row_tracking_path),
        "supported_ancestry_groups": list(config.supported_ancestry_groups),
        "heldout_target_ancestry_group": config.heldout_target_ancestry_group,
        "function_name": config.function_name,
        "calibration_penalties": list(config.calibration_penalties),
        "calibration_partitions": config.calibration_partitions,
        "scoring_partitions": config.scoring_partitions,
        "distance_cache_enabled": config.distance_cache_enabled,
        "distance_cache_dir": str(config.distance_cache_dir) if config.distance_cache_dir is not None else None,
        "output_root_dir": str(config.output_root_dir),
        "output_dir_tag": config.output_dir_tag,
        "seed_prio_function_path": (
            str(config.seed_prio_function_path) if config.seed_prio_function_path is not None else None
        ),
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


def _evaluations_to_payload(
    evaluations: tuple[PriorityFunctionEvaluation, ...],
    *,
    priority_function_output_paths: dict[str, Path] | None = None,
    priority_function_metadata_paths: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "label": evaluation.label,
            "source_kind": evaluation.source_kind,
            "cycle_index": evaluation.cycle_index,
            "prio_function_path": str(evaluation.prio_function_path),
            "discovery_snapshot_path": str(evaluation.discovery_snapshot_path),
            "discovery_source_island_id": evaluation.discovery_source_island_id,
            "discovery_reduced_score": evaluation.discovery_reduced_score,
            "discovery_scores_per_test": dict(sorted(evaluation.discovery_scores_per_test.items())),
            "priority_source_hash": evaluation.priority_source_hash,
            "priority_function_artifact_path": (
                str(priority_function_output_paths[evaluation.label])
                if priority_function_output_paths is not None
                else None
            ),
            "priority_function_artifact_metadata_path": (
                str(priority_function_metadata_paths[evaluation.label])
                if priority_function_metadata_paths is not None
                else None
            ),
            "reused_from_label": evaluation.reused_from_label,
            "selected_calibration_penalty": evaluation.selected_calibration_penalty,
            "heldout_subject_count": evaluation.heldout_subject_count,
            "heldout_auc_roc": evaluation.heldout_auc_roc,
            "heldout_ancestry_evaluations": _heldout_ancestry_evaluations_to_payload(
                evaluation.heldout_ancestry_evaluations
            ),
        }
        for evaluation in evaluations
    ]


def load_config(config_path: str | Path) -> DynamicsOfHeldOutConfig:
    normalized_config_path = Path(config_path).expanduser().resolve()
    raw_config = json.loads(normalized_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("The evaluation config must be a JSON object.")

    if "run_dir" not in raw_config:
        raise ValueError("Missing required config field 'run_dir'.")

    base_dir = normalized_config_path.parent
    supported_ancestry_groups = _parse_supported_ancestry_groups(
        raw_config.get("supported_ancestry_groups", list(DEFAULT_SUPPORTED_ANCESTRY_GROUPS))
    )
    return DynamicsOfHeldOutConfig(
        run_dir=_resolve_path(base_dir, str(raw_config["run_dir"])),
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
        heldout_target_ancestry_group=_parse_optional_ancestry_group(
            raw_config.get("heldout_target_ancestry_group"),
            supported_ancestry_groups=supported_ancestry_groups,
        ),
        function_name=str(raw_config.get("function_name", DEFAULT_FUNCTION_NAME)),
        calibration_penalties=_parse_calibration_penalties(
            raw_config.get("calibration_penalties")
        ),
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
        output_root_dir=_resolve_path(
            base_dir,
            str(raw_config.get("output_root_dir", _DEFAULT_OUTPUT_ROOT_DIR)),
        ),
        output_dir_tag=_parse_output_dir_tag(raw_config.get("output_dir_tag")),
        seed_prio_function_path=(
            _resolve_path(base_dir, str(raw_config["seed_prio_function_path"]))
            if raw_config.get("seed_prio_function_path")
            else None
        ),
    )


def _filter_dataset_to_target_ancestry(
    *,
    data_frame: pd.DataFrame,
    tracking_rows: pd.DataFrame,
    target_ancestry_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data_frame) != len(tracking_rows):
        raise ValueError(
            "Heldout row count did not match tracking rows: "
            f"data={len(data_frame)} tracking={len(tracking_rows)}."
        )
    ancestry_mask = tracking_rows["ancestry_group"] == target_ancestry_group
    selected_tracking_rows = tracking_rows.loc[ancestry_mask].reset_index(drop=True)
    selected_data_frame = data_frame.loc[ancestry_mask.to_numpy(copy=False)].reset_index(drop=True)
    if selected_data_frame.empty:
        raise ValueError(
            f"No heldout rows matched target ancestry {target_ancestry_group!r}."
        )
    return selected_data_frame, selected_tracking_rows


def _priority_source_hash(program_source: str) -> str:
    return hashlib.sha256(program_source.encode("utf-8")).hexdigest()


def _normalize_discovery_scores(raw_scores: Any) -> dict[str, float]:
    if raw_scores is None:
        return {}
    if not isinstance(raw_scores, dict):
        raise TypeError(
            "Expected snapshot scores_per_test to be a dict or null, "
            f"got {type(raw_scores).__name__}."
        )
    return {str(name): float(value) for name, value in raw_scores.items()}


def _load_best_program_artifact(snapshot_path: Path) -> Any:
    with snapshot_path.open("rb") as handle:
        database = pickle.load(handle)
    build_artifact = getattr(database, "build_global_best_program_artifact", None)
    if not callable(build_artifact):
        raise TypeError(
            f"Snapshot at {snapshot_path} did not contain a compatible program database."
        )
    artifact = build_artifact()
    if artifact is None:
        raise ValueError(f"No best program found in snapshot: {snapshot_path}")
    return artifact


def _resolve_seed_priority_path(config: DynamicsOfHeldOutConfig) -> Path:
    if config.seed_prio_function_path is not None:
        return config.seed_prio_function_path

    run_config_path = config.run_dir / _RUN_CONFIG_FILE_NAME
    if not run_config_path.exists():
        raise ValueError(
            "Could not infer the seed priority function because the run config copy is missing: "
            f"{run_config_path}. Set seed_prio_function_path explicitly in the troubleshooting config."
        )

    raw_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_run_config, dict):
        raise ValueError(f"Run config at {run_config_path} is malformed.")
    experiment_config = raw_run_config.get("experiment")
    if not isinstance(experiment_config, dict):
        raise ValueError(f"Run config at {run_config_path} is missing the 'experiment' object.")
    raw_seed_path = experiment_config.get("seed_priority_path")
    if not isinstance(raw_seed_path, str) or not raw_seed_path.strip():
        raise ValueError(
            "Run config did not contain a usable experiment.seed_priority_path; "
            "set seed_prio_function_path explicitly in the troubleshooting config."
        )
    return _resolve_path(run_config_path.parent, raw_seed_path)


def _discover_priority_functions(
    config: DynamicsOfHeldOutConfig,
    *,
    progress_reporter: Callable[[str], None],
) -> tuple[PriorityFunctionReference, ...]:
    seed_priority_path = _resolve_seed_priority_path(config)
    bootstrap_snapshot_path = config.run_dir / "program_db" / "bootstrap.pkl"
    if not bootstrap_snapshot_path.exists():
        raise ValueError(
            "Could not recover seed discovery metadata because the bootstrap snapshot is missing: "
            f"{bootstrap_snapshot_path}"
        )
    seed_artifact = _load_best_program_artifact(bootstrap_snapshot_path)
    cycle_priority_paths = _sorted_cycle_priority_paths(config.run_dir)
    if not cycle_priority_paths:
        raise ValueError(
            f"No completed cycle directories with best_prio.py were found under {config.run_dir}."
        )

    references = [
        PriorityFunctionReference(
            label="seed",
            source_kind="seed",
            cycle_index=None,
            prio_function_path=seed_priority_path,
            discovery_snapshot_path=bootstrap_snapshot_path,
            discovery_source_island_id=int(seed_artifact.island_id),
            discovery_reduced_score=float(seed_artifact.reduced_score),
            discovery_scores_per_test=_normalize_discovery_scores(seed_artifact.scores_per_test),
            program_source=str(seed_artifact.program_source),
            priority_source_hash=_priority_source_hash(str(seed_artifact.program_source)),
        )
    ]
    for cycle_index, prio_function_path in cycle_priority_paths:
        snapshot_path = prio_function_path.parent / "program_db_end.pkl"
        if not snapshot_path.exists():
            raise ValueError(
                f"Missing cycle snapshot for {prio_function_path}: {snapshot_path}"
            )
        cycle_artifact = _load_best_program_artifact(snapshot_path)
        program_source = str(cycle_artifact.program_source)
        references.append(
            PriorityFunctionReference(
                label=f"cycle_{cycle_index:04d}",
                source_kind="cycle_best",
                cycle_index=cycle_index,
                prio_function_path=prio_function_path,
                discovery_snapshot_path=snapshot_path,
                discovery_source_island_id=int(cycle_artifact.island_id),
                discovery_reduced_score=float(cycle_artifact.reduced_score),
                discovery_scores_per_test=_normalize_discovery_scores(
                    cycle_artifact.scores_per_test
                ),
                program_source=program_source,
                priority_source_hash=_priority_source_hash(program_source),
            )
        )
    progress_reporter(
        f"Discovered {len(cycle_priority_paths)} cycle best priority functions plus the initial seed"
    )
    return tuple(references)


def _prepare_evaluation_datasets(
    config: DynamicsOfHeldOutConfig,
    *,
    progress_reporter: Callable[[str], None],
) -> PreparedEvaluationDatasets:
    progress_reporter(
        "Loading datasets: "
        f"training={_format_paths_for_logging(config.training_pickle_paths)} "
        f"calibration={_format_paths_for_logging(config.calibrating_pickle_paths)} "
        f"heldout={_format_paths_for_logging(config.heldout_pickle_paths)} "
        f"tracking={config.output_row_tracking_path}"
    )
    training_data, _ = _read_and_impute_dataset(config.training_pickle_paths)
    calibration_data, _ = _read_and_impute_dataset(config.calibrating_pickle_paths)
    heldout_data, heldout_lengths = _read_and_impute_dataset(config.heldout_pickle_paths)
    heldout_tracking_rows = _build_dataset_tracking_rows(
        dataset_pickle_paths=config.heldout_pickle_paths,
        dataset_lengths=heldout_lengths,
        output_row_tracking_path=config.output_row_tracking_path,
        supported_ancestry_groups=config.supported_ancestry_groups,
    )
    if config.heldout_target_ancestry_group is not None:
        heldout_target_data, heldout_target_tracking_rows = _filter_dataset_to_target_ancestry(
            data_frame=heldout_data,
            tracking_rows=heldout_tracking_rows,
            target_ancestry_group=config.heldout_target_ancestry_group,
        )
    else:
        heldout_target_data = heldout_data
        heldout_target_tracking_rows = heldout_tracking_rows

    heldout_target_ancestry_groups = tuple(
        heldout_target_tracking_rows["ancestry_group"].tolist()
    )
    variant_names = tuple(_list_variant_names(training_data))
    if not variant_names:
        raise ValueError("No dosage columns were found in the training pickle set.")

    heldout_reference_data = _combine_data_objects([training_data, calibration_data])
    cache_root = config.distance_cache_dir or _default_distance_cache_dir(
        config.run_dir,
        config.heldout_target_ancestry_group,
    )
    calibration_distance_cache_manifest = None
    heldout_distance_cache_manifest = None
    if config.distance_cache_enabled:
        progress_reporter(f"Preparing calibration distance cache under {cache_root}")
        calibration_cache = ensure_distance_cache(
            reference_data=training_data,
            target_data=calibration_data,
            cache_root=cache_root,
            cache_name="postprocessing.dynamics_of_heldout.calibration_full",
            reference_source_paths=tuple(str(path) for path in config.training_pickle_paths),
            target_source_paths=tuple(str(path) for path in config.calibrating_pickle_paths),
        )
        calibration_distance_cache_manifest = (
            calibration_cache.manifest_path if calibration_cache is not None else None
        )

        heldout_scope = config.heldout_target_ancestry_group or "ALL"
        progress_reporter(
            f"Preparing heldout distance cache under {cache_root} for scope {heldout_scope}"
        )
        heldout_cache = ensure_distance_cache(
            reference_data=heldout_reference_data,
            target_data=heldout_target_data,
            cache_root=cache_root,
            cache_name=f"postprocessing.dynamics_of_heldout.heldout.{heldout_scope.lower()}",
            reference_source_paths=tuple(
                str(path) for path in config.training_pickle_paths + config.calibrating_pickle_paths
            ),
            target_source_paths=tuple(str(path) for path in config.heldout_pickle_paths),
        )
        heldout_distance_cache_manifest = (
            heldout_cache.manifest_path if heldout_cache is not None else None
        )

    progress_reporter(
        "Prepared evaluation datasets with rows: "
        f"training={len(training_data)} calibration={len(calibration_data)} "
        f"heldout_full={len(heldout_data)} heldout_target={len(heldout_target_data)}"
    )
    return PreparedEvaluationDatasets(
        training_data=training_data,
        calibration_data=calibration_data,
        heldout_reference_data=heldout_reference_data,
        heldout_target_data=heldout_target_data,
        heldout_target_tracking_rows=heldout_target_tracking_rows,
        heldout_target_ancestry_groups=heldout_target_ancestry_groups,
        variant_names=variant_names,
        calibration_distance_cache_manifest=calibration_distance_cache_manifest,
        heldout_distance_cache_manifest=heldout_distance_cache_manifest,
    )


def _evaluate_priority_function_reference(
    config: DynamicsOfHeldOutConfig,
    prepared_datasets: PreparedEvaluationDatasets,
    priority_reference: PriorityFunctionReference,
    *,
    progress_reporter: Callable[[str], None],
    progress_log_path: Path | None,
) -> PriorityFunctionEvaluation:
    calibration_partitions = _resolve_partition_count(config.calibration_partitions)
    scoring_partitions = _resolve_partition_count(config.scoring_partitions)
    priority_function = _load_priority_function(
        priority_reference.program_source,
        config.function_name,
    )
    _validate_priority_signature(priority_function)

    calibration_labels = _extract_labels(prepared_datasets.calibration_data)
    progress_reporter(
        f"{priority_reference.label}: building calibration oracle features on {len(prepared_datasets.calibration_data)} rows"
    )
    calibration_oracle_features = _build_calibration_oracle_feature_matrix(
        training_data=prepared_datasets.training_data,
        calibration_data=prepared_datasets.calibration_data,
        variant_names=prepared_datasets.variant_names,
        candidate_source=priority_reference.program_source,
        function_name=config.function_name,
        calibration_partitions=calibration_partitions,
        distance_cache_manifest_path=prepared_datasets.calibration_distance_cache_manifest,
        progress_reporter=progress_reporter,
        progress_log_path=progress_log_path,
        progress_label=f"{priority_reference.label} calibration oracle features",
    )
    calibration_covariates = _extract_covariates(
        prepared_datasets.calibration_data,
        include_covariates=False,
    )
    calibration_model = _fit_best_calibration_model(
        oracle_features=calibration_oracle_features,
        covariates=calibration_covariates,
        labels=calibration_labels,
        penalties=config.calibration_penalties,
    )
    progress_reporter(
        f"{priority_reference.label}: selected calibration penalty {calibration_model.penalty}"
    )

    heldout_labels = _extract_labels(prepared_datasets.heldout_target_data)
    progress_reporter(
        f"{priority_reference.label}: building heldout oracle features on {len(prepared_datasets.heldout_target_data)} rows"
    )
    heldout_oracle_features = _build_scoring_oracle_feature_matrix(
        training_data=prepared_datasets.heldout_reference_data,
        scoring_data=prepared_datasets.heldout_target_data,
        variant_names=prepared_datasets.variant_names,
        candidate_source=priority_reference.program_source,
        function_name=config.function_name,
        scoring_partitions=scoring_partitions,
        distance_cache_manifest_path=prepared_datasets.heldout_distance_cache_manifest,
        progress_reporter=progress_reporter,
        progress_log_path=progress_log_path,
        progress_label=f"{priority_reference.label} heldout oracle features",
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
        f"{priority_reference.label}: heldout ROC AUC={heldout_auc_roc:.6f}"
    )
    return PriorityFunctionEvaluation(
        label=priority_reference.label,
        source_kind=priority_reference.source_kind,
        cycle_index=priority_reference.cycle_index,
        prio_function_path=priority_reference.prio_function_path,
        discovery_snapshot_path=priority_reference.discovery_snapshot_path,
        discovery_source_island_id=priority_reference.discovery_source_island_id,
        discovery_reduced_score=priority_reference.discovery_reduced_score,
        discovery_scores_per_test=dict(priority_reference.discovery_scores_per_test),
        priority_source_hash=priority_reference.priority_source_hash,
        program_source=priority_reference.program_source,
        reused_from_label=None,
        selected_calibration_penalty=float(calibration_model.penalty),
        heldout_subject_count=len(prepared_datasets.heldout_target_data),
        heldout_auc_roc=heldout_auc_roc,
        heldout_ancestry_evaluations=heldout_ancestry_evaluations,
    )


def evaluate_run_dynamics(
    config: DynamicsOfHeldOutConfig,
    *,
    progress_reporter: Callable[[str], None] | None = None,
    progress_log_path: Path | None = None,
) -> tuple[PriorityFunctionEvaluation, ...]:
    report = progress_reporter or (lambda message: None)
    calibration_partitions = _resolve_partition_count(config.calibration_partitions)
    scoring_partitions = _resolve_partition_count(config.scoring_partitions)
    visible_cpu_count = _visible_cpu_count()
    heldout_scope = config.heldout_target_ancestry_group or "ALL"
    report(
        "Runtime CPU detection: "
        f"visible_cpus={visible_cpu_count} calibration_workers={calibration_partitions} "
        f"scoring_workers={scoring_partitions}"
    )
    report(
        f"Starting dynamics-of-heldout evaluation for run={config.run_dir} heldout_scope={heldout_scope}"
    )
    priority_references = _discover_priority_functions(
        config,
        progress_reporter=report,
    )
    prepared_datasets = _prepare_evaluation_datasets(
        config,
        progress_reporter=report,
    )
    unique_results_by_hash: dict[str, PriorityFunctionEvaluation] = {}
    evaluations: list[PriorityFunctionEvaluation] = []
    for priority_reference in priority_references:
        existing_evaluation = unique_results_by_hash.get(priority_reference.priority_source_hash)
        if existing_evaluation is not None:
            report(
                f"Skipping {priority_reference.label}; source matches {existing_evaluation.label}"
            )
            evaluations.append(
                PriorityFunctionEvaluation(
                    label=priority_reference.label,
                    source_kind=priority_reference.source_kind,
                    cycle_index=priority_reference.cycle_index,
                    prio_function_path=priority_reference.prio_function_path,
                    discovery_snapshot_path=priority_reference.discovery_snapshot_path,
                    discovery_source_island_id=priority_reference.discovery_source_island_id,
                    discovery_reduced_score=priority_reference.discovery_reduced_score,
                    discovery_scores_per_test=dict(priority_reference.discovery_scores_per_test),
                    priority_source_hash=priority_reference.priority_source_hash,
                    program_source=priority_reference.program_source,
                    reused_from_label=existing_evaluation.label,
                    selected_calibration_penalty=existing_evaluation.selected_calibration_penalty,
                    heldout_subject_count=existing_evaluation.heldout_subject_count,
                    heldout_auc_roc=existing_evaluation.heldout_auc_roc,
                    heldout_ancestry_evaluations=existing_evaluation.heldout_ancestry_evaluations,
                )
            )
            continue

        report(
            f"Evaluating {priority_reference.label} from {priority_reference.prio_function_path}"
        )
        evaluation = _evaluate_priority_function_reference(
            config,
            prepared_datasets,
            priority_reference,
            progress_reporter=report,
            progress_log_path=progress_log_path,
        )
        unique_results_by_hash[priority_reference.priority_source_hash] = evaluation
        evaluations.append(evaluation)
    return tuple(evaluations)


def _evaluations_to_frame(
    config: DynamicsOfHeldOutConfig,
    evaluations: tuple[PriorityFunctionEvaluation, ...],
    *,
    priority_function_output_paths: dict[str, Path],
    priority_function_metadata_paths: dict[str, Path],
) -> pd.DataFrame:
    discovery_score_keys = sorted(
        {
            score_name
            for evaluation in evaluations
            for score_name in evaluation.discovery_scores_per_test
        }
    )
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        ancestry_by_name = {
            ancestry_evaluation.ancestry_group: ancestry_evaluation
            for ancestry_evaluation in evaluation.heldout_ancestry_evaluations
        }
        row: dict[str, Any] = {
            "priority_label": evaluation.label,
            "source_kind": evaluation.source_kind,
            "cycle_index": evaluation.cycle_index,
            "prio_function_path": str(evaluation.prio_function_path),
            "priority_function_artifact_path": str(
                priority_function_output_paths[evaluation.label]
            ),
            "priority_function_artifact_metadata_path": str(
                priority_function_metadata_paths[evaluation.label]
            ),
            "discovery_snapshot_path": str(evaluation.discovery_snapshot_path),
            "discovery_source_island_id": evaluation.discovery_source_island_id,
            "discovery_reduced_score": evaluation.discovery_reduced_score,
            "priority_source_hash": evaluation.priority_source_hash,
            "reused_from_label": evaluation.reused_from_label,
            "selected_calibration_penalty": evaluation.selected_calibration_penalty,
            "heldout_subject_count": evaluation.heldout_subject_count,
            "heldout_auc_roc": evaluation.heldout_auc_roc,
        }
        for score_name in discovery_score_keys:
            row[f"discovery_score[{score_name}]"] = evaluation.discovery_scores_per_test.get(
                score_name
            )
        for ancestry_group in config.supported_ancestry_groups:
            ancestry_evaluation = ancestry_by_name.get(ancestry_group)
            row[f"heldout_subject_count[{ancestry_group}]"] = (
                ancestry_evaluation.subject_count if ancestry_evaluation is not None else None
            )
            row[f"heldout_auc_roc[{ancestry_group}]"] = (
                ancestry_evaluation.auc_roc if ancestry_evaluation is not None else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary_markdown(
    report: DynamicsOfHeldOutReport,
    *,
    report_table_path: Path,
    summary_json_path: Path,
    priority_functions_dir: Path,
) -> str:
    unique_priority_function_count = len(
        {evaluation.priority_source_hash for evaluation in report.evaluations}
    )
    heldout_scope = report.config.heldout_target_ancestry_group or "ALL"
    lines = [
        "# Dynamics Of Heldout Report",
        "",
        f"- run_dir: {report.config.run_dir}",
        f"- heldout_scope: {heldout_scope}",
        f"- evaluated_rows: {len(report.evaluations)}",
        f"- unique_priority_functions: {unique_priority_function_count}",
        f"- report_table: {report_table_path.name}",
        f"- summary_json: {summary_json_path.name}",
        f"- priority_functions_dir: {priority_functions_dir.name}",
        "",
        "## Priority Functions",
        "",
    ]
    for evaluation in report.evaluations:
        reused_suffix = (
            f" (reused from {evaluation.reused_from_label})"
            if evaluation.reused_from_label is not None
            else ""
        )
        lines.append(
            f"- {evaluation.label}: heldout_auc_roc={evaluation.heldout_auc_roc:.6f}{reused_suffix}"
        )
    return "\n".join(lines) + "\n"


def _build_output_dir(config: DynamicsOfHeldOutConfig) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sanitized_tag = _sanitize_output_dir_tag(config.output_dir_tag)
    directory_name = timestamp if sanitized_tag is None else f"{timestamp}_{sanitized_tag}"
    output_dir = config.output_root_dir / directory_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_used_config_file(
    *,
    config_path: str | Path,
    config: DynamicsOfHeldOutConfig,
    output_dir: Path,
) -> Path:
    payload = {
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config": _config_to_payload(config),
    }
    output_path = output_dir / "config.used.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _write_summary_json(
    *,
    config_path: str | Path,
    report: DynamicsOfHeldOutReport,
    output_dir: Path,
    priority_function_output_paths: dict[str, Path],
    priority_function_metadata_paths: dict[str, Path],
) -> Path:
    payload = {
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config": _config_to_payload(report.config),
        "results": {
            "output_dir": str(output_dir),
            "priority_functions_dir": str(output_dir / "priority_functions"),
            "evaluations": _evaluations_to_payload(
                report.evaluations,
                priority_function_output_paths=priority_function_output_paths,
                priority_function_metadata_paths=priority_function_metadata_paths,
            ),
        },
    }
    output_path = output_dir / "heldout_priority_scores.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _priority_function_artifact_file_stem(evaluation: PriorityFunctionEvaluation) -> str:
    if evaluation.cycle_index is None:
        return "seed"
    return f"cycle_{evaluation.cycle_index:04d}"


def _write_priority_function_artifacts(
    *,
    output_dir: Path,
    evaluations: tuple[PriorityFunctionEvaluation, ...],
) -> tuple[dict[str, Path], dict[str, Path]]:
    artifacts_dir = output_dir / "priority_functions"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {}
    metadata_paths: dict[str, Path] = {}

    for evaluation in evaluations:
        file_stem = _priority_function_artifact_file_stem(evaluation)
        source_output_path = artifacts_dir / f"{file_stem}.py"
        metadata_output_path = artifacts_dir / f"{file_stem}.metadata.json"
        metadata_payload = {
            "label": evaluation.label,
            "source_kind": evaluation.source_kind,
            "cycle_index": evaluation.cycle_index,
            "original_prio_function_path": str(evaluation.prio_function_path),
            "discovery_snapshot_path": str(evaluation.discovery_snapshot_path),
            "discovery_source_island_id": evaluation.discovery_source_island_id,
            "discovery_reduced_score": evaluation.discovery_reduced_score,
            "discovery_scores_per_test": dict(sorted(evaluation.discovery_scores_per_test.items())),
            "priority_source_hash": evaluation.priority_source_hash,
            "reused_from_label": evaluation.reused_from_label,
            "selected_calibration_penalty": evaluation.selected_calibration_penalty,
            "heldout_subject_count": evaluation.heldout_subject_count,
            "heldout_auc_roc": evaluation.heldout_auc_roc,
            "heldout_ancestry_evaluations": _heldout_ancestry_evaluations_to_payload(
                evaluation.heldout_ancestry_evaluations
            ),
        }
        file_lines = [
            f"# label={evaluation.label}",
            f"# source_kind={evaluation.source_kind}",
            f"# cycle_index={evaluation.cycle_index}",
            f"# original_prio_function_path={evaluation.prio_function_path}",
            f"# discovery_snapshot_path={evaluation.discovery_snapshot_path}",
            f"# discovery_source_island_id={evaluation.discovery_source_island_id}",
            f"# discovery_reduced_score={evaluation.discovery_reduced_score}",
            "# discovery_scores_per_test="
            + json.dumps(metadata_payload["discovery_scores_per_test"], sort_keys=True),
            f"# priority_source_hash={evaluation.priority_source_hash}",
            f"# reused_from_label={evaluation.reused_from_label}",
            f"# selected_calibration_penalty={evaluation.selected_calibration_penalty}",
            f"# heldout_subject_count={evaluation.heldout_subject_count}",
            f"# heldout_auc_roc={evaluation.heldout_auc_roc}",
        ]
        source_output_path.write_text(
            "\n".join(file_lines) + "\n\n" + evaluation.program_source,
            encoding="utf-8",
        )
        metadata_output_path.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_paths[evaluation.label] = source_output_path
        metadata_paths[evaluation.label] = metadata_output_path

    return output_paths, metadata_paths


def run_from_config_path(config_path: str | Path) -> DynamicsOfHeldOutReport:
    config = load_config(config_path)
    output_dir = _build_output_dir(config)
    _write_used_config_file(config_path=config_path, config=config, output_dir=output_dir)
    progress_log_path = output_dir / "dynamics_of_heldout.progress.log"
    _initialize_progress_log(progress_log_path)
    progress_reporter = _build_progress_reporter(progress_log_path)
    progress_reporter(
        "Starting dynamics-of-heldout run for "
        f"config={Path(config_path).expanduser().resolve()}"
    )
    progress_reporter(f"Output directory={output_dir}")

    evaluations = evaluate_run_dynamics(
        config,
        progress_reporter=progress_reporter,
        progress_log_path=progress_log_path,
    )
    report = DynamicsOfHeldOutReport(
        config=config,
        output_dir=output_dir,
        evaluations=evaluations,
    )
    priority_function_output_paths, priority_function_metadata_paths = _write_priority_function_artifacts(
        output_dir=output_dir,
        evaluations=evaluations,
    )
    report_table_path = output_dir / "heldout_priority_scores.csv"
    _evaluations_to_frame(
        config,
        evaluations,
        priority_function_output_paths=priority_function_output_paths,
        priority_function_metadata_paths=priority_function_metadata_paths,
    ).to_csv(report_table_path, index=False)
    summary_json_path = _write_summary_json(
        config_path=config_path,
        report=report,
        output_dir=output_dir,
        priority_function_output_paths=priority_function_output_paths,
        priority_function_metadata_paths=priority_function_metadata_paths,
    )
    summary_markdown_path = output_dir / "summary.md"
    summary_markdown_path.write_text(
        _build_summary_markdown(
            report,
            report_table_path=report_table_path,
            summary_json_path=summary_json_path,
            priority_functions_dir=output_dir / "priority_functions",
        ),
        encoding="utf-8",
    )
    progress_reporter(f"Saved priority-function artifacts to {output_dir / 'priority_functions'}")
    progress_reporter(f"Saved report table to {report_table_path}")
    progress_reporter(f"Saved summary JSON to {summary_json_path}")
    progress_reporter(f"Saved summary markdown to {summary_markdown_path}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the initial seed priority function and each cycle's best_prio.py "
            "on heldout data, using the same calibration-then-heldout flow as "
            "PostProcesingData.evaluate_priofunction."
        )
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to a JSON config file for troubleshooting heldout dynamics.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_from_config_path(args.config_path)
    report_table_path = report.output_dir / "heldout_priority_scores.csv"
    print(f"output_dir={report.output_dir}")
    print(f"report_table={report_table_path}")
    for evaluation in report.evaluations:
        reused_suffix = (
            f" reused_from={evaluation.reused_from_label}"
            if evaluation.reused_from_label is not None
            else ""
        )
        print(
            f"{evaluation.label} heldout_auc_roc={evaluation.heldout_auc_roc:.6f}"
            f" subjects={evaluation.heldout_subject_count}{reused_suffix}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())