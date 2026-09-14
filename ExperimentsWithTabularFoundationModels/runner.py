from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence
import argparse
import hashlib
import json
import logging
import math
import os
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GenomicsHelpers.ancestry_distance_cache import activate_distance_context
from GenomicsHelpers.ancestry_distance_cache import ensure_distance_cache
from GenomicsHelpers.ancestry_distance_cache import load_distance_cache
from GenomicsHelpers.oracle_data_adapter import DEFAULT_ANCESTRY_FIELDS
from GenomicsHelpers.oracle_data_adapter import DEFAULT_LABEL_FIELD
from GenomicsHelpers.oracle_data_adapter import DOSAGE_COLUMN_PREFIX
from PostProcesingData.evaluate_priofunction import _build_dataset_tracking_rows
from PostProcesingData.heldout_model_auc_ci import _bootstrap_auc_confidence_interval
from funsearch_pipeline.evaluation.procedure2 import _build_priority_ancestry_coordinate
from funsearch_pipeline.evaluation.procedure2 import _build_priority_target_variants
from funsearch_pipeline.evaluation.procedure2 import _build_priority_training_data_contract
from funsearch_pipeline.evaluation.procedure2 import _call_priority_function
from funsearch_pipeline.evaluation.procedure2 import _combine_data_objects
from funsearch_pipeline.evaluation.procedure2 import _impute_missing_feature_columns
from funsearch_pipeline.evaluation.procedure2 import _load_priority_function
from funsearch_pipeline.evaluation.procedure2 import _validate_priority_signature


DEFAULT_FUNCTION_NAME = "priority"
DEFAULT_SCHEMES = (
    "mixture_learning",
    "independent_learning_scheme",
    "priority_function_curated_context",
)
DEFAULT_PLOT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_RUN_LABEL = "tabular_foundation_models"
RADIUS_CACHE_SCHEMA_VERSION = 1
PREDICTION_COLUMNS = (
    "dataset_pair_name",
    "model_name",
    "scheme_name",
    "heldout_subject_index",
    "heldout_output_pickle_name",
    "heldout_output_pickle_path",
    "heldout_output_row_number",
    "source_pickle_name",
    "source_pickle_path",
    "source_row_number",
    "ancestry_group",
    "actual_label",
    "predicted_label",
    "predicted_disease_probability",
    "predicted_no_disease_probability",
    "correct_label_probability",
)
SCHEME_DISPLAY_NAME = {
    "mixture_learning": "Mixture Learning",
    "independent_learning_scheme": "Independent Learning Scheme",
    "priority_function_curated_context": "Priority Function Curated Context",
}
SCHEME_PLOT_ORDER = {
    "priority_function_curated_context": 0,
    "mixture_learning": 1,
    "independent_learning_scheme": 2,
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_name: str
    training_pickle_paths: tuple[Path, ...]
    calibrating_pickle_paths: tuple[Path, ...]
    heldout_pickle_paths: tuple[Path, ...]
    output_row_tracking_path: Path
    supported_ancestry_groups: tuple[str, ...]
    heldout_target_ancestry_group: str
    label_column: str
    dosage_prefix: str
    ancestry_columns: tuple[str, ...]
    additional_feature_columns: tuple[str, ...]
    priority_radius_cache_path: Path | None
    max_context_rows: int | None
    priority_min_variant_support_fraction: float
    predict_proba_batch_size: int | None
    model_init_kwargs: dict[str, Any]
    schemes: tuple[str, ...]
    plot_ci_level: float | None
    plot_bootstrap_iterations: int


@dataclass(frozen=True)
class RunnerConfig:
    config_path: Path
    prio_function_path: Path
    function_name: str
    output_root_dir: Path
    random_seed: int
    should_overwrite: bool
    experiments: tuple[ExperimentConfig, ...]


@dataclass(frozen=True)
class ModelContextCap:
    rows_cap: int
    source_name: str
    rationale: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RadiusCacheResult:
    manifest_path: Path
    radius_matrix_path: Path
    long_frame_path: Path
    distance_cache_manifest_path: Path
    reference_tracking_path: Path
    heldout_tracking_path: Path
    variant_metadata_path: Path
    radius_matrix: np.ndarray
    long_frame: pd.DataFrame
    reuse_mode: str
    reuse_source_cache_path: Path | None


@dataclass
class PreparedExperiment:
    experiment: ExperimentConfig
    experiment_dir: Path
    training_imputation_counts: dict[str, int]
    calibration_imputation_counts: dict[str, int]
    heldout_imputation_counts: dict[str, int]
    reference_frame: pd.DataFrame
    reference_tracking_rows: pd.DataFrame
    heldout_frame: pd.DataFrame
    heldout_tracking_rows: pd.DataFrame
    variant_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    actual_labels: np.ndarray
    model_cap: ModelContextCap
    effective_cap: int
    radius_cache: RadiusCacheResult


def _priority_function_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _normalize_group_key(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _resolve_target_ancestry_group(
    requested_group: str,
    supported_ancestry_groups: Sequence[str],
) -> str:
    normalized_requested = _normalize_group_key(requested_group)
    matches = [
        candidate
        for candidate in supported_ancestry_groups
        if _normalize_group_key(candidate) == normalized_requested
    ]
    if not matches:
        raise ValueError(
            "heldout_target_ancestry_group did not match supported_ancestry_groups: "
            f"requested={requested_group!r} supported={list(supported_ancestry_groups)!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            "heldout_target_ancestry_group matched multiple supported ancestry groups: "
            f"requested={requested_group!r} matches={matches!r}."
        )
    return matches[0]


def _parse_path_list(base_dir: Path, raw_value: Any, *, field_name: str) -> tuple[Path, ...]:
    if isinstance(raw_value, str):
        raw_values = [raw_value]
    elif isinstance(raw_value, list) and raw_value:
        raw_values = raw_value
    else:
        raise ValueError(f"{field_name} must be a non-empty string or JSON array of strings.")

    resolved_paths: list[Path] = []
    for raw_path in raw_values:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings.")
        path = _resolve_path(base_dir, raw_path)
        if not path.exists():
            raise ValueError(f"{field_name} path does not exist: {path}")
        resolved_paths.append(path)
    return tuple(resolved_paths)


def _parse_supported_ancestry_groups(raw_value: Any) -> tuple[str, ...]:
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("supported_ancestry_groups must be a non-empty JSON array.")
    groups: list[str] = []
    for raw_group in raw_value:
        if not isinstance(raw_group, str) or not raw_group.strip():
            raise ValueError("supported_ancestry_groups entries must be non-empty strings.")
        normalized_group = raw_group.strip().upper()
        if normalized_group not in groups:
            groups.append(normalized_group)
    return tuple(groups)


def _parse_string_list(raw_value: Any, *, field_name: str) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, list):
        raise ValueError(f"{field_name} must be a JSON array of strings.")
    values: list[str] = []
    for raw_item in raw_value:
        if not isinstance(raw_item, str) or not raw_item.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings.")
        values.append(raw_item.strip())
    return tuple(values)


def _parse_optional_positive_int(raw_value: Any, *, field_name: str) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer or null.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive integer or null.")
    return parsed


def _parse_optional_existing_path(base_dir: Path, raw_value: Any, *, field_name: str) -> Path | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{field_name} must be a non-empty string path or null.")
    path = _resolve_path(base_dir, raw_value)
    if not path.exists():
        raise ValueError(f"{field_name} path does not exist: {path}")
    return path


def _parse_optional_probability(raw_value: Any, *, field_name: str) -> float | None:
    if raw_value is None:
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number such as 0.8 or 80, or null.") from exc
    if parsed > 1.0:
        parsed /= 100.0
    if not 0.0 < parsed < 1.0:
        raise ValueError(f"{field_name} must lie between 0 and 1, or between 0 and 100.")
    return parsed


def _parse_support_fraction_threshold(raw_value: Any, *, field_name: str) -> float:
    if raw_value is None:
        return 0.0
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number between 0 and 1, or between 0 and 100.") from exc
    if parsed > 1.0:
        parsed /= 100.0
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must lie between 0 and 1, or between 0 and 100.")
    return parsed


def _normalize_scheme_name(raw_value: Any) -> str:
    if isinstance(raw_value, str):
        key = raw_value.strip().lower()
    elif isinstance(raw_value, dict):
        if not raw_value.get("enabled", True):
            return ""
        key = str(raw_value.get("name") or raw_value.get("type") or "").strip().lower()
    else:
        raise ValueError("schemes entries must be strings or JSON objects.")

    aliases = {
        "mixture_learning": "mixture_learning",
        "mixture random": "mixture_learning",
        "mixture_random": "mixture_learning",
        "same_ancestry_random": "independent_learning_scheme",
        "independent_learning_scheme": "independent_learning_scheme",
        "priority_function_curated_context": "priority_function_curated_context",
        "priority_support_fraction": "priority_function_curated_context",
    }
    normalized = aliases.get(key.replace("-", "_").replace(" ", "_"))
    if normalized is None:
        raise ValueError(f"Unsupported scheme name: {key!r}.")
    return normalized


def _parse_schemes(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return DEFAULT_SCHEMES
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError("schemes must be a non-empty JSON array when provided.")
    schemes: list[str] = []
    for raw_scheme in raw_value:
        normalized = _normalize_scheme_name(raw_scheme)
        if normalized and normalized not in schemes:
            schemes.append(normalized)
    if not schemes:
        raise ValueError("schemes must contain at least one enabled scheme.")
    return tuple(schemes)


def load_config(config_path: str | Path) -> RunnerConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("The config file must contain a JSON object.")

    base_dir = resolved_config_path.parent
    for required_field in ("prio_function_path", "output_root_dir", "experiments"):
        if required_field not in raw_config:
            raise ValueError(f"Missing required config field {required_field!r}.")

    prio_function_path = _resolve_path(base_dir, str(raw_config["prio_function_path"]))
    if not prio_function_path.exists():
        raise ValueError(f"prio_function_path does not exist: {prio_function_path}")

    output_root_dir = _resolve_path(base_dir, str(raw_config["output_root_dir"]))
    raw_experiments = raw_config["experiments"]
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("experiments must be a non-empty JSON array.")

    experiments: list[ExperimentConfig] = []
    for raw_experiment in raw_experiments:
        if not isinstance(raw_experiment, dict):
            raise ValueError("Each experiments entry must be a JSON object.")
        for required_field in (
            "name",
            "model_name",
            "training_pickle_path",
            "calibrating_pickle_path",
            "heldout_pickle_path",
            "output_row_tracking_path",
            "supported_ancestry_groups",
            "heldout_target_ancestry_group",
        ):
            if required_field not in raw_experiment:
                raise ValueError(
                    f"Experiment is missing required field {required_field!r}: {raw_experiment!r}"
                )

        supported_ancestry_groups = _parse_supported_ancestry_groups(
            raw_experiment["supported_ancestry_groups"]
        )
        experiments.append(
            ExperimentConfig(
                name=str(raw_experiment["name"]).strip(),
                model_name=str(raw_experiment["model_name"]).strip().lower(),
                training_pickle_paths=_parse_path_list(
                    base_dir,
                    raw_experiment["training_pickle_path"],
                    field_name="training_pickle_path",
                ),
                calibrating_pickle_paths=_parse_path_list(
                    base_dir,
                    raw_experiment["calibrating_pickle_path"],
                    field_name="calibrating_pickle_path",
                ),
                heldout_pickle_paths=_parse_path_list(
                    base_dir,
                    raw_experiment["heldout_pickle_path"],
                    field_name="heldout_pickle_path",
                ),
                output_row_tracking_path=_resolve_path(
                    base_dir,
                    str(raw_experiment["output_row_tracking_path"]),
                ),
                supported_ancestry_groups=supported_ancestry_groups,
                heldout_target_ancestry_group=_resolve_target_ancestry_group(
                    str(raw_experiment["heldout_target_ancestry_group"]),
                    supported_ancestry_groups,
                ),
                label_column=str(raw_experiment.get("label_column", DEFAULT_LABEL_FIELD)).strip(),
                dosage_prefix=str(raw_experiment.get("dosage_prefix", DOSAGE_COLUMN_PREFIX)).strip(),
                ancestry_columns=tuple(
                    raw_experiment.get("ancestry_columns", list(DEFAULT_ANCESTRY_FIELDS))
                ),
                additional_feature_columns=_parse_string_list(
                    raw_experiment.get("additional_feature_columns", []),
                    field_name="additional_feature_columns",
                ),
                priority_radius_cache_path=_parse_optional_existing_path(
                    base_dir,
                    raw_experiment.get("priority_radius_cache_path"),
                    field_name="priority_radius_cache_path",
                ),
                max_context_rows=_parse_optional_positive_int(
                    raw_experiment.get("max_context_rows"),
                    field_name="max_context_rows",
                ),
                priority_min_variant_support_fraction=_parse_support_fraction_threshold(
                    raw_experiment.get("priority_min_variant_support_fraction"),
                    field_name="priority_min_variant_support_fraction",
                ),
                predict_proba_batch_size=_parse_optional_positive_int(
                    raw_experiment.get("predict_proba_batch_size"),
                    field_name="predict_proba_batch_size",
                ),
                model_init_kwargs=dict(raw_experiment.get("model_init_kwargs", {})),
                schemes=_parse_schemes(raw_experiment.get("schemes")),
                plot_ci_level=_parse_optional_probability(
                    raw_experiment.get("plot_ci_level"),
                    field_name="plot_ci_level",
                ),
                plot_bootstrap_iterations=int(
                    raw_experiment.get(
                        "plot_bootstrap_iterations",
                        DEFAULT_PLOT_BOOTSTRAP_ITERATIONS,
                    )
                ),
            )
        )

    return RunnerConfig(
        config_path=resolved_config_path,
        prio_function_path=prio_function_path,
        function_name=str(raw_config.get("function_name", DEFAULT_FUNCTION_NAME)).strip(),
        output_root_dir=output_root_dir,
        random_seed=int(raw_config.get("random_seed", 0)),
        should_overwrite=bool(raw_config.get("should_overwrite", False)),
        experiments=tuple(experiments),
    )


def _read_dataframe(path: Path) -> pd.DataFrame:
    data = pd.read_pickle(path)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame at {path}, got {type(data)!r}.")
    return data.reset_index(drop=True)


def _load_and_impute_dataframe(paths: Sequence[Path]) -> tuple[pd.DataFrame, tuple[int, ...], dict[str, int]]:
    frames = tuple(_read_dataframe(path) for path in paths)
    combined = _combine_data_objects(frames)
    if not isinstance(combined, pd.DataFrame):
        raise TypeError("Expected concatenated training data to be a pandas DataFrame.")
    imputed, imputed_counts = _impute_missing_feature_columns(combined)
    if not isinstance(imputed, pd.DataFrame):
        raise TypeError("Expected imputed training data to remain a pandas DataFrame.")
    return imputed.reset_index(drop=True), tuple(len(frame) for frame in frames), imputed_counts


def _stable_seed(*parts: object) -> int:
    payload = "\n".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def _safe_slug(value: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    )
    slug = "_".join(part for part in slug.split("_") if part)
    if not slug:
        raise ValueError(f"Could not derive a slug from {value!r}.")
    return slug


def _priority_cache_identity(
    *,
    experiment: ExperimentConfig,
    prio_function_path: Path,
    function_name: str,
) -> tuple[Any, ...]:
    return (
        str(prio_function_path.resolve()),
        _priority_function_sha256(prio_function_path),
        function_name,
        tuple(str(path.resolve()) for path in experiment.training_pickle_paths),
        tuple(str(path.resolve()) for path in experiment.calibrating_pickle_paths),
        tuple(str(path.resolve()) for path in experiment.heldout_pickle_paths),
        str(experiment.output_row_tracking_path.resolve()),
        tuple(experiment.supported_ancestry_groups),
        experiment.heldout_target_ancestry_group,
        tuple(experiment.ancestry_columns),
        experiment.dosage_prefix,
    )


def _create_run_logger(run_dir: Path) -> logging.Logger:
    logger_name = f"ExperimentsWithTabularFoundationModels.{run_dir.name}.{time.time_ns()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "run_events.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def _close_run_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _format_log_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return json.dumps([_format_log_value(item) for item in value])
    return str(value)


def _log_event(logger: logging.Logger | None, event: str, **details: Any) -> None:
    if logger is None:
        return
    fragments = [f"event={event}"]
    for key, value in details.items():
        if value is None:
            continue
        fragments.append(f"{key}={_format_log_value(value)}")
    logger.info(" ".join(fragments))


@contextmanager
def _timed_event(logger: logging.Logger | None, event: str, **details: Any):
    _log_event(logger, f"{event}.started", **details)
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        _log_event(
            logger,
            f"{event}.failed",
            elapsed_seconds=time.perf_counter() - start,
            error_type=type(exc).__name__,
            error=str(exc),
            **details,
        )
        raise
    _log_event(logger, f"{event}.completed", elapsed_seconds=time.perf_counter() - start, **details)


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {description} at {path} to contain a JSON object.")
    return payload


def _normalize_priority_cache_source_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != "manifest.json":
            raise ValueError(
                "priority_radius_cache_path must point to a priority_radius_cache directory "
                "or to its manifest.json file."
            )
        return resolved.parent
    return resolved


def _validate_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"Cached priority radius data mismatch for {label}: actual={actual!r} expected={expected!r}.")


def _link_existing_path(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve():
            return
        if target.is_dir() and not target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing directory with symlink: {target}")
        target.unlink()
    os.symlink(source, target, target_is_directory=source.is_dir())


def _validate_reusable_priority_cache(
    *,
    source_cache_path: Path,
    prio_function_path: Path,
    function_name: str,
    experiment: ExperimentConfig,
    variant_columns: Sequence[str],
    reference_row_count: int,
    heldout_row_count: int,
) -> tuple[dict[str, Any], Path]:
    source_manifest_path = source_cache_path / "manifest.json"
    if not source_manifest_path.exists():
        raise ValueError(f"priority_radius_cache_path is missing manifest.json: {source_cache_path}")

    manifest_payload = _read_json_object(source_manifest_path, description="priority radius cache manifest")
    if int(manifest_payload.get("schema_version", 0)) != RADIUS_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported priority radius cache schema at {source_manifest_path}: "
            f"{manifest_payload.get('schema_version')!r}"
        )

    expected_prio_hash = _priority_function_sha256(prio_function_path)
    cached_prio_hash = str(manifest_payload.get("priority_function_sha256") or "").strip()
    if cached_prio_hash:
        _validate_equal("priority_function_sha256", cached_prio_hash, expected_prio_hash)
    else:
        cached_priority_path = Path(str(manifest_payload.get("priority_function_path", ""))).expanduser()
        if not cached_priority_path.exists():
            raise ValueError(
                "priority_radius_cache_path manifest does not record a reusable priority function file: "
                f"{cached_priority_path}"
            )
        _validate_equal(
            "priority_function_sha256",
            _priority_function_sha256(cached_priority_path),
            expected_prio_hash,
        )

    _validate_equal("function_name", str(manifest_payload.get("function_name")), function_name)
    _validate_equal(
        "training_pickle_paths",
        list(manifest_payload.get("training_pickle_paths", [])),
        [str(path.resolve()) for path in experiment.training_pickle_paths],
    )
    _validate_equal(
        "calibrating_pickle_paths",
        list(manifest_payload.get("calibrating_pickle_paths", [])),
        [str(path.resolve()) for path in experiment.calibrating_pickle_paths],
    )
    _validate_equal(
        "heldout_pickle_paths",
        list(manifest_payload.get("heldout_pickle_paths", [])),
        [str(path.resolve()) for path in experiment.heldout_pickle_paths],
    )
    _validate_equal(
        "output_row_tracking_path",
        str(manifest_payload.get("output_row_tracking_path")),
        str(experiment.output_row_tracking_path.resolve()),
    )
    _validate_equal(
        "supported_ancestry_groups",
        list(manifest_payload.get("supported_ancestry_groups", [])),
        list(experiment.supported_ancestry_groups),
    )
    _validate_equal(
        "heldout_target_ancestry_group",
        str(manifest_payload.get("heldout_target_ancestry_group")),
        experiment.heldout_target_ancestry_group,
    )
    _validate_equal(
        "ancestry_columns",
        list(manifest_payload.get("ancestry_columns", [])),
        list(experiment.ancestry_columns),
    )
    _validate_equal(
        "variant_dosage_fields",
        list(manifest_payload.get("variant_dosage_fields", [])),
        list(variant_columns),
    )
    _validate_equal("reference_row_count", int(manifest_payload.get("reference_row_count", -1)), reference_row_count)
    _validate_equal(
        "heldout_target_row_count",
        int(manifest_payload.get("heldout_target_row_count", -1)),
        heldout_row_count,
    )

    distance_cache_manifest_path = Path(str(manifest_payload.get("distance_cache_manifest_path", ""))).expanduser()
    if not distance_cache_manifest_path.exists():
        raise ValueError(
            "priority_radius_cache_path manifest references a missing distance cache manifest: "
            f"{distance_cache_manifest_path}"
        )
    distance_manifest_payload = _read_json_object(
        distance_cache_manifest_path,
        description="ancestry distance cache manifest",
    )
    _validate_equal(
        "distance_cache.reference_source_paths",
        list(distance_manifest_payload.get("reference_source_paths", [])),
        [str(path.resolve()) for path in experiment.training_pickle_paths + experiment.calibrating_pickle_paths],
    )
    _validate_equal(
        "distance_cache.target_source_paths",
        list(distance_manifest_payload.get("target_source_paths", [])),
        [str(path.resolve()) for path in experiment.heldout_pickle_paths],
    )
    _validate_equal(
        "distance_cache.reference_row_count",
        int(distance_manifest_payload.get("reference_row_count", -1)),
        reference_row_count,
    )
    _validate_equal(
        "distance_cache.target_row_count",
        int(distance_manifest_payload.get("target_row_count", -1)),
        heldout_row_count,
    )
    _validate_equal(
        "distance_cache.ancestry_dimension",
        int(distance_manifest_payload.get("ancestry_dimension", -1)),
        len(experiment.ancestry_columns),
    )
    if load_distance_cache(distance_cache_manifest_path) is None:
        raise ValueError(f"priority_radius_cache_path references an unreadable distance cache: {distance_cache_manifest_path}")

    return manifest_payload, distance_cache_manifest_path


def _stage_priority_cache_links(
    *,
    cache_dir: Path,
    source_cache_path: Path,
    manifest_payload: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _link_existing_path(source_cache_path / "manifest.json", cache_dir / "manifest.json")
    for field_name, file_name in (
        ("radius_matrix_path", "radius_matrix.npy"),
        ("radii_by_subject_variant_path", "radii_by_subject_variant.pkl"),
        ("reference_tracking_rows_path", "reference_tracking_rows.pkl"),
        ("heldout_tracking_rows_path", "heldout_tracking_rows.pkl"),
        ("variant_metadata_path", "variant_metadata.json"),
    ):
        source_path = Path(str(manifest_payload[field_name])).expanduser().resolve()
        if not source_path.exists():
            raise ValueError(f"Reusable priority radius cache artifact is missing: {source_path}")
        _link_existing_path(source_path, cache_dir / file_name)

    source_distance_cache_dir = Path(str(manifest_payload["distance_cache_manifest_path"])).expanduser().resolve().parent
    if not source_distance_cache_dir.exists():
        raise ValueError(f"Reusable distance cache directory is missing: {source_distance_cache_dir}")
    _link_existing_path(source_distance_cache_dir, cache_dir / "distance_cache")


def _stage_distance_cache_link(
    *,
    cache_dir: Path,
    distance_cache_manifest_path: Path,
) -> Path:
    source_distance_cache_dir = distance_cache_manifest_path.expanduser().resolve().parent
    if not source_distance_cache_dir.exists():
        raise ValueError(f"Reusable distance cache directory is missing: {source_distance_cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _link_existing_path(source_distance_cache_dir, cache_dir / "distance_cache")
    return cache_dir / "distance_cache" / distance_cache_manifest_path.name


def _try_link_reusable_priority_radius_cache(
    *,
    experiment: ExperimentConfig,
    experiment_dir: Path,
    prio_function_path: Path,
    function_name: str,
    variant_columns: Sequence[str],
    reference_row_count: int,
    heldout_row_count: int,
    event_logger: logging.Logger | None,
) -> RadiusCacheResult | None:
    if experiment.priority_radius_cache_path is None:
        return None

    source_cache_path = _normalize_priority_cache_source_path(experiment.priority_radius_cache_path)
    manifest_payload, _ = _validate_reusable_priority_cache(
        source_cache_path=source_cache_path,
        prio_function_path=prio_function_path,
        function_name=function_name,
        experiment=experiment,
        variant_columns=variant_columns,
        reference_row_count=reference_row_count,
        heldout_row_count=heldout_row_count,
    )

    required_fields = (
        "radius_matrix_path",
        "radii_by_subject_variant_path",
        "reference_tracking_rows_path",
        "heldout_tracking_rows_path",
        "variant_metadata_path",
    )
    is_complete = all(Path(str(manifest_payload.get(field_name, ""))).expanduser().exists() for field_name in required_fields)
    if not is_complete:
        _log_event(
            event_logger,
            "priority_cache.partial_reuse_requested",
            experiment_name=experiment.name,
            source_cache_path=source_cache_path,
        )
        return None

    cache_dir = experiment_dir / "priority_radius_cache"
    _stage_priority_cache_links(
        cache_dir=cache_dir,
        source_cache_path=source_cache_path,
        manifest_payload=manifest_payload,
    )
    _log_event(
        event_logger,
        "priority_cache.reused_complete_cache",
        experiment_name=experiment.name,
        source_cache_path=source_cache_path,
    )

    radius_matrix_path = cache_dir / "radius_matrix.npy"
    long_frame_path = cache_dir / "radii_by_subject_variant.pkl"
    reference_tracking_path = cache_dir / "reference_tracking_rows.pkl"
    heldout_tracking_path = cache_dir / "heldout_tracking_rows.pkl"
    variant_metadata_path = cache_dir / "variant_metadata.json"
    manifest_path = cache_dir / "manifest.json"
    distance_cache_manifest_path = cache_dir / "distance_cache" / Path(
        str(manifest_payload["distance_cache_manifest_path"])
    ).name
    return RadiusCacheResult(
        manifest_path=manifest_path,
        radius_matrix_path=radius_matrix_path,
        long_frame_path=long_frame_path,
        distance_cache_manifest_path=distance_cache_manifest_path,
        reference_tracking_path=reference_tracking_path,
        heldout_tracking_path=heldout_tracking_path,
        variant_metadata_path=variant_metadata_path,
        radius_matrix=np.load(radius_matrix_path, mmap_mode="r"),
        long_frame=pd.read_pickle(long_frame_path),
        reuse_mode="linked_complete_cache",
        reuse_source_cache_path=source_cache_path,
    )


def _prepare_distance_cache(
    *,
    experiment: ExperimentConfig,
    cache_dir: Path,
    reference_frame: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    prio_function_path: Path,
    function_name: str,
    variant_columns: Sequence[str],
    event_logger: logging.Logger | None,
) -> tuple[OpenedDistanceCache, Path, str, Path | None]:
    local_distance_cache_dir = cache_dir / "distance_cache"
    if experiment.priority_radius_cache_path is not None:
        source_cache_path = _normalize_priority_cache_source_path(experiment.priority_radius_cache_path)
        manifest_payload, distance_cache_manifest_path = _validate_reusable_priority_cache(
            source_cache_path=source_cache_path,
            prio_function_path=prio_function_path,
            function_name=function_name,
            experiment=experiment,
            variant_columns=variant_columns,
            reference_row_count=int(reference_frame.shape[0]),
            heldout_row_count=int(heldout_frame.shape[0]),
        )
        local_distance_cache_manifest_path = _stage_distance_cache_link(
            cache_dir=cache_dir,
            distance_cache_manifest_path=distance_cache_manifest_path,
        )
        opened_distance_cache = load_distance_cache(local_distance_cache_manifest_path)
        if opened_distance_cache is None:
            raise ValueError("Could not open the reused distance cache after linking it into the new run directory.")
        _log_event(
            event_logger,
            "priority_cache.reused_distance_cache",
            experiment_name=experiment.name,
            source_cache_path=source_cache_path,
        )
        return (
            opened_distance_cache,
            local_distance_cache_manifest_path,
            "linked_distance_cache_then_recomputed_radius_cache",
            source_cache_path,
        )

    distance_cache_artifacts = ensure_distance_cache(
        reference_data=reference_frame,
        target_data=heldout_frame,
        cache_root=local_distance_cache_dir,
        cache_name=f"{experiment.name}.heldout_to_reference",
        reference_source_paths=[
            str(path)
            for path in experiment.training_pickle_paths + experiment.calibrating_pickle_paths
        ],
        target_source_paths=[str(path) for path in experiment.heldout_pickle_paths],
    )
    if distance_cache_artifacts is None:
        raise ValueError("Could not create a distance cache for the reference and heldout data.")
    opened_distance_cache = load_distance_cache(distance_cache_artifacts.manifest_path)
    if opened_distance_cache is None:
        raise ValueError("Could not open the distance cache after creating it.")
    _log_event(event_logger, "priority_cache.computed_distance_cache", experiment_name=experiment.name)
    return (
        opened_distance_cache,
        Path(distance_cache_artifacts.manifest_path),
        "recomputed",
        None,
    )


def _create_run_directory(config: RunnerConfig) -> Path:
    config.output_root_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root_dir / f"{timestamp}_{DEFAULT_RUN_LABEL}"
    if run_dir.exists():
        if not config.should_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _extract_binary_labels(frame: pd.DataFrame, *, label_column: str) -> np.ndarray:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column {label_column!r}.")
    labels = frame[label_column].to_numpy(dtype=float, copy=True)
    unique_labels = sorted(set(float(value) for value in labels.tolist()))
    if unique_labels != [0.0, 1.0]:
        raise ValueError(
            f"Expected binary labels encoded as 0/1 in column {label_column!r}, got {unique_labels!r}."
        )
    return labels.astype(np.int64, copy=False)


def _resolve_feature_columns(
    reference_frame: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    *,
    dosage_prefix: str,
    ancestry_columns: Sequence[str],
    additional_feature_columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    variant_columns = tuple(
        str(column_name)
        for column_name in reference_frame.columns
        if str(column_name).startswith(dosage_prefix)
    )
    if not variant_columns:
        raise ValueError(f"No dosage columns with prefix {dosage_prefix!r} were found.")

    feature_columns = variant_columns + tuple(ancestry_columns) + tuple(additional_feature_columns)
    missing_reference = [column for column in feature_columns if column not in reference_frame.columns]
    missing_heldout = [column for column in feature_columns if column not in heldout_frame.columns]
    if missing_reference:
        raise ValueError(f"Reference data is missing feature columns: {missing_reference!r}")
    if missing_heldout:
        raise ValueError(f"Heldout data is missing feature columns: {missing_heldout!r}")
    return variant_columns, feature_columns


def _derive_model_context_cap(
    *,
    model_name: str,
    feature_count: int,
    reference_row_count: int,
) -> ModelContextCap:
    warnings: list[str] = []
    if model_name == "tabpfn":
        if feature_count <= 200:
            rows_cap = 1_000_000
        elif feature_count <= 2_000:
            rows_cap = 100_000
        elif feature_count <= 20_000:
            rows_cap = 1_000
        else:
            raise ValueError(
                "TabPFN documentation describes the default model within these row/feature operating points: "
                "1,000,000x200, 100,000x2,000, or 1,000x20,000. "
                f"This experiment has {feature_count} features, which is outside that documented range."
            )
        return ModelContextCap(
            rows_cap=rows_cap,
            source_name="TabPFN README usage tips",
            rationale=(
                "Auto cap uses the documented TabPFN-3 row-by-feature support point that covers "
                f"{feature_count} features."
            ),
            warnings=(),
        )

    if model_name == "tabicl":
        if feature_count > 2_000:
            raise ValueError(
                "TabICL README states support up to 2,000 features; "
                f"this experiment has {feature_count} features."
            )
        if feature_count < 2 or feature_count > 100:
            warnings.append(
                "TabICL FAQ states the model was pre-trained on datasets with 2 to 100 columns. "
                f"This experiment has {feature_count} features."
            )
        if reference_row_count < 300 or reference_row_count > 48_000:
            warnings.append(
                "TabICL FAQ states the model was pre-trained on datasets with 300 to 48K training samples. "
                f"This experiment has {reference_row_count} reference rows."
            )
        return ModelContextCap(
            rows_cap=48_000,
            source_name="TabICL README FAQ pre-training range",
            rationale=(
                "Auto cap uses the explicit TabICL pre-training range upper bound of 48K training rows "
                "to avoid exceeding the documented training regime."
            ),
            warnings=tuple(warnings),
        )

    raise ValueError(f"Unsupported model_name {model_name!r}; expected 'tabpfn' or 'tabicl'.")


def _effective_context_cap(
    *,
    requested_cap: int | None,
    model_cap: ModelContextCap,
) -> int:
    if requested_cap is None:
        return model_cap.rows_cap
    return min(requested_cap, model_cap.rows_cap)


def _build_model(
    *,
    model_name: str,
    model_init_kwargs: dict[str, Any],
):
    if model_name == "tabpfn":
        from tabpfn import TabPFNClassifier

        kwargs = {"show_progress_bar": False, **model_init_kwargs}
        return TabPFNClassifier(**kwargs)
    if model_name == "tabicl":
        from tabicl import TabICLClassifier

        return TabICLClassifier(**model_init_kwargs)
    raise ValueError(f"Unsupported model_name {model_name!r}.")


def _predict_positive_probability(
    estimator: Any,
    X_target: pd.DataFrame,
    *,
    batch_size: int | None,
) -> np.ndarray:
    classes = np.asarray(getattr(estimator, "classes_", ()), dtype=float)
    if classes.shape[0] != 2 or sorted(classes.tolist()) != [0.0, 1.0]:
        raise ValueError(f"Expected binary estimator classes [0, 1], got {classes.tolist()!r}.")
    positive_index = int(np.where(classes == 1.0)[0][0])

    if batch_size is None or len(X_target) <= batch_size:
        return np.asarray(estimator.predict_proba(X_target)[:, positive_index], dtype=float)

    probability_chunks: list[np.ndarray] = []
    for start in range(0, len(X_target), batch_size):
        stop = min(len(X_target), start + batch_size)
        probability_chunks.append(
            np.asarray(estimator.predict_proba(X_target.iloc[start:stop])[:, positive_index], dtype=float)
        )
    return np.concatenate(probability_chunks, axis=0)


def _rows_from_tracking(
    frame: pd.DataFrame,
    tracking_rows: pd.DataFrame,
) -> pd.DataFrame:
    dataset_row_indices = tracking_rows["dataset_row_index"].to_numpy(dtype=int, copy=False)
    return frame.iloc[dataset_row_indices].reset_index(drop=True)


def _bootstrap_ci_if_requested(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    ci_level: float | None,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if ci_level is None:
        return None, None
    lower, upper = _bootstrap_auc_confidence_interval(
        labels=labels.astype(float, copy=False),
        risk_scores=probabilities.astype(float, copy=False),
        ci_level=ci_level,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=seed,
    )
    return float(lower), float(upper)


def _select_random_context_indices(
    eligible_indices: np.ndarray,
    *,
    desired_count: int,
    seed: int,
) -> np.ndarray:
    if desired_count >= eligible_indices.shape[0]:
        return np.asarray(eligible_indices, dtype=int)
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible_indices, size=desired_count, replace=False)
    return np.sort(selected.astype(int, copy=False))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reference_tracking_rows(
    *,
    training_pickle_paths: tuple[Path, ...],
    training_lengths: tuple[int, ...],
    calibrating_pickle_paths: tuple[Path, ...],
    calibrating_lengths: tuple[int, ...],
    output_row_tracking_path: Path,
    supported_ancestry_groups: tuple[str, ...],
) -> pd.DataFrame:
    return _build_dataset_tracking_rows(
        dataset_pickle_paths=training_pickle_paths + calibrating_pickle_paths,
        dataset_lengths=training_lengths + calibrating_lengths,
        output_row_tracking_path=output_row_tracking_path,
        supported_ancestry_groups=supported_ancestry_groups,
    ).reset_index(drop=True)


def _heldout_tracking_rows(
    *,
    heldout_pickle_paths: tuple[Path, ...],
    heldout_lengths: tuple[int, ...],
    output_row_tracking_path: Path,
    supported_ancestry_groups: tuple[str, ...],
) -> pd.DataFrame:
    return _build_dataset_tracking_rows(
        dataset_pickle_paths=heldout_pickle_paths,
        dataset_lengths=heldout_lengths,
        output_row_tracking_path=output_row_tracking_path,
        supported_ancestry_groups=supported_ancestry_groups,
    ).reset_index(drop=True)


def _filter_tracking_rows_for_target_ancestry(
    tracking_rows: pd.DataFrame,
    *,
    target_ancestry_group: str,
) -> pd.DataFrame:
    target_key = _normalize_group_key(target_ancestry_group)
    mask = tracking_rows["ancestry_group"].astype(str).map(_normalize_group_key) == target_key
    filtered = tracking_rows.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"No heldout rows matched target ancestry group {target_ancestry_group!r}."
        )
    filtered["target_subject_local_index"] = np.arange(filtered.shape[0], dtype=int)
    return filtered


def _build_priority_radius_cache(
    *,
    experiment: ExperimentConfig,
    experiment_dir: Path,
    prio_function_path: Path,
    function_name: str,
    reference_frame: pd.DataFrame,
    reference_tracking_rows: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    heldout_tracking_rows: pd.DataFrame,
    variant_columns: Sequence[str],
    event_logger: logging.Logger | None = None,
) -> RadiusCacheResult:
    cache_dir = experiment_dir / "priority_radius_cache"
    reusable_cache = _try_link_reusable_priority_radius_cache(
        experiment=experiment,
        experiment_dir=experiment_dir,
        prio_function_path=prio_function_path,
        function_name=function_name,
        variant_columns=variant_columns,
        reference_row_count=int(reference_frame.shape[0]),
        heldout_row_count=int(heldout_frame.shape[0]),
        event_logger=event_logger,
    )
    if reusable_cache is not None:
        return reusable_cache

    cache_dir.mkdir(parents=True, exist_ok=True)

    program_source = prio_function_path.read_text(encoding="utf-8")
    priority_function = _load_priority_function(program_source, function_name)
    _validate_priority_signature(priority_function)

    priority_training_data = _build_priority_training_data_contract(reference_frame, variant_columns)
    priority_target_variants = _build_priority_target_variants(variant_columns)

    opened_distance_cache, distance_cache_manifest_path, reuse_mode, reuse_source_cache_path = _prepare_distance_cache(
        experiment=experiment,
        cache_dir=cache_dir,
        reference_frame=reference_frame,
        heldout_frame=heldout_frame,
        prio_function_path=prio_function_path,
        function_name=function_name,
        variant_columns=variant_columns,
        event_logger=event_logger,
    )

    ancestry_matrix = heldout_frame.loc[:, list(experiment.ancestry_columns)].to_numpy(dtype=float)
    radius_matrix = np.zeros((heldout_frame.shape[0], len(priority_target_variants)), dtype=np.float64)
    radius_rows: list[dict[str, Any]] = []

    logical_variant_names = [
        target_variant.name for target_variant in priority_target_variants
    ]
    for target_index in range(heldout_frame.shape[0]):
        ancestry_coordinate = _build_priority_ancestry_coordinate(ancestry_matrix[target_index])
        row_view = opened_distance_cache.row_view(target_index)
        with activate_distance_context(
            raw_training_data=reference_frame,
            priority_training_data=priority_training_data,
            target_ancestry_values=ancestry_matrix[target_index],
            target_distance_view=row_view,
            novelty_baseline_median=opened_distance_cache.artifacts.novelty_baseline_median,
        ):
            for variant_index, target_variant in enumerate(priority_target_variants):
                radius = _call_priority_function(
                    priority_function,
                    priority_training_data,
                    ancestry_coordinate,
                    target_variant,
                )
                radius_matrix[target_index, variant_index] = radius
                radius_rows.append(
                    {
                        "dataset_pair_name": experiment.name,
                        "heldout_subject_index": int(heldout_tracking_rows.iloc[target_index]["target_subject_local_index"]),
                        "heldout_output_pickle_name": heldout_tracking_rows.iloc[target_index]["dataset_pickle_name"],
                        "heldout_output_row_number": int(heldout_tracking_rows.iloc[target_index]["dataset_row_number"]),
                        "ancestry_group": heldout_tracking_rows.iloc[target_index]["ancestry_group"],
                        "variant_index": variant_index,
                        "variant_name": target_variant.name,
                        "dosage_field": target_variant.dosage_field,
                        "radius": float(radius),
                        "reference_cohort_size": int(reference_frame.shape[0]),
                    }
                )

    radius_matrix_path = cache_dir / "radius_matrix.npy"
    np.save(radius_matrix_path, radius_matrix)
    long_frame = pd.DataFrame(radius_rows)
    long_frame_path = cache_dir / "radii_by_subject_variant.pkl"
    long_frame.to_pickle(long_frame_path)

    reference_tracking_path = cache_dir / "reference_tracking_rows.pkl"
    heldout_tracking_path = cache_dir / "heldout_tracking_rows.pkl"
    reference_tracking_rows.to_pickle(reference_tracking_path)
    heldout_tracking_rows.to_pickle(heldout_tracking_path)

    variant_metadata_path = cache_dir / "variant_metadata.json"
    _write_json(
        variant_metadata_path,
        {
            "variant_names": list(logical_variant_names),
            "variant_dosage_fields": list(variant_columns),
        },
    )

    manifest_path = cache_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": RADIUS_CACHE_SCHEMA_VERSION,
            "dataset_pair_name": experiment.name,
            "priority_function_path": str(prio_function_path.resolve()),
            "priority_function_sha256": _priority_function_sha256(prio_function_path),
            "function_name": function_name,
            "training_pickle_paths": [str(path.resolve()) for path in experiment.training_pickle_paths],
            "calibrating_pickle_paths": [str(path.resolve()) for path in experiment.calibrating_pickle_paths],
            "heldout_pickle_paths": [str(path.resolve()) for path in experiment.heldout_pickle_paths],
            "output_row_tracking_path": str(experiment.output_row_tracking_path.resolve()),
            "distance_cache_manifest_path": str(distance_cache_manifest_path.resolve()),
            "supported_ancestry_groups": list(experiment.supported_ancestry_groups),
            "heldout_target_ancestry_group": experiment.heldout_target_ancestry_group,
            "ancestry_columns": list(experiment.ancestry_columns),
            "variant_dosage_fields": list(variant_columns),
            "reference_row_count": int(reference_frame.shape[0]),
            "heldout_target_row_count": int(heldout_frame.shape[0]),
            "radius_matrix_path": str(radius_matrix_path.resolve()),
            "radii_by_subject_variant_path": str(long_frame_path.resolve()),
            "reference_tracking_rows_path": str(reference_tracking_path.resolve()),
            "heldout_tracking_rows_path": str(heldout_tracking_path.resolve()),
            "variant_metadata_path": str(variant_metadata_path.resolve()),
        },
    )

    return RadiusCacheResult(
        manifest_path=manifest_path,
        radius_matrix_path=radius_matrix_path,
        long_frame_path=long_frame_path,
        distance_cache_manifest_path=distance_cache_manifest_path,
        reference_tracking_path=reference_tracking_path,
        heldout_tracking_path=heldout_tracking_path,
        variant_metadata_path=variant_metadata_path,
        radius_matrix=radius_matrix,
        long_frame=long_frame,
        reuse_mode=reuse_mode,
        reuse_source_cache_path=reuse_source_cache_path,
    )


def _build_priority_selection_for_target(
    *,
    target_index: int,
    radius_matrix: np.ndarray,
    distance_cache_manifest_path: Path,
    reference_tracking_rows: pd.DataFrame,
    context_cap: int,
    min_support_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    opened_distance_cache = load_distance_cache(distance_cache_manifest_path)
    if opened_distance_cache is None:
        raise ValueError("Could not load priority distance cache for selection.")
    row_view = opened_distance_cache.row_view(target_index)
    distances = np.asarray(row_view.distances, dtype=float)
    radii = np.asarray(radius_matrix[target_index], dtype=float)
    if radii.size == 0:
        raise ValueError("Priority radius matrix is empty.")

    sorted_radii = np.sort(radii)
    counts_ge_distance = radii.size - np.searchsorted(sorted_radii, distances, side="left")
    support_fraction = counts_ge_distance.astype(np.float64) / float(radii.size)

    positive_support_indices = np.flatnonzero(support_fraction > min_support_fraction)
    candidate_indices = positive_support_indices
    if candidate_indices.size == 0:
        candidate_indices = np.array([int(np.argmin(distances))], dtype=int)

    candidate_support = support_fraction[candidate_indices]
    candidate_distances = distances[candidate_indices]
    candidate_row_indices = reference_tracking_rows.iloc[candidate_indices]["dataset_row_index"].to_numpy(
        dtype=int,
        copy=False,
    )
    rank_order = np.lexsort((candidate_row_indices, candidate_distances, -candidate_support))
    selected_candidate_indices = candidate_indices[rank_order]
    if selected_candidate_indices.shape[0] > context_cap:
        selected_candidate_indices = selected_candidate_indices[:context_cap]
    return selected_candidate_indices.astype(int, copy=False), support_fraction, distances


def _build_prediction_frame(
    *,
    dataset_pair_name: str,
    model_name: str,
    scheme_name: str,
    heldout_tracking_rows: pd.DataFrame,
    actual_labels: np.ndarray,
    disease_probabilities: np.ndarray,
) -> pd.DataFrame:
    predicted_labels = (disease_probabilities >= 0.5).astype(np.int64)
    no_disease_probabilities = 1.0 - disease_probabilities
    correct_label_probabilities = np.where(actual_labels == 1, disease_probabilities, no_disease_probabilities)
    prediction_frame = pd.DataFrame(
        {
            "dataset_pair_name": dataset_pair_name,
            "model_name": model_name,
            "scheme_name": scheme_name,
            "heldout_subject_index": heldout_tracking_rows["target_subject_local_index"].to_numpy(copy=False),
            "heldout_output_pickle_name": heldout_tracking_rows["dataset_pickle_name"].to_numpy(copy=False),
            "heldout_output_pickle_path": heldout_tracking_rows["dataset_pickle_path"].to_numpy(copy=False),
            "heldout_output_row_number": heldout_tracking_rows["dataset_row_number"].to_numpy(copy=False),
            "source_pickle_name": heldout_tracking_rows["source_pickle_name"].to_numpy(copy=False),
            "source_pickle_path": heldout_tracking_rows["source_pickle_path"].to_numpy(copy=False),
            "source_row_number": heldout_tracking_rows["source_row_number"].to_numpy(copy=False),
            "ancestry_group": heldout_tracking_rows["ancestry_group"].to_numpy(copy=False),
            "actual_label": actual_labels.astype(np.int64, copy=False),
            "predicted_label": predicted_labels,
            "predicted_disease_probability": disease_probabilities,
            "predicted_no_disease_probability": no_disease_probabilities,
            "correct_label_probability": correct_label_probabilities,
        }
    )
    return prediction_frame.loc[:, list(PREDICTION_COLUMNS)]


def _plot_experiment_metrics(
    *,
    metric_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    subtitle: str | None,
) -> None:
    plot_frame = metric_frame.sort_values("plot_order").reset_index(drop=True)
    figure_height = max(3.5, 1.2 * plot_frame.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(11.0, figure_height))

    y_positions = np.arange(plot_frame.shape[0], dtype=float)
    colors = [
        "#1f4e79" if name == "priority_function_curated_context" else "#9c8f7a" if name == "mixture_learning" else "#6c757d"
        for name in plot_frame["scheme_name"].tolist()
    ]
    ax.barh(
        y_positions,
        plot_frame["auc_roc"].to_numpy(dtype=float),
        color=colors,
        edgecolor="#222222",
        height=0.68,
    )

    lower_errors = []
    upper_errors = []
    has_any_ci = False
    for _, row in plot_frame.iterrows():
        ci_lower = row["ci_lower"]
        ci_upper = row["ci_upper"]
        if pd.isna(ci_lower) or pd.isna(ci_upper):
            lower_errors.append(0.0)
            upper_errors.append(0.0)
            continue
        has_any_ci = True
        auc_roc = float(row["auc_roc"])
        lower_errors.append(max(0.0, auc_roc - float(ci_lower)))
        upper_errors.append(max(0.0, float(ci_upper) - auc_roc))
    if has_any_ci:
        ax.errorbar(
            plot_frame["auc_roc"].to_numpy(dtype=float),
            y_positions,
            xerr=np.vstack([np.asarray(lower_errors), np.asarray(upper_errors)]),
            fmt="none",
            ecolor="#7a1f1f",
            elinewidth=2.0,
            capsize=5.0,
        )

    for index, row in plot_frame.iterrows():
        x_value = float(row["auc_roc"])
        ax.text(
            x_value + 0.01,
            float(index),
            f"{x_value:.3f}",
            va="center",
            ha="left",
            fontsize=11,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [SCHEME_DISPLAY_NAME.get(name, name) for name in plot_frame["scheme_name"].tolist()],
        fontsize=12,
    )
    ax.set_xlabel("ROC AUC", fontsize=13)
    ax.set_ylabel("Scheme", fontsize=13)
    ax.set_title(title, fontsize=15, loc="left", pad=12)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _run_shared_context_scheme(
    *,
    dataset_pair_name: str,
    model_name: str,
    scheme_name: str,
    experiment: ExperimentConfig,
    experiment_dir: Path,
    reference_frame: pd.DataFrame,
    reference_tracking_rows: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    heldout_tracking_rows: pd.DataFrame,
    feature_columns: Sequence[str],
    actual_labels: np.ndarray,
    selected_reference_indices: np.ndarray,
    ci_level: float | None,
    event_logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    with _timed_event(
        event_logger,
        "scheme",
        experiment_name=dataset_pair_name,
        model_name=model_name,
        scheme_name=scheme_name,
        context_row_count=int(selected_reference_indices.shape[0]),
        heldout_subject_count=int(heldout_tracking_rows.shape[0]),
    ):
        context_frame = reference_frame.iloc[selected_reference_indices].reset_index(drop=True)
        context_labels = _extract_binary_labels(context_frame, label_column=experiment.label_column)

        estimator = _build_model(model_name=model_name, model_init_kwargs=experiment.model_init_kwargs)
        fit_start = time.perf_counter()
        estimator.fit(context_frame.loc[:, list(feature_columns)], context_labels)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        disease_probabilities = _predict_positive_probability(
            estimator,
            heldout_frame.loc[:, list(feature_columns)],
            batch_size=experiment.predict_proba_batch_size,
        )
        predict_seconds = time.perf_counter() - predict_start

        auc_roc = float(roc_auc_score(actual_labels, disease_probabilities))
        ci_lower, ci_upper = _bootstrap_ci_if_requested(
            labels=actual_labels,
            probabilities=disease_probabilities,
            ci_level=ci_level,
            bootstrap_iterations=experiment.plot_bootstrap_iterations,
            seed=_stable_seed(dataset_pair_name, model_name, scheme_name, "ci"),
        )

        prediction_frame = _build_prediction_frame(
            dataset_pair_name=dataset_pair_name,
            model_name=model_name,
            scheme_name=scheme_name,
            heldout_tracking_rows=heldout_tracking_rows,
            actual_labels=actual_labels,
            disease_probabilities=disease_probabilities,
        )

        applies_to_subject_count = heldout_tracking_rows.shape[0]
        audit_frame = reference_tracking_rows.iloc[selected_reference_indices].copy().reset_index(drop=True)
        audit_frame.insert(0, "dataset_pair_name", dataset_pair_name)
        audit_frame.insert(1, "model_name", model_name)
        audit_frame.insert(2, "scheme_name", scheme_name)
        audit_frame.insert(3, "heldout_subject_index", pd.NA)
        audit_frame["target_scope"] = "all_filtered_heldout_subjects"
        audit_frame["applies_to_subject_count"] = applies_to_subject_count
        audit_frame["selection_rank"] = np.arange(audit_frame.shape[0], dtype=int)
        audit_frame["ancestry_distance"] = np.nan
        audit_frame["priority_support_fraction"] = np.nan

        metrics = {
            "dataset_pair_name": dataset_pair_name,
            "model_name": model_name,
            "scheme_name": scheme_name,
            "auc_roc": auc_roc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "plot_order": SCHEME_PLOT_ORDER[scheme_name],
            "subject_count": int(actual_labels.shape[0]),
            "context_row_count": int(context_frame.shape[0]),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "experiment_dir": str(experiment_dir),
        }
    return prediction_frame, audit_frame, metrics


def _run_priority_context_scheme(
    *,
    dataset_pair_name: str,
    model_name: str,
    experiment: ExperimentConfig,
    experiment_dir: Path,
    reference_frame: pd.DataFrame,
    reference_tracking_rows: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    heldout_tracking_rows: pd.DataFrame,
    feature_columns: Sequence[str],
    actual_labels: np.ndarray,
    radius_cache: RadiusCacheResult,
    context_cap: int,
    ci_level: float | None,
    event_logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    with _timed_event(
        event_logger,
        "scheme",
        experiment_name=dataset_pair_name,
        model_name=model_name,
        scheme_name="priority_function_curated_context",
        heldout_subject_count=int(heldout_tracking_rows.shape[0]),
        context_cap=int(context_cap),
        cache_reuse_mode=radius_cache.reuse_mode,
    ):
        per_subject_probabilities: list[float] = []
        audit_frames: list[pd.DataFrame] = []
        fit_seconds_total = 0.0
        predict_seconds_total = 0.0
        context_sizes: list[int] = []

        for target_index in range(heldout_frame.shape[0]):
            selected_indices, support_fraction, distances = _build_priority_selection_for_target(
                target_index=target_index,
                radius_matrix=radius_cache.radius_matrix,
                distance_cache_manifest_path=radius_cache.distance_cache_manifest_path,
                reference_tracking_rows=reference_tracking_rows,
                context_cap=context_cap,
                min_support_fraction=experiment.priority_min_variant_support_fraction,
            )
            context_sizes.append(int(selected_indices.shape[0]))
            context_frame = reference_frame.iloc[selected_indices].reset_index(drop=True)
            context_labels = _extract_binary_labels(context_frame, label_column=experiment.label_column)

            estimator = _build_model(model_name=model_name, model_init_kwargs=experiment.model_init_kwargs)
            fit_start = time.perf_counter()
            estimator.fit(context_frame.loc[:, list(feature_columns)], context_labels)
            fit_seconds_total += time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            probability = float(
                _predict_positive_probability(
                    estimator,
                    heldout_frame.iloc[[target_index]].loc[:, list(feature_columns)],
                    batch_size=None,
                )[0]
            )
            predict_seconds_total += time.perf_counter() - predict_start
            per_subject_probabilities.append(probability)

            selected_tracking = reference_tracking_rows.iloc[selected_indices].copy().reset_index(drop=True)
            selected_tracking.insert(0, "dataset_pair_name", dataset_pair_name)
            selected_tracking.insert(1, "model_name", model_name)
            selected_tracking.insert(2, "scheme_name", "priority_function_curated_context")
            selected_tracking.insert(
                3,
                "heldout_subject_index",
                int(heldout_tracking_rows.iloc[target_index]["target_subject_local_index"]),
            )
            selected_tracking["target_scope"] = "single_filtered_heldout_subject"
            selected_tracking["applies_to_subject_count"] = 1
            selected_tracking["selection_rank"] = np.arange(selected_tracking.shape[0], dtype=int)
            selected_tracking["ancestry_distance"] = distances[selected_indices]
            selected_tracking["priority_support_fraction"] = support_fraction[selected_indices]
            selected_tracking["priority_min_variant_support_fraction"] = experiment.priority_min_variant_support_fraction
            audit_frames.append(selected_tracking)

        disease_probabilities = np.asarray(per_subject_probabilities, dtype=float)
        auc_roc = float(roc_auc_score(actual_labels, disease_probabilities))
        ci_lower, ci_upper = _bootstrap_ci_if_requested(
            labels=actual_labels,
            probabilities=disease_probabilities,
            ci_level=ci_level,
            bootstrap_iterations=experiment.plot_bootstrap_iterations,
            seed=_stable_seed(dataset_pair_name, model_name, "priority_function_curated_context", "ci"),
        )

        prediction_frame = _build_prediction_frame(
            dataset_pair_name=dataset_pair_name,
            model_name=model_name,
            scheme_name="priority_function_curated_context",
            heldout_tracking_rows=heldout_tracking_rows,
            actual_labels=actual_labels,
            disease_probabilities=disease_probabilities,
        )
        audit_frame = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
        metrics = {
            "dataset_pair_name": dataset_pair_name,
            "model_name": model_name,
            "scheme_name": "priority_function_curated_context",
            "auc_roc": auc_roc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "plot_order": SCHEME_PLOT_ORDER["priority_function_curated_context"],
            "subject_count": int(actual_labels.shape[0]),
            "context_row_count": float(np.mean(context_sizes)) if context_sizes else 0.0,
            "min_context_row_count": int(min(context_sizes)) if context_sizes else 0,
            "max_context_row_count": int(max(context_sizes)) if context_sizes else 0,
            "fit_seconds": fit_seconds_total,
            "predict_seconds": predict_seconds_total,
            "experiment_dir": str(experiment_dir),
        }
    return prediction_frame, audit_frame, metrics


def _build_plot_title_and_subtitle(
    *,
    experiment: ExperimentConfig,
    model_name: str,
    subject_count: int,
    ci_level: float | None,
) -> tuple[str, str, str]:
    model_display = model_name.upper() if model_name == "tabpfn" else "TabICL"
    experiment_name_lower = experiment.name.lower()
    if "onco" in experiment_name_lower:
        dataset_label = "OncoArray"
        file_prefix = "oncoarray"
    elif experiment_name_lower.startswith("mec"):
        dataset_label = "MEC"
        file_prefix = "mec"
    elif experiment_name_lower.startswith("aou"):
        dataset_label = "AOU"
        file_prefix = "aou"
    else:
        dataset_label = experiment.name.replace("_", " ").title()
        file_prefix = _safe_slug(experiment.name)

    title = (
        f"Held-out {experiment.heldout_target_ancestry_group} {dataset_label} "
        f"with {model_display}"
    )
    subtitle = None
    if ci_level is not None:
        subtitle = f"Error bars: {int(round(ci_level * 100))}% bootstrap CI. n={subject_count}"
    file_name = f"{file_prefix}_roc_auc_{_safe_slug(model_name)}.png"
    return title, subtitle, file_name


def _dataset_histogram_key(experiment_name: str) -> str:
    normalized_name = _safe_slug(experiment_name)
    for suffix in ("_tabpfn", "_tabicl"):
        if normalized_name.endswith(suffix):
            return normalized_name[: -len(suffix)]
    return normalized_name


def _build_context_histogram_descriptor(experiment_name: str) -> tuple[str, str]:
    dataset_key = _dataset_histogram_key(experiment_name)
    if dataset_key.startswith("oncoarray"):
        title = "OncoArray"
    elif dataset_key.startswith("mec"):
        title = "MEC"
    else:
        title = dataset_key.replace("_", " ").title()
    return title, f"{dataset_key}_incontext_exemplar_counts.png"


def _context_count_profile(result: dict[str, Any]) -> tuple[str, int, int, tuple[int, ...]] | None:
    metric_frame = result.get("metric_frame")
    audit_frame = result.get("audit_frame")
    if not isinstance(metric_frame, pd.DataFrame) or not isinstance(audit_frame, pd.DataFrame):
        return None

    priority_rows = audit_frame.loc[
        audit_frame["scheme_name"] == "priority_function_curated_context",
        ["heldout_subject_index"],
    ]
    if priority_rows.empty:
        return None
    priority_counts = tuple(
        int(value)
        for value in priority_rows.groupby("heldout_subject_index", dropna=False).size().sort_index().tolist()
    )

    def _metric_context_count(scheme_name: str) -> int:
        scheme_rows = metric_frame.loc[metric_frame["scheme_name"] == scheme_name, "context_row_count"]
        if scheme_rows.empty:
            raise ValueError(f"Missing scheme metrics for {scheme_name!r} in experiment {result['experiment_name']!r}.")
        return int(round(float(scheme_rows.iloc[0])))

    return (
        result["experiment_name"],
        _metric_context_count("mixture_learning"),
        _metric_context_count("independent_learning_scheme"),
        priority_counts,
    )


def _plot_context_size_histogram(
    *,
    dataset_label: str,
    output_path: Path,
    priority_counts: Sequence[int],
    mixture_count: int,
    independent_count: int,
) -> None:
    if not priority_counts:
        raise ValueError(f"Cannot plot context-size histogram for {dataset_label}: no priority counts were provided.")

    priority_array = np.asarray(priority_counts, dtype=int)
    min_count = int(min(priority_array.min(), mixture_count, independent_count))
    max_count = int(max(priority_array.max(), mixture_count, independent_count))
    bins = np.arange(min_count - 0.5, max_count + 1.5, 1.0)
    if bins.size < 2:
        bins = np.array([min_count - 0.5, min_count + 0.5], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.hist(priority_array, bins=bins, color="#1f4e79", edgecolor="#1b1b1b", alpha=0.88, label="Priority")
    ax.axvline(mixture_count, color="#9c8f7a", linewidth=2.4, label="Mixture")
    ax.axvline(independent_count, color="#6c757d", linewidth=2.4, label="Independent")
    ax.set_title(dataset_label, fontsize=14, loc="left", pad=10)
    ax.set_xlabel("Num In-Context Exemplars", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_context_size_histograms(
    *,
    run_dir: Path,
    experiment_results: Sequence[dict[str, Any]],
) -> list[str]:
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    grouped_results: dict[str, list[dict[str, Any]]] = {}
    for result in experiment_results:
        dataset_key = _dataset_histogram_key(str(result["experiment_name"]))
        grouped_results.setdefault(dataset_key, []).append(result)

    histogram_paths: list[str] = []
    for dataset_results in grouped_results.values():
        reference_profile = _context_count_profile(dataset_results[0])
        if reference_profile is None:
            continue
        for candidate_result in dataset_results[1:]:
            candidate_profile = _context_count_profile(candidate_result)
            if candidate_profile is None:
                continue
            if candidate_profile[1:] != reference_profile[1:]:
                raise ValueError(
                    "Selection counts differed across models for dataset-level histogram generation: "
                    f"{reference_profile[0]!r} vs {candidate_profile[0]!r}."
                )

        dataset_label, file_name = _build_context_histogram_descriptor(reference_profile[0])
        output_path = figure_dir / file_name
        _plot_context_size_histogram(
            dataset_label=dataset_label,
            output_path=output_path,
            priority_counts=reference_profile[3],
            mixture_count=reference_profile[1],
            independent_count=reference_profile[2],
        )
        histogram_paths.append(str(output_path))
    return histogram_paths


def _prepare_experiment(
    *,
    config: RunnerConfig,
    experiment: ExperimentConfig,
    run_dir: Path,
    shared_priority_cache_paths: dict[tuple[Any, ...], Path],
    event_logger: logging.Logger | None = None,
) -> PreparedExperiment:
    experiment_dir = run_dir / experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with _timed_event(
        event_logger,
        "experiment_prepare",
        experiment_name=experiment.name,
        model_name=experiment.model_name,
    ):
        with _timed_event(event_logger, "data_load", experiment_name=experiment.name, slice_name="training"):
            training_frame, training_lengths, training_imputed = _load_and_impute_dataframe(
                experiment.training_pickle_paths
            )
        with _timed_event(event_logger, "data_load", experiment_name=experiment.name, slice_name="calibration"):
            calibrating_frame, calibrating_lengths, calibration_imputed = _load_and_impute_dataframe(
                experiment.calibrating_pickle_paths
            )
        with _timed_event(event_logger, "data_load", experiment_name=experiment.name, slice_name="heldout"):
            heldout_frame_all, heldout_lengths, heldout_imputed = _load_and_impute_dataframe(
                experiment.heldout_pickle_paths
            )

        reference_frame = _combine_data_objects((training_frame, calibrating_frame))
        if not isinstance(reference_frame, pd.DataFrame):
            raise TypeError("Reference cohort must be a pandas DataFrame.")
        reference_frame = reference_frame.reset_index(drop=True)

        with _timed_event(event_logger, "tracking_rows", experiment_name=experiment.name):
            reference_tracking_rows = _reference_tracking_rows(
                training_pickle_paths=experiment.training_pickle_paths,
                training_lengths=training_lengths,
                calibrating_pickle_paths=experiment.calibrating_pickle_paths,
                calibrating_lengths=calibrating_lengths,
                output_row_tracking_path=experiment.output_row_tracking_path,
                supported_ancestry_groups=experiment.supported_ancestry_groups,
            )
            heldout_tracking_rows_all = _heldout_tracking_rows(
                heldout_pickle_paths=experiment.heldout_pickle_paths,
                heldout_lengths=heldout_lengths,
                output_row_tracking_path=experiment.output_row_tracking_path,
                supported_ancestry_groups=experiment.supported_ancestry_groups,
            )
            filtered_heldout_tracking_rows = _filter_tracking_rows_for_target_ancestry(
                heldout_tracking_rows_all,
                target_ancestry_group=experiment.heldout_target_ancestry_group,
            )
            heldout_frame = _rows_from_tracking(heldout_frame_all, filtered_heldout_tracking_rows)

        with _timed_event(event_logger, "feature_resolution", experiment_name=experiment.name):
            variant_columns, feature_columns = _resolve_feature_columns(
                reference_frame,
                heldout_frame,
                dosage_prefix=experiment.dosage_prefix,
                ancestry_columns=experiment.ancestry_columns,
                additional_feature_columns=experiment.additional_feature_columns,
            )
            actual_labels = _extract_binary_labels(heldout_frame, label_column=experiment.label_column)
            model_cap = _derive_model_context_cap(
                model_name=experiment.model_name,
                feature_count=len(feature_columns),
                reference_row_count=reference_frame.shape[0],
            )
            effective_cap = _effective_context_cap(requested_cap=experiment.max_context_rows, model_cap=model_cap)

        cache_identity = _priority_cache_identity(
            experiment=experiment,
            prio_function_path=config.prio_function_path,
            function_name=config.function_name,
        )
        cache_source_experiment = experiment
        if experiment.priority_radius_cache_path is None and cache_identity in shared_priority_cache_paths:
            shared_cache_path = shared_priority_cache_paths[cache_identity]
            cache_source_experiment = replace(experiment, priority_radius_cache_path=shared_cache_path)
            _log_event(
                event_logger,
                "priority_cache.reuse_same_run_candidate",
                experiment_name=experiment.name,
                source_cache_path=shared_cache_path,
            )

        with _timed_event(event_logger, "priority_cache", experiment_name=experiment.name):
            radius_cache = _build_priority_radius_cache(
                experiment=cache_source_experiment,
                experiment_dir=experiment_dir,
                prio_function_path=config.prio_function_path,
                function_name=config.function_name,
                reference_frame=reference_frame,
                reference_tracking_rows=reference_tracking_rows,
                heldout_frame=heldout_frame,
                heldout_tracking_rows=filtered_heldout_tracking_rows,
                variant_columns=variant_columns,
                event_logger=event_logger,
            )
        shared_priority_cache_paths.setdefault(cache_identity, radius_cache.manifest_path.parent.resolve())

    return PreparedExperiment(
        experiment=experiment,
        experiment_dir=experiment_dir,
        training_imputation_counts=training_imputed,
        calibration_imputation_counts=calibration_imputed,
        heldout_imputation_counts=heldout_imputed,
        reference_frame=reference_frame,
        reference_tracking_rows=reference_tracking_rows,
        heldout_frame=heldout_frame,
        heldout_tracking_rows=filtered_heldout_tracking_rows,
        variant_columns=tuple(variant_columns),
        feature_columns=tuple(feature_columns),
        actual_labels=actual_labels,
        model_cap=model_cap,
        effective_cap=effective_cap,
        radius_cache=radius_cache,
    )


def _evaluate_prepared_experiment(
    *,
    config: RunnerConfig,
    prepared: PreparedExperiment,
    event_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    experiment = prepared.experiment
    experiment_dir = prepared.experiment_dir

    with _timed_event(
        event_logger,
        "experiment_evaluate",
        experiment_name=experiment.name,
        model_name=experiment.model_name,
        cache_reuse_mode=prepared.radius_cache.reuse_mode,
    ):
        scheme_metric_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        audit_frames: list[pd.DataFrame] = []

        if "mixture_learning" in experiment.schemes:
            mixture_indices = np.arange(prepared.reference_frame.shape[0], dtype=int)
            if mixture_indices.shape[0] > prepared.effective_cap:
                mixture_indices = _select_random_context_indices(
                    mixture_indices,
                    desired_count=prepared.effective_cap,
                    seed=_stable_seed(config.random_seed, experiment.name, experiment.model_name, "mixture_learning"),
                )
            prediction_frame, audit_frame, metrics = _run_shared_context_scheme(
                dataset_pair_name=experiment.name,
                model_name=experiment.model_name,
                scheme_name="mixture_learning",
                experiment=experiment,
                experiment_dir=experiment_dir,
                reference_frame=prepared.reference_frame,
                reference_tracking_rows=prepared.reference_tracking_rows,
                heldout_frame=prepared.heldout_frame,
                heldout_tracking_rows=prepared.heldout_tracking_rows,
                feature_columns=prepared.feature_columns,
                actual_labels=prepared.actual_labels,
                selected_reference_indices=mixture_indices,
                ci_level=experiment.plot_ci_level,
                event_logger=event_logger,
            )
            prediction_frames.append(prediction_frame)
            audit_frames.append(audit_frame)
            scheme_metric_rows.append(metrics)

        if "independent_learning_scheme" in experiment.schemes:
            target_key = _normalize_group_key(experiment.heldout_target_ancestry_group)
            independent_indices = np.flatnonzero(
                prepared.reference_tracking_rows["ancestry_group"].astype(str).map(_normalize_group_key) == target_key
            ).astype(int, copy=False)
            if independent_indices.size == 0:
                raise ValueError(
                    "No same-ancestry reference rows were available for the Independent Learning Scheme."
                )
            if independent_indices.shape[0] > prepared.effective_cap:
                independent_indices = _select_random_context_indices(
                    independent_indices,
                    desired_count=prepared.effective_cap,
                    seed=_stable_seed(
                        config.random_seed,
                        experiment.name,
                        experiment.model_name,
                        "independent_learning_scheme",
                    ),
                )
            prediction_frame, audit_frame, metrics = _run_shared_context_scheme(
                dataset_pair_name=experiment.name,
                model_name=experiment.model_name,
                scheme_name="independent_learning_scheme",
                experiment=experiment,
                experiment_dir=experiment_dir,
                reference_frame=prepared.reference_frame,
                reference_tracking_rows=prepared.reference_tracking_rows,
                heldout_frame=prepared.heldout_frame,
                heldout_tracking_rows=prepared.heldout_tracking_rows,
                feature_columns=prepared.feature_columns,
                actual_labels=prepared.actual_labels,
                selected_reference_indices=independent_indices,
                ci_level=experiment.plot_ci_level,
                event_logger=event_logger,
            )
            prediction_frames.append(prediction_frame)
            audit_frames.append(audit_frame)
            scheme_metric_rows.append(metrics)

        if "priority_function_curated_context" in experiment.schemes:
            prediction_frame, audit_frame, metrics = _run_priority_context_scheme(
                dataset_pair_name=experiment.name,
                model_name=experiment.model_name,
                experiment=experiment,
                experiment_dir=experiment_dir,
                reference_frame=prepared.reference_frame,
                reference_tracking_rows=prepared.reference_tracking_rows,
                heldout_frame=prepared.heldout_frame,
                heldout_tracking_rows=prepared.heldout_tracking_rows,
                feature_columns=prepared.feature_columns,
                actual_labels=prepared.actual_labels,
                radius_cache=prepared.radius_cache,
                context_cap=prepared.effective_cap,
                ci_level=experiment.plot_ci_level,
                event_logger=event_logger,
            )
            prediction_frames.append(prediction_frame)
            audit_frames.append(audit_frame)
            scheme_metric_rows.append(metrics)

        if not scheme_metric_rows:
            raise ValueError(f"Experiment {experiment.name!r} produced no scheme outputs.")

        metric_frame = pd.DataFrame(scheme_metric_rows).sort_values("plot_order").reset_index(drop=True)
        prediction_frame = pd.concat(prediction_frames, ignore_index=True)
        audit_frame = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

        metric_frame["model_context_cap_rows"] = int(prepared.model_cap.rows_cap)
        metric_frame["requested_max_context_rows"] = experiment.max_context_rows
        metric_frame["effective_max_context_rows"] = int(prepared.effective_cap)
        metric_frame["priority_min_variant_support_fraction"] = experiment.priority_min_variant_support_fraction
        metric_frame["feature_count"] = int(len(prepared.feature_columns))
        metric_frame["reference_row_count"] = int(prepared.reference_frame.shape[0])
        metric_frame["heldout_subject_count"] = int(prepared.heldout_frame.shape[0])

        prediction_path = experiment_dir / "heldout_predictions.pkl"
        prediction_frame.to_pickle(prediction_path)
        audit_path = experiment_dir / "selected_context_audit.pkl"
        audit_frame.to_pickle(audit_path)
        metric_path = experiment_dir / "scheme_metrics.csv"
        metric_frame.to_csv(metric_path, index=False)

        figure_dir = experiment_dir.parent / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        title, subtitle, file_name = _build_plot_title_and_subtitle(
            experiment=experiment,
            model_name=experiment.model_name,
            subject_count=prepared.heldout_frame.shape[0],
            ci_level=experiment.plot_ci_level,
        )
        figure_path = figure_dir / file_name
        _plot_experiment_metrics(
            metric_frame=metric_frame,
            output_path=figure_path,
            title=title,
            subtitle=subtitle,
        )

        summary_payload = {
            "experiment_name": experiment.name,
            "model_name": experiment.model_name,
            "priority_function_path": str(config.prio_function_path),
            "function_name": config.function_name,
            "training_pickle_paths": [str(path) for path in experiment.training_pickle_paths],
            "calibrating_pickle_paths": [str(path) for path in experiment.calibrating_pickle_paths],
            "heldout_pickle_paths": [str(path) for path in experiment.heldout_pickle_paths],
            "output_row_tracking_path": str(experiment.output_row_tracking_path),
            "supported_ancestry_groups": list(experiment.supported_ancestry_groups),
            "heldout_target_ancestry_group": experiment.heldout_target_ancestry_group,
            "feature_columns": list(prepared.feature_columns),
            "variant_columns": list(prepared.variant_columns),
            "label_column": experiment.label_column,
            "plot_path": str(figure_path),
            "prediction_path": str(prediction_path),
            "selected_context_audit_path": str(audit_path),
            "scheme_metrics_path": str(metric_path),
            "priority_radius_cache_manifest_path": str(prepared.radius_cache.manifest_path),
            "priority_radius_cache_reuse_mode": prepared.radius_cache.reuse_mode,
            "priority_radius_cache_reuse_source_path": (
                str(prepared.radius_cache.reuse_source_cache_path)
                if prepared.radius_cache.reuse_source_cache_path is not None
                else None
            ),
            "priority_radius_cache_requested_path": (
                str(experiment.priority_radius_cache_path) if experiment.priority_radius_cache_path is not None else None
            ),
            "model_context_cap": asdict(prepared.model_cap),
            "requested_max_context_rows": experiment.max_context_rows,
            "effective_max_context_rows": prepared.effective_cap,
            "priority_min_variant_support_fraction": experiment.priority_min_variant_support_fraction,
            "reference_row_count": int(prepared.reference_frame.shape[0]),
            "heldout_subject_count": int(prepared.heldout_frame.shape[0]),
            "training_imputation_counts": prepared.training_imputation_counts,
            "calibration_imputation_counts": prepared.calibration_imputation_counts,
            "heldout_imputation_counts": prepared.heldout_imputation_counts,
            "scheme_metrics": metric_frame.to_dict(orient="records"),
        }
        summary_path = experiment_dir / "summary.json"
        _write_json(summary_path, summary_payload)
    return {
        "experiment_name": experiment.name,
        "model_name": experiment.model_name,
        "experiment_dir": str(experiment_dir),
        "prediction_path": str(prediction_path),
        "audit_path": str(audit_path),
        "scheme_metrics_path": str(metric_path),
        "summary_path": str(summary_path),
        "plot_path": str(figure_path),
        "priority_radius_cache_manifest_path": str(prepared.radius_cache.manifest_path),
        "metric_frame": metric_frame,
        "prediction_frame": prediction_frame,
        "audit_frame": audit_frame,
        "summary_payload": summary_payload,
    }


def _build_run_notes(config: RunnerConfig, run_dir: Path, experiment_results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Tabular Foundation Model Run Notes",
        "",
        f"- Run directory: {run_dir}",
        f"- Config path: {config.config_path}",
        f"- Priority function path: {config.prio_function_path}",
        f"- Function name: {config.function_name}",
        f"- Random seed: {config.random_seed}",
        "",
        "## Plot Contents",
        "",
        "Each PNG figure is a horizontal ROC-AUC bar chart comparing the three context-selection schemes for one dataset pair and one off-the-shelf tabular foundation model.",
        "",
        "- The x-axis is ROC AUC on the requested heldout focal-ancestry subjects only.",
        "- The y-axis lists Priority Function Curated Context, Mixture Learning, and Independent Learning Scheme.",
        "- OncoArray figures include bootstrap confidence-interval error bars when requested in the config.",
        "- MEC figures omit confidence intervals when the config sets plot_ci_level to null.",
        "- The priority-function bar uses the subset of reference rows selected by the stored per-subject per-variant radius cache; Mixture Learning uses the full eligible pooled reference cohort up to the model-spec cap; Independent Learning uses the same-ancestry pooled reference cohort up to the model-spec cap.",
        "- The central figures directory also includes one dataset-level histogram of in-context exemplar counts. The histogram bars show the Priority Function Curated Context distribution across heldout subjects, and vertical lines mark the Mixture Learning and Independent Learning counts.",
        "",
        "## Experiments",
        "",
    ]
    for result in experiment_results:
        summary_payload = result["summary_payload"]
        lines.append(f"### {summary_payload['experiment_name']}")
        lines.append("")
        lines.append(f"- Model: {summary_payload['model_name']}")
        lines.append(f"- Heldout target ancestry: {summary_payload['heldout_target_ancestry_group']}")
        lines.append(f"- Reference row count: {summary_payload['reference_row_count']}")
        lines.append(f"- Heldout subject count: {summary_payload['heldout_subject_count']}")
        lines.append(
            f"- Model-spec context cap: {summary_payload['model_context_cap']['rows_cap']} rows "
            f"from {summary_payload['model_context_cap']['source_name']}"
        )
        for warning in summary_payload["model_context_cap"]["warnings"]:
            lines.append(f"- Warning: {warning}")
        lines.append(f"- Plot: {summary_payload['plot_path']}")
        lines.append(f"- Predictions: {summary_payload['prediction_path']}")
        lines.append(f"- Radius cache manifest: {summary_payload['priority_radius_cache_manifest_path']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_from_config_path(config_path: str | Path) -> Path:
    config = load_config(config_path)
    run_dir = _create_run_directory(config)
    event_logger = _create_run_logger(run_dir)
    try:
        with _timed_event(
            event_logger,
            "run",
            run_dir=run_dir,
            config_path=config.config_path,
            experiment_count=len(config.experiments),
        ):
            _write_json(
                run_dir / "run_config.used.json",
                json.loads(config.config_path.read_text(encoding="utf-8")),
            )
            _log_event(event_logger, "run_config.written", path=run_dir / "run_config.used.json")

            shared_priority_cache_paths: dict[tuple[Any, ...], Path] = {}
            with _timed_event(event_logger, "priority_cache_phase", experiment_count=len(config.experiments)):
                prepared_experiments = [
                    _prepare_experiment(
                        config=config,
                        experiment=experiment,
                        run_dir=run_dir,
                        shared_priority_cache_paths=shared_priority_cache_paths,
                        event_logger=event_logger,
                    )
                    for experiment in config.experiments
                ]

            with _timed_event(event_logger, "evaluation_phase", experiment_count=len(config.experiments)):
                experiment_results = [
                    _evaluate_prepared_experiment(
                        config=config,
                        prepared=prepared_experiment,
                        event_logger=event_logger,
                    )
                    for prepared_experiment in prepared_experiments
                ]

            aggregate_prediction_frame = pd.concat(
                [result["prediction_frame"] for result in experiment_results],
                ignore_index=True,
            )
            aggregate_prediction_frame.to_pickle(run_dir / "heldout_predictions.pkl")
            _log_event(event_logger, "aggregate_predictions.written", row_count=int(aggregate_prediction_frame.shape[0]))

            aggregate_audit_frame = pd.concat(
                [result["audit_frame"] for result in experiment_results],
                ignore_index=True,
            )
            aggregate_audit_frame.to_pickle(run_dir / "selected_context_audit.pkl")
            _log_event(event_logger, "aggregate_audit.written", row_count=int(aggregate_audit_frame.shape[0]))

            aggregate_metric_frame = pd.concat(
                [result["metric_frame"] for result in experiment_results],
                ignore_index=True,
            )
            aggregate_metric_frame.to_csv(run_dir / "scheme_metrics.csv", index=False)
            _log_event(event_logger, "aggregate_metrics.written", row_count=int(aggregate_metric_frame.shape[0]))

            context_histogram_paths = _write_context_size_histograms(
                run_dir=run_dir,
                experiment_results=experiment_results,
            )
            _log_event(event_logger, "context_histograms.written", count=len(context_histogram_paths))

            summary_payload = {
                "run_dir": str(run_dir),
                "config_path": str(config.config_path),
                "priority_function_path": str(config.prio_function_path),
                "function_name": config.function_name,
                "random_seed": config.random_seed,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_events_log_path": str((run_dir / "run_events.log").resolve()),
                "context_size_histogram_paths": context_histogram_paths,
                "experiment_summaries": [result["summary_payload"] for result in experiment_results],
            }
            _write_json(run_dir / "summary.json", summary_payload)
            _log_event(event_logger, "run_summary.written", path=run_dir / "summary.json")

            (run_dir / "README_run_notes.md").write_text(
                _build_run_notes(config, run_dir, experiment_results),
                encoding="utf-8",
            )
            _log_event(event_logger, "run_notes.written", path=run_dir / "README_run_notes.md")

        print(f"Wrote run outputs to {run_dir}")
        return run_dir
    finally:
        _close_run_logger(event_logger)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run off-the-shelf TabPFN and TabICL experiments with mixture, same-ancestry, "
            "and priority-function-curated context selection."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    args = parser.parse_args(argv)
    run_from_config_path(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())