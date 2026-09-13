from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Sequence
import json
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GenomicsHelpers.ancestry_distance_cache import load_distance_cache

from ExperimentsWithTabularFoundationModels import runner as base_runner


DEFAULT_RUN_LABEL = "tabular_foundation_models2"
VARIANT_PREDICTION_COLUMNS = (
    "dataset_pair_name",
    "model_name",
    "scheme_name",
    "variant_index",
    "variant_name",
    "variant_dosage_field",
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
VARIANT_CONTEXT_SUMMARY_COLUMNS = (
    "dataset_pair_name",
    "model_name",
    "scheme_name",
    "variant_index",
    "variant_name",
    "variant_dosage_field",
    "heldout_subject_index",
    "context_scope",
    "context_row_count",
    "priority_radius",
    "selected_min_distance",
    "selected_max_distance",
    "fallback_used",
)
VARIANT_METRIC_COLUMNS = (
    "dataset_pair_name",
    "model_name",
    "scheme_name",
    "variant_index",
    "variant_name",
    "variant_dosage_field",
    "auc_roc",
    "subject_count",
    "context_row_count",
    "min_context_row_count",
    "max_context_row_count",
    "fit_seconds",
    "predict_seconds",
    "plot_order",
)
SCHEME_COLORS = {
    "priority_function_curated_context": "#1f4e79",
    "mixture_learning": "#9c8f7a",
    "independent_learning_scheme": "#6c757d",
}


@dataclass(frozen=True)
class VariantMetadata:
    variant_index: int
    variant_name: str
    dosage_field: str


@dataclass
class PreparedVariantExperiment:
    experiment: base_runner.ExperimentConfig
    experiment_dir: Path
    training_imputation_counts: dict[str, int]
    calibration_imputation_counts: dict[str, int]
    heldout_imputation_counts: dict[str, int]
    reference_frame: pd.DataFrame
    reference_tracking_rows: pd.DataFrame
    heldout_frame: pd.DataFrame
    heldout_tracking_rows: pd.DataFrame
    variant_metadata: tuple[VariantMetadata, ...]
    actual_labels: np.ndarray
    model_cap: base_runner.ModelContextCap
    effective_cap: int
    radius_cache: base_runner.RadiusCacheResult


def _create_run_directory(config: base_runner.RunnerConfig) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root_dir / f"{timestamp}_{DEFAULT_RUN_LABEL}"
    if run_dir.exists() and not config.should_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_variant_metadata(
    radius_cache: base_runner.RadiusCacheResult,
    variant_columns: Sequence[str],
) -> tuple[VariantMetadata, ...]:
    metadata_frame = (
        radius_cache.long_frame.loc[:, ["variant_index", "variant_name", "dosage_field"]]
        .drop_duplicates()
        .sort_values("variant_index")
        .reset_index(drop=True)
    )
    if metadata_frame.shape[0] != len(variant_columns):
        raise ValueError(
            "Priority radius cache metadata did not match the discovered dosage columns: "
            f"cache_variants={metadata_frame.shape[0]} discovered_variants={len(variant_columns)}."
        )
    metadata: list[VariantMetadata] = []
    for index, row in metadata_frame.iterrows():
        dosage_field = str(row["dosage_field"])
        expected_dosage_field = str(variant_columns[index])
        if dosage_field != expected_dosage_field:
            raise ValueError(
                "Priority radius cache dosage-field order did not match the experiment data: "
                f"cache={dosage_field!r} discovered={expected_dosage_field!r}."
            )
        metadata.append(
            VariantMetadata(
                variant_index=int(row["variant_index"]),
                variant_name=str(row["variant_name"]),
                dosage_field=dosage_field,
            )
        )
    return tuple(metadata)


def _build_variant_feature_columns(
    experiment: base_runner.ExperimentConfig,
    dosage_field: str,
) -> tuple[str, ...]:
    return (
        dosage_field,
        *experiment.ancestry_columns,
        *experiment.additional_feature_columns,
    )


def _build_variant_prediction_frame(
    *,
    dataset_pair_name: str,
    model_name: str,
    scheme_name: str,
    variant: VariantMetadata,
    heldout_tracking_rows: pd.DataFrame,
    actual_labels: np.ndarray,
    disease_probabilities: np.ndarray,
) -> pd.DataFrame:
    prediction_frame = base_runner._build_prediction_frame(
        dataset_pair_name=dataset_pair_name,
        model_name=model_name,
        scheme_name=scheme_name,
        heldout_tracking_rows=heldout_tracking_rows,
        actual_labels=actual_labels,
        disease_probabilities=disease_probabilities,
    )
    prediction_frame.insert(3, "variant_dosage_field", variant.dosage_field)
    prediction_frame.insert(3, "variant_name", variant.variant_name)
    prediction_frame.insert(3, "variant_index", variant.variant_index)
    return prediction_frame.loc[:, list(VARIANT_PREDICTION_COLUMNS)]


def _select_priority_context_indices_for_variant(
    *,
    target_index: int,
    variant_index: int,
    radius_matrix: np.ndarray,
    opened_distance_cache: Any,
    reference_tracking_rows: pd.DataFrame,
    context_cap: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    row_view = opened_distance_cache.row_view(target_index)
    distances = np.asarray(row_view.distances, dtype=float)
    radius = float(radius_matrix[target_index, variant_index])

    candidate_indices = np.flatnonzero(distances <= radius).astype(int, copy=False)
    fallback_used = False
    if candidate_indices.size == 0:
        candidate_indices = np.array([int(np.argmin(distances))], dtype=int)
        fallback_used = True

    candidate_distances = distances[candidate_indices]
    candidate_row_indices = reference_tracking_rows.iloc[candidate_indices]["dataset_row_index"].to_numpy(
        dtype=int,
        copy=False,
    )
    rank_order = np.lexsort((candidate_row_indices, candidate_distances))
    selected_indices = candidate_indices[rank_order]
    if selected_indices.shape[0] > context_cap:
        selected_indices = selected_indices[:context_cap]
    return selected_indices.astype(int, copy=False), distances, radius, fallback_used


def _run_shared_context_scheme_across_variants(
    *,
    dataset_pair_name: str,
    model_name: str,
    scheme_name: str,
    experiment: base_runner.ExperimentConfig,
    reference_frame: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    heldout_tracking_rows: pd.DataFrame,
    actual_labels: np.ndarray,
    selected_reference_indices: np.ndarray,
    variant_metadata: Sequence[VariantMetadata],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context_frame = reference_frame.iloc[selected_reference_indices].reset_index(drop=True)
    context_labels = base_runner._extract_binary_labels(context_frame, label_column=experiment.label_column)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for variant in variant_metadata:
        feature_columns = list(_build_variant_feature_columns(experiment, variant.dosage_field))

        estimator = base_runner._build_model(
            model_name=model_name,
            model_init_kwargs=experiment.model_init_kwargs,
        )
        fit_start = time.perf_counter()
        estimator.fit(context_frame.loc[:, feature_columns], context_labels)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        disease_probabilities = base_runner._predict_positive_probability(
            estimator,
            heldout_frame.loc[:, feature_columns],
            batch_size=experiment.predict_proba_batch_size,
        )
        predict_seconds = time.perf_counter() - predict_start
        auc_roc = float(roc_auc_score(actual_labels, disease_probabilities))

        prediction_frames.append(
            _build_variant_prediction_frame(
                dataset_pair_name=dataset_pair_name,
                model_name=model_name,
                scheme_name=scheme_name,
                variant=variant,
                heldout_tracking_rows=heldout_tracking_rows,
                actual_labels=actual_labels,
                disease_probabilities=disease_probabilities,
            )
        )
        metric_rows.append(
            {
                "dataset_pair_name": dataset_pair_name,
                "model_name": model_name,
                "scheme_name": scheme_name,
                "variant_index": variant.variant_index,
                "variant_name": variant.variant_name,
                "variant_dosage_field": variant.dosage_field,
                "auc_roc": auc_roc,
                "subject_count": int(actual_labels.shape[0]),
                "context_row_count": int(context_frame.shape[0]),
                "min_context_row_count": int(context_frame.shape[0]),
                "max_context_row_count": int(context_frame.shape[0]),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "plot_order": base_runner.SCHEME_PLOT_ORDER[scheme_name],
            }
        )
        context_rows.append(
            {
                "dataset_pair_name": dataset_pair_name,
                "model_name": model_name,
                "scheme_name": scheme_name,
                "variant_index": variant.variant_index,
                "variant_name": variant.variant_name,
                "variant_dosage_field": variant.dosage_field,
                "heldout_subject_index": pd.NA,
                "context_scope": "all_filtered_heldout_subjects",
                "context_row_count": int(context_frame.shape[0]),
                "priority_radius": np.nan,
                "selected_min_distance": np.nan,
                "selected_max_distance": np.nan,
                "fallback_used": pd.NA,
            }
        )

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(metric_rows).loc[:, list(VARIANT_METRIC_COLUMNS)],
        pd.DataFrame(context_rows).loc[:, list(VARIANT_CONTEXT_SUMMARY_COLUMNS)],
    )


def _run_priority_context_scheme_across_variants(
    *,
    dataset_pair_name: str,
    model_name: str,
    experiment: base_runner.ExperimentConfig,
    reference_frame: pd.DataFrame,
    reference_tracking_rows: pd.DataFrame,
    heldout_frame: pd.DataFrame,
    heldout_tracking_rows: pd.DataFrame,
    actual_labels: np.ndarray,
    radius_cache: base_runner.RadiusCacheResult,
    context_cap: int,
    variant_metadata: Sequence[VariantMetadata],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opened_distance_cache = load_distance_cache(radius_cache.distance_cache_manifest_path)
    if opened_distance_cache is None:
        raise ValueError("Could not load the priority ancestry distance cache for variant-level selection.")

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for variant in variant_metadata:
        variant_probabilities: list[float] = []
        context_sizes: list[int] = []
        fit_seconds_total = 0.0
        predict_seconds_total = 0.0

        feature_columns = list(_build_variant_feature_columns(experiment, variant.dosage_field))
        for target_index in range(heldout_frame.shape[0]):
            selected_indices, distances, radius, fallback_used = _select_priority_context_indices_for_variant(
                target_index=target_index,
                variant_index=variant.variant_index,
                radius_matrix=radius_cache.radius_matrix,
                opened_distance_cache=opened_distance_cache,
                reference_tracking_rows=reference_tracking_rows,
                context_cap=context_cap,
            )
            selected_distances = distances[selected_indices]
            context_sizes.append(int(selected_indices.shape[0]))

            context_frame = reference_frame.iloc[selected_indices].reset_index(drop=True)
            context_labels = base_runner._extract_binary_labels(context_frame, label_column=experiment.label_column)

            estimator = base_runner._build_model(
                model_name=model_name,
                model_init_kwargs=experiment.model_init_kwargs,
            )
            fit_start = time.perf_counter()
            estimator.fit(context_frame.loc[:, feature_columns], context_labels)
            fit_seconds_total += time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            probability = float(
                base_runner._predict_positive_probability(
                    estimator,
                    heldout_frame.iloc[[target_index]].loc[:, feature_columns],
                    batch_size=None,
                )[0]
            )
            predict_seconds_total += time.perf_counter() - predict_start
            variant_probabilities.append(probability)

            context_rows.append(
                {
                    "dataset_pair_name": dataset_pair_name,
                    "model_name": model_name,
                    "scheme_name": "priority_function_curated_context",
                    "variant_index": variant.variant_index,
                    "variant_name": variant.variant_name,
                    "variant_dosage_field": variant.dosage_field,
                    "heldout_subject_index": int(heldout_tracking_rows.iloc[target_index]["target_subject_local_index"]),
                    "context_scope": "single_filtered_heldout_subject",
                    "context_row_count": int(selected_indices.shape[0]),
                    "priority_radius": radius,
                    "selected_min_distance": float(selected_distances.min()),
                    "selected_max_distance": float(selected_distances.max()),
                    "fallback_used": fallback_used,
                }
            )

        disease_probabilities = np.asarray(variant_probabilities, dtype=float)
        auc_roc = float(roc_auc_score(actual_labels, disease_probabilities))
        prediction_frames.append(
            _build_variant_prediction_frame(
                dataset_pair_name=dataset_pair_name,
                model_name=model_name,
                scheme_name="priority_function_curated_context",
                variant=variant,
                heldout_tracking_rows=heldout_tracking_rows,
                actual_labels=actual_labels,
                disease_probabilities=disease_probabilities,
            )
        )
        metric_rows.append(
            {
                "dataset_pair_name": dataset_pair_name,
                "model_name": model_name,
                "scheme_name": "priority_function_curated_context",
                "variant_index": variant.variant_index,
                "variant_name": variant.variant_name,
                "variant_dosage_field": variant.dosage_field,
                "auc_roc": auc_roc,
                "subject_count": int(actual_labels.shape[0]),
                "context_row_count": float(np.mean(context_sizes)) if context_sizes else 0.0,
                "min_context_row_count": int(min(context_sizes)) if context_sizes else 0,
                "max_context_row_count": int(max(context_sizes)) if context_sizes else 0,
                "fit_seconds": fit_seconds_total,
                "predict_seconds": predict_seconds_total,
                "plot_order": base_runner.SCHEME_PLOT_ORDER["priority_function_curated_context"],
            }
        )

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(metric_rows).loc[:, list(VARIANT_METRIC_COLUMNS)],
        pd.DataFrame(context_rows).loc[:, list(VARIANT_CONTEXT_SUMMARY_COLUMNS)],
    )


def _build_variant_auc_triplet_frame(metric_frame: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "dataset_pair_name",
        "model_name",
        "variant_index",
        "variant_name",
        "variant_dosage_field",
    ]
    pivot = (
        metric_frame.pivot_table(
            index=index_columns,
            columns="scheme_name",
            values="auc_roc",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for scheme_name in base_runner.DEFAULT_SCHEMES:
        if scheme_name not in pivot.columns:
            pivot[scheme_name] = np.nan
    rename_map = {
        scheme_name: f"{scheme_name}_auc_roc"
        for scheme_name in base_runner.DEFAULT_SCHEMES
    }
    return pivot.rename(columns=rename_map)


def _plot_variant_task_boxplots(
    *,
    metric_frame: pd.DataFrame,
    output_path: Path,
    seed: int,
) -> None:
    ordered_schemes = sorted(base_runner.DEFAULT_SCHEMES, key=base_runner.SCHEME_PLOT_ORDER.get)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    rng = np.random.default_rng(seed)

    values_by_scheme: list[np.ndarray] = []
    for scheme_name in ordered_schemes:
        values = metric_frame.loc[metric_frame["scheme_name"] == scheme_name, "auc_roc"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No per-variant ROC AUC values were available for scheme {scheme_name!r}.")
        values_by_scheme.append(values)

    positions = np.arange(1, len(ordered_schemes) + 1, dtype=float)
    box = ax.boxplot(
        values_by_scheme,
        positions=positions,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#222222", "markeredgecolor": "#222222", "markersize": 5},
    )
    for patch, scheme_name in zip(box["boxes"], ordered_schemes):
        patch.set_facecolor(SCHEME_COLORS[scheme_name])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#222222")
    for median in box["medians"]:
        median.set_color("#7a1f1f")
        median.set_linewidth(2.0)
    for whisker in box["whiskers"]:
        whisker.set_color("#222222")
    for cap in box["caps"]:
        cap.set_color("#222222")

    for position, values in zip(positions, values_by_scheme):
        jitter = rng.normal(loc=position, scale=0.04, size=values.shape[0])
        ax.scatter(jitter, values, color="#222222", alpha=0.5, s=16)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [base_runner.SCHEME_DISPLAY_NAME[scheme_name] for scheme_name in ordered_schemes],
        fontsize=10,
    )
    ax.set_ylabel("ROC AUC", fontsize=12)
    ax.set_xlabel("Scheme", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_plot_file_name(
    *,
    experiment: base_runner.ExperimentConfig,
    model_name: str,
) -> str:
    experiment_name_lower = experiment.name.lower()
    if "onco" in experiment_name_lower:
        return f"oncoarray_{base_runner._safe_slug(model_name)}_variant_task_boxplots.png"
    return f"mec_{base_runner._safe_slug(model_name)}_variant_task_boxplots.png"


def _prepare_experiment(
    *,
    config: base_runner.RunnerConfig,
    experiment: base_runner.ExperimentConfig,
    run_dir: Path,
    shared_priority_cache_paths: dict[tuple[Any, ...], Path],
    event_logger: Any,
) -> PreparedVariantExperiment:
    experiment_dir = run_dir / experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with base_runner._timed_event(
        event_logger,
        "experiment_prepare",
        experiment_name=experiment.name,
        model_name=experiment.model_name,
    ):
        with base_runner._timed_event(event_logger, "data_load", experiment_name=experiment.name, slice_name="training"):
            training_frame, training_lengths, training_imputed = base_runner._load_and_impute_dataframe(
                experiment.training_pickle_paths
            )
        with base_runner._timed_event(event_logger, "data_load", experiment_name=experiment.name, slice_name="calibration"):
            calibrating_frame, calibrating_lengths, calibration_imputed = base_runner._load_and_impute_dataframe(
                experiment.calibrating_pickle_paths
            )
        with base_runner._timed_event(event_logger, "data_load", experiment_name=experiment.name, slice_name="heldout"):
            heldout_frame_all, heldout_lengths, heldout_imputed = base_runner._load_and_impute_dataframe(
                experiment.heldout_pickle_paths
            )

        reference_frame = base_runner._combine_data_objects((training_frame, calibrating_frame))
        if not isinstance(reference_frame, pd.DataFrame):
            raise TypeError("Reference cohort must be a pandas DataFrame.")
        reference_frame = reference_frame.reset_index(drop=True)

        with base_runner._timed_event(event_logger, "tracking_rows", experiment_name=experiment.name):
            reference_tracking_rows = base_runner._reference_tracking_rows(
                training_pickle_paths=experiment.training_pickle_paths,
                training_lengths=training_lengths,
                calibrating_pickle_paths=experiment.calibrating_pickle_paths,
                calibrating_lengths=calibrating_lengths,
                output_row_tracking_path=experiment.output_row_tracking_path,
                supported_ancestry_groups=experiment.supported_ancestry_groups,
            )
            heldout_tracking_rows_all = base_runner._heldout_tracking_rows(
                heldout_pickle_paths=experiment.heldout_pickle_paths,
                heldout_lengths=heldout_lengths,
                output_row_tracking_path=experiment.output_row_tracking_path,
                supported_ancestry_groups=experiment.supported_ancestry_groups,
            )
            filtered_heldout_tracking_rows = base_runner._filter_tracking_rows_for_target_ancestry(
                heldout_tracking_rows_all,
                target_ancestry_group=experiment.heldout_target_ancestry_group,
            )
            heldout_frame = base_runner._rows_from_tracking(heldout_frame_all, filtered_heldout_tracking_rows)

        with base_runner._timed_event(event_logger, "feature_resolution", experiment_name=experiment.name):
            variant_columns, _ = base_runner._resolve_feature_columns(
                reference_frame,
                heldout_frame,
                dosage_prefix=experiment.dosage_prefix,
                ancestry_columns=experiment.ancestry_columns,
                additional_feature_columns=experiment.additional_feature_columns,
            )
            actual_labels = base_runner._extract_binary_labels(heldout_frame, label_column=experiment.label_column)
            per_task_feature_count = 1 + len(experiment.ancestry_columns) + len(experiment.additional_feature_columns)
            model_cap = base_runner._derive_model_context_cap(
                model_name=experiment.model_name,
                feature_count=per_task_feature_count,
                reference_row_count=reference_frame.shape[0],
            )
            effective_cap = base_runner._effective_context_cap(
                requested_cap=experiment.max_context_rows,
                model_cap=model_cap,
            )

        cache_identity = base_runner._priority_cache_identity(
            experiment=experiment,
            prio_function_path=config.prio_function_path,
            function_name=config.function_name,
        )
        cache_source_experiment = experiment
        if experiment.priority_radius_cache_path is None and cache_identity in shared_priority_cache_paths:
            shared_cache_path = shared_priority_cache_paths[cache_identity]
            cache_source_experiment = replace(experiment, priority_radius_cache_path=shared_cache_path)
            base_runner._log_event(
                event_logger,
                "priority_cache.reuse_same_run_candidate",
                experiment_name=experiment.name,
                source_cache_path=shared_cache_path,
            )

        with base_runner._timed_event(event_logger, "priority_cache", experiment_name=experiment.name):
            radius_cache = base_runner._build_priority_radius_cache(
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
        variant_metadata = _resolve_variant_metadata(radius_cache, variant_columns)

    return PreparedVariantExperiment(
        experiment=experiment,
        experiment_dir=experiment_dir,
        training_imputation_counts=training_imputed,
        calibration_imputation_counts=calibration_imputed,
        heldout_imputation_counts=heldout_imputed,
        reference_frame=reference_frame,
        reference_tracking_rows=reference_tracking_rows,
        heldout_frame=heldout_frame,
        heldout_tracking_rows=filtered_heldout_tracking_rows,
        variant_metadata=variant_metadata,
        actual_labels=actual_labels,
        model_cap=model_cap,
        effective_cap=effective_cap,
        radius_cache=radius_cache,
    )


def _evaluate_prepared_experiment(
    *,
    config: base_runner.RunnerConfig,
    prepared: PreparedVariantExperiment,
    run_dir: Path,
    event_logger: Any,
) -> dict[str, Any]:
    experiment = prepared.experiment
    experiment_dir = prepared.experiment_dir

    with base_runner._timed_event(
        event_logger,
        "experiment_evaluate",
        experiment_name=experiment.name,
        model_name=experiment.model_name,
        cache_reuse_mode=prepared.radius_cache.reuse_mode,
    ):
        metric_frames: list[pd.DataFrame] = []
        prediction_frames: list[pd.DataFrame] = []
        context_frames: list[pd.DataFrame] = []

        if "mixture_learning" in experiment.schemes:
            mixture_indices = np.arange(prepared.reference_frame.shape[0], dtype=int)
            if mixture_indices.shape[0] > prepared.effective_cap:
                mixture_indices = base_runner._select_random_context_indices(
                    mixture_indices,
                    desired_count=prepared.effective_cap,
                    seed=base_runner._stable_seed(
                        config.random_seed,
                        experiment.name,
                        experiment.model_name,
                        "mixture_learning",
                    ),
                )
            prediction_frame, metric_frame, context_frame = _run_shared_context_scheme_across_variants(
                dataset_pair_name=experiment.name,
                model_name=experiment.model_name,
                scheme_name="mixture_learning",
                experiment=experiment,
                reference_frame=prepared.reference_frame,
                heldout_frame=prepared.heldout_frame,
                heldout_tracking_rows=prepared.heldout_tracking_rows,
                actual_labels=prepared.actual_labels,
                selected_reference_indices=mixture_indices,
                variant_metadata=prepared.variant_metadata,
            )
            prediction_frames.append(prediction_frame)
            metric_frames.append(metric_frame)
            context_frames.append(context_frame)

        if "independent_learning_scheme" in experiment.schemes:
            target_key = base_runner._normalize_group_key(experiment.heldout_target_ancestry_group)
            independent_indices = np.flatnonzero(
                prepared.reference_tracking_rows["ancestry_group"].astype(str).map(base_runner._normalize_group_key)
                == target_key
            ).astype(int, copy=False)
            if independent_indices.size == 0:
                raise ValueError("No same-ancestry reference rows were available for the Independent Learning Scheme.")
            if independent_indices.shape[0] > prepared.effective_cap:
                independent_indices = base_runner._select_random_context_indices(
                    independent_indices,
                    desired_count=prepared.effective_cap,
                    seed=base_runner._stable_seed(
                        config.random_seed,
                        experiment.name,
                        experiment.model_name,
                        "independent_learning_scheme",
                    ),
                )
            prediction_frame, metric_frame, context_frame = _run_shared_context_scheme_across_variants(
                dataset_pair_name=experiment.name,
                model_name=experiment.model_name,
                scheme_name="independent_learning_scheme",
                experiment=experiment,
                reference_frame=prepared.reference_frame,
                heldout_frame=prepared.heldout_frame,
                heldout_tracking_rows=prepared.heldout_tracking_rows,
                actual_labels=prepared.actual_labels,
                selected_reference_indices=independent_indices,
                variant_metadata=prepared.variant_metadata,
            )
            prediction_frames.append(prediction_frame)
            metric_frames.append(metric_frame)
            context_frames.append(context_frame)

        if "priority_function_curated_context" in experiment.schemes:
            prediction_frame, metric_frame, context_frame = _run_priority_context_scheme_across_variants(
                dataset_pair_name=experiment.name,
                model_name=experiment.model_name,
                experiment=experiment,
                reference_frame=prepared.reference_frame,
                reference_tracking_rows=prepared.reference_tracking_rows,
                heldout_frame=prepared.heldout_frame,
                heldout_tracking_rows=prepared.heldout_tracking_rows,
                actual_labels=prepared.actual_labels,
                radius_cache=prepared.radius_cache,
                context_cap=prepared.effective_cap,
                variant_metadata=prepared.variant_metadata,
            )
            prediction_frames.append(prediction_frame)
            metric_frames.append(metric_frame)
            context_frames.append(context_frame)

        if not metric_frames:
            raise ValueError(f"Experiment {experiment.name!r} produced no scheme outputs.")

        metric_frame = pd.concat(metric_frames, ignore_index=True).sort_values(
            ["plot_order", "variant_index"]
        ).reset_index(drop=True)
        metric_frame["model_context_cap_rows"] = int(prepared.model_cap.rows_cap)
        metric_frame["requested_max_context_rows"] = experiment.max_context_rows
        metric_frame["effective_max_context_rows"] = int(prepared.effective_cap)
        metric_frame["feature_count_per_task"] = 1 + len(experiment.ancestry_columns) + len(
            experiment.additional_feature_columns
        )
        metric_frame["reference_row_count"] = int(prepared.reference_frame.shape[0])
        metric_frame["heldout_subject_count"] = int(prepared.heldout_frame.shape[0])

        prediction_frame = pd.concat(prediction_frames, ignore_index=True)
        context_frame = pd.concat(context_frames, ignore_index=True)
        triplet_frame = _build_variant_auc_triplet_frame(metric_frame)

        prediction_path = experiment_dir / "heldout_variant_predictions.pkl"
        prediction_frame.to_pickle(prediction_path)
        prediction_csv_path = experiment_dir / "heldout_variant_predictions.csv.gz"
        prediction_frame.to_csv(prediction_csv_path, index=False)

        context_path = experiment_dir / "variant_context_summary.csv"
        context_frame.to_csv(context_path, index=False)
        metric_path = experiment_dir / "variant_scheme_metrics.csv"
        metric_frame.to_csv(metric_path, index=False)
        triplet_path = experiment_dir / "variant_auc_triplets.csv"
        triplet_frame.to_csv(triplet_path, index=False)

        figure_dir = run_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        file_name = _build_plot_file_name(
            experiment=experiment,
            model_name=experiment.model_name,
        )
        figure_path = figure_dir / file_name
        _plot_variant_task_boxplots(
            metric_frame=metric_frame,
            output_path=figure_path,
            seed=base_runner._stable_seed(config.random_seed, experiment.name, experiment.model_name, "boxplot"),
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
            "variant_count": len(prepared.variant_metadata),
            "feature_count_per_task": 1 + len(experiment.ancestry_columns) + len(experiment.additional_feature_columns),
            "variant_names": [variant.variant_name for variant in prepared.variant_metadata],
            "variant_dosage_fields": [variant.dosage_field for variant in prepared.variant_metadata],
            "plot_path": str(figure_path),
            "prediction_path": str(prediction_path),
            "prediction_csv_path": str(prediction_csv_path),
            "context_summary_path": str(context_path),
            "variant_scheme_metrics_path": str(metric_path),
            "variant_auc_triplets_path": str(triplet_path),
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
            "reference_row_count": int(prepared.reference_frame.shape[0]),
            "heldout_subject_count": int(prepared.heldout_frame.shape[0]),
            "training_imputation_counts": prepared.training_imputation_counts,
            "calibration_imputation_counts": prepared.calibration_imputation_counts,
            "heldout_imputation_counts": prepared.heldout_imputation_counts,
        }
        summary_path = experiment_dir / "summary.json"
        base_runner._write_json(summary_path, summary_payload)
    return {
        "experiment_name": experiment.name,
        "model_name": experiment.model_name,
        "experiment_dir": str(experiment_dir),
        "prediction_path": str(prediction_path),
        "prediction_csv_path": str(prediction_csv_path),
        "context_path": str(context_path),
        "metric_path": str(metric_path),
        "triplet_path": str(triplet_path),
        "summary_path": str(summary_path),
        "plot_path": str(figure_path),
        "priority_radius_cache_manifest_path": str(prepared.radius_cache.manifest_path),
        "metric_frame": metric_frame,
        "prediction_frame": prediction_frame,
        "context_frame": context_frame,
        "triplet_frame": triplet_frame,
        "summary_payload": summary_payload,
    }


def _build_run_notes(
    config: base_runner.RunnerConfig,
    run_dir: Path,
    experiment_results: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Tabular Foundation Model Variant-Task Run Notes",
        "",
        f"- Run directory: {run_dir}",
        f"- Config path: {config.config_path}",
        f"- Priority function path: {config.prio_function_path}",
        f"- Function name: {config.function_name}",
        f"- Random seed: {config.random_seed}",
        "",
        "## Outputs",
        "",
        "Each experiment directory contains per-subject predictions for every variant-task and scheme, one wide table of per-variant ROC-AUC triplets, one long table of scheme-by-variant metrics, and one context-summary CSV.",
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
        lines.append(f"- Variant count: {summary_payload['variant_count']}")
        lines.append(f"- Heldout subject count: {summary_payload['heldout_subject_count']}")
        lines.append(f"- Plot: {summary_payload['plot_path']}")
        lines.append(f"- Prediction table: {summary_payload['prediction_path']}")
        lines.append(f"- Variant ROC-AUC triplets: {summary_payload['variant_auc_triplets_path']}")
        lines.append(f"- Radius cache manifest: {summary_payload['priority_radius_cache_manifest_path']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_from_config_path(config_path: str | Path) -> Path:
    config = base_runner.load_config(config_path)
    run_dir = _create_run_directory(config)
    event_logger = base_runner._create_run_logger(run_dir)
    try:
        with base_runner._timed_event(
            event_logger,
            "run",
            run_dir=run_dir,
            config_path=config.config_path,
            experiment_count=len(config.experiments),
        ):
            base_runner._write_json(
                run_dir / "run_config.used.json",
                json.loads(config.config_path.read_text(encoding="utf-8")),
            )
            base_runner._log_event(event_logger, "run_config.written", path=run_dir / "run_config.used.json")

            shared_priority_cache_paths: dict[tuple[Any, ...], Path] = {}
            with base_runner._timed_event(event_logger, "priority_cache_phase", experiment_count=len(config.experiments)):
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

            with base_runner._timed_event(event_logger, "evaluation_phase", experiment_count=len(config.experiments)):
                experiment_results = [
                    _evaluate_prepared_experiment(
                        config=config,
                        prepared=prepared_experiment,
                        run_dir=run_dir,
                        event_logger=event_logger,
                    )
                    for prepared_experiment in prepared_experiments
                ]

            aggregate_prediction_frame = pd.concat(
                [result["prediction_frame"] for result in experiment_results],
                ignore_index=True,
            )
            aggregate_prediction_frame.to_pickle(run_dir / "heldout_variant_predictions.pkl")
            aggregate_prediction_frame.to_csv(run_dir / "heldout_variant_predictions.csv.gz", index=False)

            aggregate_context_frame = pd.concat(
                [result["context_frame"] for result in experiment_results],
                ignore_index=True,
            )
            aggregate_context_frame.to_csv(run_dir / "variant_context_summary.csv", index=False)

            aggregate_metric_frame = pd.concat(
                [result["metric_frame"] for result in experiment_results],
                ignore_index=True,
            )
            aggregate_metric_frame.to_csv(run_dir / "variant_scheme_metrics.csv", index=False)

            aggregate_triplet_frame = pd.concat(
                [result["triplet_frame"] for result in experiment_results],
                ignore_index=True,
            )
            aggregate_triplet_frame.to_csv(run_dir / "variant_auc_triplets.csv", index=False)

            summary_payload = {
                "run_dir": str(run_dir),
                "config_path": str(config.config_path),
                "priority_function_path": str(config.prio_function_path),
                "function_name": config.function_name,
                "random_seed": config.random_seed,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_events_log_path": str((run_dir / "run_events.log").resolve()),
                "aggregate_prediction_path": str((run_dir / "heldout_variant_predictions.pkl").resolve()),
                "aggregate_prediction_csv_path": str((run_dir / "heldout_variant_predictions.csv.gz").resolve()),
                "aggregate_context_summary_path": str((run_dir / "variant_context_summary.csv").resolve()),
                "aggregate_metric_path": str((run_dir / "variant_scheme_metrics.csv").resolve()),
                "aggregate_triplet_path": str((run_dir / "variant_auc_triplets.csv").resolve()),
                "experiment_summaries": [result["summary_payload"] for result in experiment_results],
            }
            base_runner._write_json(run_dir / "summary.json", summary_payload)
            (run_dir / "README_run_notes.md").write_text(
                _build_run_notes(config, run_dir, experiment_results),
                encoding="utf-8",
            )

        print(f"Wrote run outputs to {run_dir}")
        return run_dir
    finally:
        base_runner._close_run_logger(event_logger)