from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import argparse
import hashlib
import json
import re

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_HELDOUT_MODEL_PREDICTIONS_FILE_NAME = "heldout_model_predictions.pkl"
DEFAULT_HELDOUT_MODEL_PREDICTIONS_BY_MODEL_DIR_NAME = "heldout_model_predictions_by_model"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_OUTPUT_FILE_NAME_TEMPLATE = "heldout_auc_ci_{ancestry_group}.csv"
MODEL_DISPLAY_NAME_BY_SLUG = {
    "priority_function": "Scheme Discovered by LLM",
    "independent_learning_scheme": "Independent Learning Scheme",
    "mixture_learning": "Mixture Learning Scheme",
    "tl_gdes": "TL-GDES Scheme",
    "tl_pr": "TL-PR Scheme",
}
MODEL_PLOT_ORDER_BY_SLUG = {
    "priority_function": 0,
    "independent_learning_scheme": 1,
    "mixture_learning": 2,
    "tl_gdes": 3,
    "tl_pr": 4,
}
REQUIRED_PREDICTION_COLUMNS = (
    "ancestry_group",
    "label",
    "model_name",
    "model_slug",
    "risk_score",
)


@dataclass(frozen=True)
class HeldoutAucCiConfig:
    precomputed_directory: Path
    target_ancestry_group: str
    ci_level: float
    bootstrap_iterations: int
    random_seed: int
    output_file_name: str


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _validate_precomputed_directory(path: Path) -> Path:
    if not path.exists():
        raise ValueError(
            "precomputed_directory does not exist. "
            f"expected an existing directory but got {path}"
        )
    if not path.is_dir():
        raise ValueError(
            "precomputed_directory must point to a directory. "
            f"got {path}"
        )
    return path


def _normalize_ci_level(raw_value: Any) -> float:
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ci_level must be a number such as 0.95 or 95.") from exc

    if parsed > 1.0:
        parsed /= 100.0
    if not 0.0 < parsed < 1.0:
        raise ValueError("ci_level must be between 0 and 1, or between 0 and 100.")
    return parsed


def _normalize_bootstrap_iterations(raw_value: Any) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bootstrap_iterations must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError("bootstrap_iterations must be at least 1.")
    return parsed


def _normalize_output_file_name(raw_value: Any, *, ancestry_group: str) -> str:
    if raw_value is None:
        file_name = DEFAULT_OUTPUT_FILE_NAME_TEMPLATE.format(
            ancestry_group=_slugify_token(ancestry_group)
        )
    elif not isinstance(raw_value, str):
        raise ValueError("output_file_name must be a string.")
    else:
        file_name = raw_value.strip()
    if not file_name:
        raise ValueError("output_file_name must not be empty.")
    path = Path(file_name)
    if path.is_absolute() or path.name != file_name or file_name in {".", ".."}:
        raise ValueError("output_file_name must be a file name, not a path.")
    if path.suffix.lower() != ".csv":
        raise ValueError("output_file_name must end with .csv.")
    return file_name


def load_config(config_path: str | Path) -> HeldoutAucCiConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("The config file must contain a JSON object.")

    if "precomputed_directory" not in raw_config:
        raise ValueError("Missing required config field 'precomputed_directory'.")
    if "target_ancestry_group" not in raw_config:
        raise ValueError("Missing required config field 'target_ancestry_group'.")
    if "ci_level" not in raw_config:
        raise ValueError("Missing required config field 'ci_level'.")

    base_dir = resolved_config_path.parent
    target_ancestry_group = str(raw_config["target_ancestry_group"]).strip()
    if not target_ancestry_group:
        raise ValueError("target_ancestry_group must not be empty.")

    precomputed_directory = _validate_precomputed_directory(
        _resolve_path(base_dir, str(raw_config["precomputed_directory"]))
    )

    return HeldoutAucCiConfig(
        precomputed_directory=precomputed_directory,
        target_ancestry_group=target_ancestry_group,
        ci_level=_normalize_ci_level(raw_config["ci_level"]),
        bootstrap_iterations=_normalize_bootstrap_iterations(
            raw_config.get("bootstrap_iterations", DEFAULT_BOOTSTRAP_ITERATIONS)
        ),
        random_seed=int(raw_config.get("random_seed", 0)),
        output_file_name=_normalize_output_file_name(
            raw_config.get("output_file_name"),
            ancestry_group=target_ancestry_group,
        ),
    )


def _slugify_token(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"Could not derive a file-safe token from {value!r}.")
    return slug


def _normalize_group_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _resolve_target_ancestry_group(
    *,
    requested_group: str,
    available_groups: Sequence[str],
) -> str:
    normalized_requested = _normalize_group_key(requested_group)
    matches = [
        candidate
        for candidate in available_groups
        if _normalize_group_key(candidate) == normalized_requested
    ]
    if not matches:
        raise ValueError(
            "target_ancestry_group did not match any ancestry_group values in the prediction files. "
            f"requested={requested_group!r} available={sorted(set(available_groups))!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "target_ancestry_group matched multiple ancestry labels after normalization: "
            f"requested={requested_group!r} matches={matches!r}"
        )
    return matches[0]


def _safe_roc_auc(labels: np.ndarray, risk_scores: np.ndarray) -> float:
    if labels.shape[0] != risk_scores.shape[0]:
        raise ValueError(
            f"ROC AUC requires equal-length arrays: labels={labels.shape[0]} scores={risk_scores.shape[0]}."
        )
    if labels.shape[0] == 0:
        raise ValueError("ROC AUC requires at least one scored row.")
    if np.unique(labels).size < 2:
        raise ValueError("ROC AUC requires both classes in the labels.")
    auc = float(roc_auc_score(labels, risk_scores))
    if not np.isfinite(auc):
        raise ValueError("ROC AUC must be finite.")
    return auc


def _bootstrap_auc_confidence_interval(
    *,
    labels: np.ndarray,
    risk_scores: np.ndarray,
    ci_level: float,
    bootstrap_iterations: int,
    random_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    sample_count = labels.shape[0]
    bootstrap_scores: list[float] = []

    for _ in range(bootstrap_iterations):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        sampled_labels = labels[sample_indices]
        if np.unique(sampled_labels).size < 2:
            continue
        bootstrap_scores.append(_safe_roc_auc(sampled_labels, risk_scores[sample_indices]))

    if not bootstrap_scores:
        raise ValueError(
            "Could not compute a bootstrap ROC AUC confidence interval because every resample "
            "contained only one class."
        )

    alpha = 1.0 - ci_level
    lower, upper = np.quantile(
        np.asarray(bootstrap_scores, dtype=float),
        [alpha / 2.0, 1.0 - (alpha / 2.0)],
        method="linear",
    )
    return float(lower), float(upper)


def _stable_model_seed(*, base_seed: int, model_name: str, ancestry_group: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\n{model_name}\n{ancestry_group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def _display_model_name(*, model_name: str, model_slug: str) -> str:
    return MODEL_DISPLAY_NAME_BY_SLUG.get(model_slug, model_name)


def _model_plot_order(model_slug: str) -> int:
    return MODEL_PLOT_ORDER_BY_SLUG.get(model_slug, 100)


def _validate_prediction_frame(frame: pd.DataFrame, *, path: Path) -> None:
    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Prediction file {path} is missing columns: {missing_columns!r}")
    if frame.empty:
        raise ValueError(f"Prediction file {path} is empty.")


def _load_prediction_frames(precomputed_directory: Path) -> list[tuple[Path, pd.DataFrame]]:
    by_model_dir = precomputed_directory / DEFAULT_HELDOUT_MODEL_PREDICTIONS_BY_MODEL_DIR_NAME
    if by_model_dir.exists():
        prediction_paths = sorted(path for path in by_model_dir.glob("*.pkl") if path.is_file())
        if not prediction_paths:
            raise ValueError(f"No prediction pickle files were found in {by_model_dir}.")
        loaded_frames: list[tuple[Path, pd.DataFrame]] = []
        for prediction_path in prediction_paths:
            prediction_frame = pd.read_pickle(prediction_path)
            if not isinstance(prediction_frame, pd.DataFrame):
                raise ValueError(f"Prediction file {prediction_path} did not contain a pandas DataFrame.")
            _validate_prediction_frame(prediction_frame, path=prediction_path)
            loaded_frames.append((prediction_path, prediction_frame.reset_index(drop=True)))
        return loaded_frames

    aggregate_path = precomputed_directory / DEFAULT_HELDOUT_MODEL_PREDICTIONS_FILE_NAME
    if not aggregate_path.exists():
        raise ValueError(
            "Could not find heldout model predictions under either "
            f"{by_model_dir} or {aggregate_path}."
        )
    aggregate_frame = pd.read_pickle(aggregate_path)
    if not isinstance(aggregate_frame, pd.DataFrame):
        raise ValueError(f"Prediction file {aggregate_path} did not contain a pandas DataFrame.")
    _validate_prediction_frame(aggregate_frame, path=aggregate_path)
    loaded_frames = []
    for model_name, model_frame in aggregate_frame.groupby("model_name", sort=True):
        loaded_frames.append((aggregate_path, model_frame.reset_index(drop=True)))
    return loaded_frames


def _build_auc_ci_summary(config: HeldoutAucCiConfig) -> pd.DataFrame:
    loaded_frames = _load_prediction_frames(config.precomputed_directory)
    available_groups = sorted(
        {
            str(group)
            for _, frame in loaded_frames
            for group in frame["ancestry_group"].dropna().astype(str).tolist()
        }
    )
    ancestry_group = _resolve_target_ancestry_group(
        requested_group=config.target_ancestry_group,
        available_groups=available_groups,
    )

    summary_rows: list[dict[str, Any]] = []
    for prediction_path, prediction_frame in loaded_frames:
        group_frame = prediction_frame.loc[
            prediction_frame["ancestry_group"].astype(str) == ancestry_group
        ].reset_index(drop=True)
        if group_frame.empty:
            continue

        original_model_name = str(group_frame.iloc[0]["model_name"])
        model_slug = str(group_frame.iloc[0]["model_slug"])
        labels = group_frame["label"].to_numpy(dtype=float, copy=False)
        risk_scores = group_frame["risk_score"].to_numpy(dtype=float, copy=False)
        auc_roc = _safe_roc_auc(labels, risk_scores)
        ci_lower, ci_hi = _bootstrap_auc_confidence_interval(
            labels=labels,
            risk_scores=risk_scores,
            ci_level=config.ci_level,
            bootstrap_iterations=config.bootstrap_iterations,
            random_seed=_stable_model_seed(
                base_seed=config.random_seed,
                model_name=original_model_name,
                ancestry_group=ancestry_group,
            ),
        )
        summary_rows.append(
            {
                "model_name": _display_model_name(
                    model_name=original_model_name,
                    model_slug=model_slug,
                ),
                "model_slug": model_slug,
                "ancestry_group": ancestry_group,
                "subject_count": int(group_frame.shape[0]),
                "auc_roc": auc_roc,
                "ci_lower": ci_lower,
                "ci_hi": ci_hi,
                "ci_level": config.ci_level,
                "bootstrap_iterations": config.bootstrap_iterations,
                "prediction_pickle_path": str(prediction_path),
                "plot_order": _model_plot_order(model_slug),
            }
        )

    if not summary_rows:
        raise ValueError(
            f"No heldout prediction rows were found for ancestry_group={ancestry_group!r}."
        )

    return pd.DataFrame(summary_rows).sort_values(["plot_order", "model_name"]).reset_index(drop=True)


def output_path_for_config(config: HeldoutAucCiConfig) -> Path:
    return config.precomputed_directory / config.output_file_name


def plot_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(".png")


def _write_summary_frame(summary_frame: pd.DataFrame, output_path: Path) -> None:
    summary_frame.drop(columns=["plot_order"]).to_csv(output_path, index=False)


def _wrap_plot_label(label: str) -> str:
    words = label.split()
    if len(words) <= 2:
        return label

    best_label = label
    best_width = len(label)
    for split_index in range(1, len(words)):
        first_line = " ".join(words[:split_index])
        second_line = " ".join(words[split_index:])
        candidate_width = max(len(first_line), len(second_line))
        if candidate_width < best_width:
            best_width = candidate_width
            best_label = f"{first_line}\n{second_line}"
    return best_label


def _plot_summary_from_csv(csv_path: Path, plot_path: Path) -> None:
    summary_frame = pd.read_csv(csv_path)
    if summary_frame.empty:
        raise ValueError(f"Summary CSV {csv_path} is empty.")

    lower_errors = (summary_frame["auc_roc"] - summary_frame["ci_lower"]).to_numpy(dtype=float)
    upper_errors = (summary_frame["ci_hi"] - summary_frame["auc_roc"]).to_numpy(dtype=float)
    positions = np.arange(summary_frame.shape[0], dtype=float)
    figure_width = max(9.0, 2.2 * float(summary_frame.shape[0]))
    fig, ax = plt.subplots(figsize=(figure_width, 6.0))
    ax.errorbar(
        positions,
        summary_frame["auc_roc"].to_numpy(dtype=float),
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        color="#1f4e79",
        ecolor="#7a1f1f",
        elinewidth=2.0,
        capsize=6.0,
        markersize=8.0,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_wrap_plot_label(label) for label in summary_frame["model_name"].tolist()],
        rotation=0,
        ha="center",
        fontsize=13,
    )
    ax.set_xlabel("Scheme Name", fontsize=16)
    ax.set_ylabel("ROC AUC", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=0.3)
    y_min = max(0.0, float(summary_frame["ci_lower"].min()) - 0.05)
    y_max = min(1.0, float(summary_frame["ci_hi"].max()) + 0.05)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def write_auc_ci_summary(config: HeldoutAucCiConfig) -> Path:
    summary_frame = _build_auc_ci_summary(config)
    output_path = output_path_for_config(config)
    plot_path = plot_path_for_output(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output_path}. "
            "Choose a different output_file_name in the config."
        )
    if plot_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing plot file: {plot_path}. "
            "Choose a different output_file_name in the config."
        )
    _write_summary_frame(summary_frame, output_path)
    _plot_summary_from_csv(output_path, plot_path)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute bootstrap ROC AUC confidence intervals for all heldout model prediction "
            "pickles within one ancestry group."
        )
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to a JSON config file with precomputed_directory, target_ancestry_group, and ci_level.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config_path)
    summary_frame = _build_auc_ci_summary(config)
    output_path = output_path_for_config(config)
    plot_path = plot_path_for_output(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output_path}. "
            "Choose a different output_file_name in the config."
        )
    if plot_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing plot file: {plot_path}. "
            "Choose a different output_file_name in the config."
        )
    _write_summary_frame(summary_frame, output_path)
    _plot_summary_from_csv(output_path, plot_path)
    print(summary_frame.drop(columns=["plot_order"]).to_string(index=False))
    print(f"output_path={output_path}")
    print(f"plot_path={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())