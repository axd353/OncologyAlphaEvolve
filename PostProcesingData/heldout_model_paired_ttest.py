from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_HELDOUT_MODEL_PREDICTIONS_FILE_NAME = "heldout_model_predictions.pkl"
DEFAULT_HELDOUT_MODEL_PREDICTIONS_BY_MODEL_DIR_NAME = "heldout_model_predictions_by_model"
DEFAULT_OUTPUT_FILE_NAME_TEMPLATE = "heldout_paired_ttest_{ancestry_group}.md"
PRIORITY_FUNCTION_MODEL_SLUG = "priority_function"
MODEL_DISPLAY_NAME_BY_SLUG = {
    "priority_function": "Scheme Discovered by LLM",
    "independent_learning_scheme": "Independent Learning Scheme",
    "mixture_learning": "Mixture Learning Scheme",
    "tl_gdes": "TL-GDES Scheme",
    "tl_pr": "TL-PR Scheme",
}
MODEL_ORDER_BY_SLUG = {
    "priority_function": 0,
    "independent_learning_scheme": 1,
    "mixture_learning": 2,
    "tl_gdes": 3,
    "tl_pr": 4,
}
REQUIRED_PREDICTION_COLUMNS = (
    "heldout_subject_index",
    "heldout_output_pickle_name",
    "heldout_output_row_number",
    "source_pickle_name",
    "source_row_number",
    "ancestry_group",
    "label",
    "model_name",
    "model_slug",
    "risk_probability",
)
SUBJECT_IDENTITY_COLUMNS = (
    "heldout_subject_index",
    "heldout_output_pickle_name",
    "heldout_output_row_number",
    "source_pickle_name",
    "source_row_number",
    "ancestry_group",
    "label",
)


@dataclass(frozen=True)
class HeldoutPairedTTestConfig:
    precomputed_directory: Path
    target_ancestry_group: str
    p_value_threshold: float
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


def _slugify_token(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"Could not derive a file-safe token from {value!r}.")
    return slug


def _normalize_group_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _normalize_probability_threshold(raw_value: Any) -> float:
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_value_threshold must be a number such as 0.05.") from exc
    if not 0.0 < parsed < 1.0:
        raise ValueError("p_value_threshold must be strictly between 0 and 1.")
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
    if path.suffix.lower() != ".md":
        raise ValueError("output_file_name must end with .md.")
    return file_name


def load_config(config_path: str | Path) -> HeldoutPairedTTestConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("The config file must contain a JSON object.")

    if "precomputed_directory" not in raw_config:
        raise ValueError("Missing required config field 'precomputed_directory'.")
    if "target_ancestry_group" not in raw_config:
        raise ValueError("Missing required config field 'target_ancestry_group'.")
    if "p_value_threshold" not in raw_config:
        raise ValueError("Missing required config field 'p_value_threshold'.")

    base_dir = resolved_config_path.parent
    target_ancestry_group = str(raw_config["target_ancestry_group"]).strip()
    if not target_ancestry_group:
        raise ValueError("target_ancestry_group must not be empty.")

    return HeldoutPairedTTestConfig(
        precomputed_directory=_validate_precomputed_directory(
            _resolve_path(base_dir, str(raw_config["precomputed_directory"]))
        ),
        target_ancestry_group=target_ancestry_group,
        p_value_threshold=_normalize_probability_threshold(raw_config["p_value_threshold"]),
        output_file_name=_normalize_output_file_name(
            raw_config.get("output_file_name"),
            ancestry_group=target_ancestry_group,
        ),
    )


def _display_model_name(*, model_name: str, model_slug: str) -> str:
    return MODEL_DISPLAY_NAME_BY_SLUG.get(model_slug, model_name)


def _model_order(model_slug: str) -> int:
    return MODEL_ORDER_BY_SLUG.get(model_slug, 100)


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
    loaded_frames: list[tuple[Path, pd.DataFrame]] = []
    for model_name, model_frame in aggregate_frame.groupby("model_name", sort=True):
        del model_name
        loaded_frames.append((aggregate_path, model_frame.reset_index(drop=True)))
    return loaded_frames


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


def _subject_identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["heldout_subject_index"].duplicated().any():
        duplicates = frame.loc[frame["heldout_subject_index"].duplicated(), "heldout_subject_index"].tolist()
        raise ValueError(f"Duplicate heldout_subject_index values found: {duplicates[:10]!r}")
    return (
        frame.loc[:, list(SUBJECT_IDENTITY_COLUMNS)]
        .sort_values("heldout_subject_index")
        .reset_index(drop=True)
    )


def _correct_label_probability(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=float, copy=False)
    risk_probability = frame["risk_probability"].to_numpy(dtype=float, copy=False)
    label_values = set(np.unique(labels).tolist())
    if not label_values.issubset({0.0, 1.0}):
        raise ValueError(
            "Labels must be binary 0/1 for correct-label probability computation. "
            f"got {sorted(label_values)!r}"
        )
    return np.where(labels == 1.0, risk_probability, 1.0 - risk_probability)


def _extract_model_frame_for_ancestry(
    *,
    prediction_frame: pd.DataFrame,
    ancestry_group: str,
) -> pd.DataFrame:
    group_frame = prediction_frame.loc[
        prediction_frame["ancestry_group"].astype(str) == ancestry_group
    ].copy()
    if group_frame.empty:
        raise ValueError(f"No rows were found for ancestry_group={ancestry_group!r}.")
    return group_frame.sort_values("heldout_subject_index").reset_index(drop=True)


def _paired_ttest_greater(
    first_values: np.ndarray,
    second_values: np.ndarray,
) -> tuple[float, float, float]:
    if first_values.shape != second_values.shape:
        raise ValueError(
            "Paired t-test requires equal-length arrays. "
            f"got {first_values.shape[0]} and {second_values.shape[0]}."
        )
    if first_values.shape[0] < 2:
        raise ValueError("Paired t-test requires at least two matched heldout subjects.")

    differences = np.asarray(first_values - second_values, dtype=float)
    mean_difference = float(np.mean(differences))
    sample_std = float(np.std(differences, ddof=1))
    if sample_std == 0.0:
        if mean_difference > 0.0:
            return float("inf"), 0.0, mean_difference
        return 0.0, 1.0, mean_difference

    t_statistic = mean_difference / (sample_std / np.sqrt(differences.shape[0]))
    p_value = float(stats.t.sf(t_statistic, df=differences.shape[0] - 1))
    return float(t_statistic), p_value, mean_difference


def _format_p_value(p_value: float) -> str:
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.6f}"


def _decision_text(*, p_value: float, p_value_threshold: float) -> str:
    if p_value <= p_value_threshold:
        return "Reject equal-performance null; accept higher-than-baseline alternative"
    return "Do not reject equal-performance null"


def _build_results_table(config: HeldoutPairedTTestConfig) -> pd.DataFrame:
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

    frames_by_slug: dict[str, pd.DataFrame] = {}
    source_path_by_slug: dict[str, Path] = {}
    for prediction_path, prediction_frame in loaded_frames:
        model_slug = str(prediction_frame.iloc[0]["model_slug"])
        frames_by_slug[model_slug] = _extract_model_frame_for_ancestry(
            prediction_frame=prediction_frame,
            ancestry_group=ancestry_group,
        )
        source_path_by_slug[model_slug] = prediction_path

    if PRIORITY_FUNCTION_MODEL_SLUG not in frames_by_slug:
        raise ValueError(
            "Could not find the priority-function predictions in the heldout model prediction files."
        )

    reference_frame = frames_by_slug[PRIORITY_FUNCTION_MODEL_SLUG]
    reference_identity = _subject_identity_frame(reference_frame)
    reference_probabilities = _correct_label_probability(reference_frame)
    reference_model_name = _display_model_name(
        model_name=str(reference_frame.iloc[0]["model_name"]),
        model_slug=PRIORITY_FUNCTION_MODEL_SLUG,
    )

    result_rows: list[dict[str, Any]] = []
    for model_slug, baseline_frame in sorted(
        frames_by_slug.items(),
        key=lambda item: (_model_order(item[0]), item[0]),
    ):
        if model_slug == PRIORITY_FUNCTION_MODEL_SLUG:
            continue

        baseline_identity = _subject_identity_frame(baseline_frame)
        if not reference_identity.equals(baseline_identity):
            raise ValueError(
                "Heldout subject matching failed between the priority-function model and baseline "
                f"{model_slug!r}. The subject identity columns did not align exactly."
            )

        baseline_probabilities = _correct_label_probability(baseline_frame)
        t_statistic, p_value, mean_difference = _paired_ttest_greater(
            reference_probabilities,
            baseline_probabilities,
        )
        baseline_model_name = _display_model_name(
            model_name=str(baseline_frame.iloc[0]["model_name"]),
            model_slug=model_slug,
        )
        result_rows.append(
            {
                "reference_model": reference_model_name,
                "alternate_baseline": baseline_model_name,
                "ancestry_group": ancestry_group,
                "subject_count": int(reference_identity.shape[0]),
                "llm_mean_correct_label_probability": float(np.mean(reference_probabilities)),
                "baseline_mean_correct_label_probability": float(np.mean(baseline_probabilities)),
                "mean_difference": mean_difference,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "p_value_threshold": config.p_value_threshold,
                "result": _decision_text(
                    p_value=p_value,
                    p_value_threshold=config.p_value_threshold,
                ),
                "baseline_prediction_pickle_path": str(source_path_by_slug[model_slug]),
            }
        )

    if not result_rows:
        raise ValueError("No alternate baseline models were available for comparison.")

    return pd.DataFrame(result_rows)


def output_path_for_config(config: HeldoutPairedTTestConfig) -> Path:
    return config.precomputed_directory / config.output_file_name


def _markdown_table_row(values: Sequence[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _build_markdown_report(
    *,
    config: HeldoutPairedTTestConfig,
    results_frame: pd.DataFrame,
) -> str:
    ancestry_group = str(results_frame.iloc[0]["ancestry_group"])
    subject_count = int(results_frame.iloc[0]["subject_count"])
    reference_model = str(results_frame.iloc[0]["reference_model"])
    lines = [
        f"# Paired t-test for {ancestry_group}",
        "",
        f"- Precomputed directory: {config.precomputed_directory}",
        f"- Reference model: {reference_model}",
        f"- Target ancestry group: {ancestry_group}",
        f"- Matched heldout subjects: {subject_count}",
        "- Per-subject score: correct-label probability, defined as `risk_probability` when `label == 1` and `1 - risk_probability` when `label == 0`.",
        "- Null hypothesis: mean correct-label probability is equal between the reference model and the alternate baseline.",
        "- Alternative hypothesis: the reference model has higher mean correct-label probability than the alternate baseline.",
        f"- Decision rule: reject the null when p-value <= {config.p_value_threshold:.6f}.",
        "",
        _markdown_table_row(
            [
                "Alternate baseline",
                "Matched subjects",
                "Mean diff (LLM - baseline)",
                "p-value",
                "Result",
            ]
        ),
        _markdown_table_row(["---", "---:", "---:", "---:", "---"]),
    ]
    for result in results_frame.itertuples(index=False):
        lines.append(
            _markdown_table_row(
                [
                    str(result.alternate_baseline),
                    str(int(result.subject_count)),
                    f"{float(result.mean_difference):.6f}",
                    _format_p_value(float(result.p_value)),
                    str(result.result),
                ]
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The subject pairing is validated using the heldout subject index plus the heldout/source row identity columns written by the evaluator."
    )
    lines.append(
        "- This test uses the saved `risk_probability` values exactly as requested."
    )
    lines.append(
        "- For ridge-style baselines, `risk_probability` is a sigmoid transform of `risk_score`, so the test is on that bounded convenience value rather than on a separately calibrated absolute probability."
    )
    return "\n".join(lines) + "\n"


def write_markdown_report(config: HeldoutPairedTTestConfig) -> Path:
    results_frame = _build_results_table(config)
    output_path = output_path_for_config(config)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output_path}. "
            "Choose a different output_file_name in the config."
        )
    markdown = _build_markdown_report(config=config, results_frame=results_frame)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Scheme Discovered by LLM against each alternate baseline using a one-sided "
            "paired t-test on heldout correct-label probability within one ancestry group."
        )
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help=(
            "Path to a JSON config file with precomputed_directory, target_ancestry_group, "
            "and p_value_threshold."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config_path)
    results_frame = _build_results_table(config)
    output_path = output_path_for_config(config)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output_path}. "
            "Choose a different output_file_name in the config."
        )
    output_path.write_text(
        _build_markdown_report(config=config, results_frame=results_frame),
        encoding="utf-8",
    )
    print(results_frame.loc[:, ["alternate_baseline", "p_value", "result"]].to_string(index=False))
    print(f"output_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())