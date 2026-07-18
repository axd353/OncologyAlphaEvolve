from __future__ import annotations

"""Plot sequential FunSearch sampler funnel conversion rates.

Inputs
======
The required input is one FunSearch run directory that has already been processed
by `PostProcesingData.funsearch_run_postprocess`. The plotter reads these pickle
files from the run directory:

* `sampler_completed_priority_function_counts.pkl`
* `sampler_validated_priority_function_counts.pkl`
* `sampler_evaluation_completed_counts.pkl`
* `sampler_island_best_improvement_counts.pkl`

The first pickle must include `configured_candidates_per_island_per_cycle`, which
is written by the current post-processor from `config.used.json`.

Rate names
==========
Use `--rates` to choose one or more adjacent funnel rates. Use `all` to plot all
four. The available rate names are:

* `completed`: completed / configured
* `validated`: validated / completed
* `evaluated`: evaluation completed / validated
* `improved`: island-best improvements / evaluation completed

Important interpretation note: the sampler can retry several LLM calls inside
one configured candidate slot. Therefore `completed / configured` may exceed 1.0
when many retry attempts return non-empty completions. The plotter intentionally
keeps this visible instead of clipping the rate.

Outputs
=======
The output is a matplotlib figure and a DataFrame of per-cycle counts and rates.
By default the CLI saves both PNG and PDF files in the run directory:

* `sampler_funnel_conversion_rates.png`
* `sampler_funnel_conversion_rates.pdf`

Use `--save-formats png`, `--save-formats pdf`, or `--save-formats png pdf` to
control saved formats. Use `--output-stem` to change the base output filename.
Use `--no-save` to compute and print the rate table without writing image files.

Example
=======
```bash
PYTHONPATH=$PWD python -m PostProcesingData.funsearch_funnel_conversion_plots \
  prio_func_disc_runs/oracle_priority_20260716_050704 \
  --rates completed validated evaluated improved \
  --save-formats png pdf
```
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from PostProcesingData.funsearch_run_postprocess import COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE
    from PostProcesingData.funsearch_run_postprocess import EVALUATION_COMPLETED_COUNTS_PICKLE
    from PostProcesingData.funsearch_run_postprocess import ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE
    from PostProcesingData.funsearch_run_postprocess import VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE
except ModuleNotFoundError:
    from funsearch_run_postprocess import COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE
    from funsearch_run_postprocess import EVALUATION_COMPLETED_COUNTS_PICKLE
    from funsearch_run_postprocess import ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE
    from funsearch_run_postprocess import VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE

DEFAULT_OUTPUT_STEM = "sampler_funnel_conversion_rates"


@dataclass(frozen=True)
class ConversionRateSpec:
    """Definition of one adjacent funnel conversion rate."""

    name: str
    numerator_column: str
    denominator_column: str
    rate_column: str
    label: str


RATE_SPECS: dict[str, ConversionRateSpec] = {
    "completed": ConversionRateSpec(
        name="completed",
        numerator_column="completed_priority_function_count",
        denominator_column="configured_candidate_slot_count",
        rate_column="completed_over_configured_rate",
        label="Completed / configured",
    ),
    "validated": ConversionRateSpec(
        name="validated",
        numerator_column="validated_priority_function_count",
        denominator_column="completed_priority_function_count",
        rate_column="validated_over_completed_rate",
        label="Validated / completed",
    ),
    "evaluated": ConversionRateSpec(
        name="evaluated",
        numerator_column="evaluation_completed_count",
        denominator_column="validated_priority_function_count",
        rate_column="evaluated_over_validated_rate",
        label="Evaluated / validated",
    ),
    "improved": ConversionRateSpec(
        name="improved",
        numerator_column="island_best_improvement_count",
        denominator_column="evaluation_completed_count",
        rate_column="improved_over_evaluated_rate",
        label="Island-best improvements / evaluated",
    ),
}


@dataclass(frozen=True)
class FunnelPlotOutputs:
    """Result returned by the plotting entry point."""

    figure_paths: tuple[Path, ...]
    rate_table: pd.DataFrame


def _normalize_run_dir(run_dir: str | Path) -> Path:
    run_dir_path = Path(run_dir).expanduser().resolve()
    if not run_dir_path.exists() and not Path(run_dir).expanduser().is_absolute():
        repo_relative_path = (Path(__file__).resolve().parent.parent / run_dir).resolve()
        if repo_relative_path.exists():
            run_dir_path = repo_relative_path
    if not run_dir_path.exists() or not run_dir_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir_path}")
    return run_dir_path


def _read_pickle(run_dir: Path, pickle_name: str) -> pd.DataFrame:
    pickle_path = run_dir / pickle_name
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Missing {pickle_path}. Run PostProcesingData.funsearch_run_postprocess first."
        )
    return pd.read_pickle(pickle_path)


def normalize_rate_names(rate_names: list[str] | tuple[str, ...]) -> list[str]:
    """Validate CLI/API rate names and expand `all`.

    Input:
        rate_names: One or more names from `RATE_SPECS`, or a single `all`.

    Output:
        Ordered list of concrete rate names to plot.
    """

    if not rate_names or "all" in rate_names:
        return list(RATE_SPECS)
    unknown_names = [rate_name for rate_name in rate_names if rate_name not in RATE_SPECS]
    if unknown_names:
        raise ValueError(
            f"Unknown rate names: {unknown_names}. Valid names: {sorted(RATE_SPECS)} or all."
        )
    deduped_names: list[str] = []
    for rate_name in rate_names:
        if rate_name not in deduped_names:
            deduped_names.append(rate_name)
    return deduped_names


def load_conversion_rate_inputs(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four post-processed funnel count DataFrames.

    Input:
        run_dir: FunSearch run directory containing the post-processing pickles.

    Output:
        Tuple of completed, validated, evaluated, and improvement count frames.
    """

    normalized_run_dir = _normalize_run_dir(run_dir)
    return (
        _read_pickle(normalized_run_dir, COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE),
        _read_pickle(normalized_run_dir, VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE),
        _read_pickle(normalized_run_dir, EVALUATION_COMPLETED_COUNTS_PICKLE),
        _read_pickle(normalized_run_dir, ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE),
    )


def build_cycle_funnel_counts(run_dir: str | Path) -> pd.DataFrame:
    """Aggregate island-level funnel counts into one row per cycle.

    Input:
        run_dir: FunSearch run directory containing post-processing pickles.

    Output:
        DataFrame with one row per cycle and these count columns:
        `configured_candidate_slot_count`, `completed_priority_function_count`,
        `validated_priority_function_count`, `evaluation_completed_count`, and
        `island_best_improvement_count`.
    """

    completed_counts, validated_counts, evaluation_counts, improvement_counts = load_conversion_rate_inputs(
        run_dir
    )
    completed_by_cycle = (
        completed_counts.assign(
            configured_candidate_slot_count=completed_counts[
                "configured_candidates_per_island_per_cycle"
            ]
        )
        .groupby("cycle_index", as_index=False)[
            ["configured_candidate_slot_count", "completed_priority_function_count"]
        ]
        .sum()
    )
    validated_by_cycle = validated_counts.groupby("cycle_index", as_index=False)[
        "validated_priority_function_count"
    ].sum()
    evaluation_by_cycle = evaluation_counts.groupby("cycle_index", as_index=False)[
        "evaluation_completed_count"
    ].sum()
    return (
        completed_by_cycle.merge(validated_by_cycle, on="cycle_index", how="outer")
        .merge(evaluation_by_cycle, on="cycle_index", how="outer")
        .merge(improvement_counts, on="cycle_index", how="outer")
        .fillna(0)
        .sort_values("cycle_index")
        .reset_index(drop=True)
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def compute_conversion_rates(cycle_counts: pd.DataFrame) -> pd.DataFrame:
    """Add adjacent funnel conversion-rate columns to cycle counts.

    Input:
        cycle_counts: Output from `build_cycle_funnel_counts(...)`.

    Output:
        Copy of `cycle_counts` with one extra rate column per `RATE_SPECS` entry.
    """

    rate_table = cycle_counts.copy()
    for rate_spec in RATE_SPECS.values():
        rate_table[rate_spec.rate_column] = [
            _safe_rate(float(numerator), float(denominator))
            for numerator, denominator in zip(
                rate_table[rate_spec.numerator_column],
                rate_table[rate_spec.denominator_column],
            )
        ]
    return rate_table


def plot_conversion_rates(
    rate_table: pd.DataFrame,
    *,
    rates: list[str] | tuple[str, ...] = ("all",),
    title: str = "FunSearch sampler funnel conversion rates",
) -> tuple[plt.Figure, plt.Axes]:
    """Create a line plot for selected adjacent funnel rates.

    Input:
        rate_table: Output from `compute_conversion_rates(...)`.
        rates: One or more rate names: `completed`, `validated`, `evaluated`,
            `improved`, or `all`.
        title: Plot title.

    Output:
        Matplotlib `(figure, axes)` pair. The caller owns saving or closing it.
    """

    selected_rate_names = normalize_rate_names(list(rates))
    figure, axes = plt.subplots(figsize=(9, 5.5))
    for rate_name in selected_rate_names:
        rate_spec = RATE_SPECS[rate_name]
        axes.plot(
            rate_table["cycle_index"],
            rate_table[rate_spec.rate_column],
            marker="o",
            linewidth=2.0,
            label=rate_spec.label,
        )

    axes.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.65)
    axes.set_title(title)
    axes.set_xlabel("Cycle")
    axes.set_ylabel("Sequential conversion rate")
    axes.set_xticks(rate_table["cycle_index"].tolist())
    axes.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    axes.legend(loc="best")
    figure.tight_layout()
    return figure, axes


def save_conversion_rate_plot(
    figure: plt.Figure,
    run_dir: str | Path,
    *,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    save_formats: list[str] | tuple[str, ...] = ("png", "pdf"),
) -> tuple[Path, ...]:
    """Save a conversion-rate figure in a FunSearch run directory.

    Input:
        figure: Matplotlib figure returned by `plot_conversion_rates(...)`.
        run_dir: Directory where image files should be written.
        output_stem: Filename stem without extension.
        save_formats: One or more of `png` and `pdf`.

    Output:
        Tuple of written image paths.
    """

    normalized_run_dir = _normalize_run_dir(run_dir)
    normalized_formats = [save_format.lower().lstrip(".") for save_format in save_formats]
    unsupported_formats = [
        save_format for save_format in normalized_formats if save_format not in {"png", "pdf"}
    ]
    if unsupported_formats:
        raise ValueError(f"Unsupported save formats: {unsupported_formats}. Use png and/or pdf.")

    output_paths: list[Path] = []
    for save_format in normalized_formats:
        output_path = normalized_run_dir / f"{output_stem}.{save_format}"
        figure.savefig(output_path, dpi=200 if save_format == "png" else None)
        output_paths.append(output_path)
    return tuple(output_paths)


def plot_funsearch_funnel_conversion_rates(
    run_dir: str | Path,
    *,
    rates: list[str] | tuple[str, ...] = ("all",),
    save_formats: list[str] | tuple[str, ...] = ("png", "pdf"),
    output_stem: str = DEFAULT_OUTPUT_STEM,
    save: bool = True,
    title: str = "FunSearch sampler funnel conversion rates",
) -> FunnelPlotOutputs:
    """Load count pickles, compute rates, plot selected rates, and optionally save.

    Input:
        run_dir: FunSearch run directory containing post-processing pickles.
        rates: Rate names to plot. Use any subset of `completed`, `validated`,
            `evaluated`, and `improved`, or use `all`.
        save_formats: Image formats to write when `save=True`. Supported values
            are `png` and `pdf`.
        output_stem: Filename stem for saved plots inside `run_dir`.
        save: Whether to write plot images into `run_dir`.
        title: Plot title.

    Output:
        `FunnelPlotOutputs` with written image paths and the rate table used for
        plotting.
    """

    normalized_run_dir = _normalize_run_dir(run_dir)
    rate_table = compute_conversion_rates(build_cycle_funnel_counts(normalized_run_dir))
    figure, _ = plot_conversion_rates(rate_table, rates=rates, title=title)
    figure_paths: tuple[Path, ...] = ()
    try:
        if save:
            figure_paths = save_conversion_rate_plot(
                figure,
                normalized_run_dir,
                output_stem=output_stem,
                save_formats=save_formats,
            )
    finally:
        plt.close(figure)
    return FunnelPlotOutputs(figure_paths=figure_paths, rate_table=rate_table)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for plotting FunSearch funnel conversion rates."""

    cli_args = list(sys.argv[1:] if argv is None else argv)
    if cli_args and not cli_args[0].startswith("-"):
        reordered_args = cli_args
    elif cli_args and not cli_args[-1].startswith("-"):
        reordered_args = [cli_args[-1], *cli_args[:-1]]
    else:
        reordered_args = cli_args

    parser = argparse.ArgumentParser(
        description="Plot selected sequential FunSearch sampler funnel conversion rates.",
        epilog=(
            "Rate choices: completed=completed/configured, "
            "validated=validated/completed, evaluated=evaluation_completed/validated, "
            "improved=island_best_improvements/evaluation_completed. "
            "Examples: --rates all; --rates completed validated; "
            "--save-formats png pdf; --no-save."
        ),
    )
    parser.add_argument(
        "run_dir",
        help="FunSearch run directory containing the post-processed sampler count pickles.",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        default=["all"],
        choices=["all", *RATE_SPECS.keys()],
        help="One or more rates to plot. Use all for every adjacent funnel rate.",
    )
    parser.add_argument(
        "--save-formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf"],
        help="Image formats to save in the run directory.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem written inside the run directory.",
    )
    parser.add_argument(
        "--title",
        default="FunSearch sampler funnel conversion rates",
        help="Plot title.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write PNG/PDF files; only print the rate table.",
    )
    args = parser.parse_args(reordered_args)

    outputs = plot_funsearch_funnel_conversion_rates(
        args.run_dir,
        rates=args.rates,
        save_formats=args.save_formats,
        output_stem=args.output_stem,
        save=not args.no_save,
        title=args.title,
    )
    print(outputs.rate_table.to_string(index=False))
    for figure_path in outputs.figure_paths:
        print(f"wrote={figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())