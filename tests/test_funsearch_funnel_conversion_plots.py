from __future__ import annotations

from pathlib import Path

import pandas as pd

from PostProcesingData.funsearch_funnel_conversion_plots import compute_conversion_rates
from PostProcesingData.funsearch_funnel_conversion_plots import normalize_rate_names
from PostProcesingData.funsearch_funnel_conversion_plots import plot_funsearch_funnel_conversion_rates
from PostProcesingData.funsearch_run_postprocess import COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE
from PostProcesingData.funsearch_run_postprocess import EVALUATION_COMPLETED_COUNTS_PICKLE
from PostProcesingData.funsearch_run_postprocess import ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE
from PostProcesingData.funsearch_run_postprocess import VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE


def _write_pickle(path: Path, data_frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_pickle(path)


def _write_plotter_input_pickles(run_dir: Path) -> None:
    _write_pickle(
        run_dir / COMPLETED_PRIORITY_FUNCTION_COUNTS_PICKLE,
        pd.DataFrame(
            [
                {
                    "cycle_index": 1,
                    "island_id": 0,
                    "completed_priority_function_count": 3,
                    "configured_candidates_per_island_per_cycle": 6,
                },
                {
                    "cycle_index": 1,
                    "island_id": 1,
                    "completed_priority_function_count": 5,
                    "configured_candidates_per_island_per_cycle": 6,
                },
                {
                    "cycle_index": 2,
                    "island_id": 0,
                    "completed_priority_function_count": 4,
                    "configured_candidates_per_island_per_cycle": 6,
                },
            ]
        ),
    )
    _write_pickle(
        run_dir / VALIDATED_PRIORITY_FUNCTION_COUNTS_PICKLE,
        pd.DataFrame(
            [
                {"cycle_index": 1, "island_id": 0, "validated_priority_function_count": 2},
                {"cycle_index": 1, "island_id": 1, "validated_priority_function_count": 2},
                {"cycle_index": 2, "island_id": 0, "validated_priority_function_count": 1},
            ]
        ),
    )
    _write_pickle(
        run_dir / EVALUATION_COMPLETED_COUNTS_PICKLE,
        pd.DataFrame(
            [
                {"cycle_index": 1, "island_id": 0, "evaluation_completed_count": 1},
                {"cycle_index": 1, "island_id": 1, "evaluation_completed_count": 1},
                {"cycle_index": 2, "island_id": 0, "evaluation_completed_count": 1},
            ]
        ),
    )
    _write_pickle(
        run_dir / ISLAND_BEST_IMPROVEMENT_COUNTS_PICKLE,
        pd.DataFrame(
            [
                {"cycle_index": 1, "island_best_improvement_count": 1},
                {"cycle_index": 2, "island_best_improvement_count": 0},
            ]
        ),
    )


def test_normalize_rate_names_expands_all_and_dedupes() -> None:
    assert normalize_rate_names(["all"]) == ["completed", "validated", "evaluated", "improved"]
    assert normalize_rate_names(["completed", "completed", "improved"]) == [
        "completed",
        "improved",
    ]


def test_compute_conversion_rates_uses_adjacent_funnel_counts() -> None:
    rate_table = compute_conversion_rates(
        pd.DataFrame(
            [
                {
                    "cycle_index": 1,
                    "configured_candidate_slot_count": 12,
                    "completed_priority_function_count": 8,
                    "validated_priority_function_count": 4,
                    "evaluation_completed_count": 2,
                    "island_best_improvement_count": 1,
                }
            ]
        )
    )

    row = rate_table.iloc[0]
    assert row["completed_over_configured_rate"] == 8 / 12
    assert row["validated_over_completed_rate"] == 4 / 8
    assert row["evaluated_over_validated_rate"] == 2 / 4
    assert row["improved_over_evaluated_rate"] == 1 / 2


def test_plot_funsearch_funnel_conversion_rates_saves_selected_formats(tmp_path: Path) -> None:
    run_dir = tmp_path / "oracle_priority_20260716_050704"
    _write_plotter_input_pickles(run_dir)

    outputs = plot_funsearch_funnel_conversion_rates(
        run_dir,
        rates=["completed", "validated"],
        save_formats=["png", "pdf"],
        output_stem="selected_funnel_rates",
    )

    assert [path.name for path in outputs.figure_paths] == [
        "selected_funnel_rates.png",
        "selected_funnel_rates.pdf",
    ]
    assert all(path.exists() for path in outputs.figure_paths)
    assert list(outputs.rate_table["cycle_index"]) == [1, 2]
    assert outputs.rate_table.loc[0, "configured_candidate_slot_count"] == 12
    assert outputs.rate_table.loc[0, "completed_priority_function_count"] == 8