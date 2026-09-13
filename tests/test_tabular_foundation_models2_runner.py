from __future__ import annotations

import numpy as np
import pandas as pd

from ExperimentsWithTabularFoundationModels2 import runner


class _FakeRowView:
    def __init__(self, distances: list[float]) -> None:
        self.distances = np.asarray(distances, dtype=float)


class _FakeDistanceCache:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def row_view(self, target_index: int) -> _FakeRowView:
        return _FakeRowView(self._rows[target_index])


def test_select_priority_context_indices_for_variant_orders_by_distance_and_truncates() -> None:
    reference_tracking_rows = pd.DataFrame({"dataset_row_index": [40, 10, 20, 30]})
    opened_distance_cache = _FakeDistanceCache([[0.40, 0.20, 0.10, 0.30]])
    radius_matrix = np.asarray([[0.31]], dtype=float)

    selected_indices, distances, radius, fallback_used = runner._select_priority_context_indices_for_variant(
        target_index=0,
        variant_index=0,
        radius_matrix=radius_matrix,
        opened_distance_cache=opened_distance_cache,
        reference_tracking_rows=reference_tracking_rows,
        context_cap=2,
    )

    assert selected_indices.tolist() == [2, 1]
    assert np.allclose(distances, [0.40, 0.20, 0.10, 0.30])
    assert radius == 0.31
    assert fallback_used is False


def test_select_priority_context_indices_for_variant_falls_back_to_nearest_neighbor() -> None:
    reference_tracking_rows = pd.DataFrame({"dataset_row_index": [40, 10, 20, 30]})
    opened_distance_cache = _FakeDistanceCache([[0.40, 0.20, 0.10, 0.30]])
    radius_matrix = np.asarray([[0.05]], dtype=float)

    selected_indices, _, radius, fallback_used = runner._select_priority_context_indices_for_variant(
        target_index=0,
        variant_index=0,
        radius_matrix=radius_matrix,
        opened_distance_cache=opened_distance_cache,
        reference_tracking_rows=reference_tracking_rows,
        context_cap=5,
    )

    assert selected_indices.tolist() == [2]
    assert radius == 0.05
    assert fallback_used is True


def test_build_variant_auc_triplet_frame_pivots_scheme_values_wide() -> None:
    metric_frame = pd.DataFrame(
        [
            {
                "dataset_pair_name": "mec",
                "model_name": "tabpfn",
                "scheme_name": "mixture_learning",
                "variant_index": 0,
                "variant_name": "rs1",
                "variant_dosage_field": "dosage__rs1",
                "auc_roc": 0.60,
            },
            {
                "dataset_pair_name": "mec",
                "model_name": "tabpfn",
                "scheme_name": "independent_learning_scheme",
                "variant_index": 0,
                "variant_name": "rs1",
                "variant_dosage_field": "dosage__rs1",
                "auc_roc": 0.65,
            },
            {
                "dataset_pair_name": "mec",
                "model_name": "tabpfn",
                "scheme_name": "priority_function_curated_context",
                "variant_index": 0,
                "variant_name": "rs1",
                "variant_dosage_field": "dosage__rs1",
                "auc_roc": 0.72,
            },
        ]
    )

    triplet_frame = runner._build_variant_auc_triplet_frame(metric_frame)

    assert triplet_frame.shape == (1, 8)
    assert triplet_frame.loc[0, "mixture_learning_auc_roc"] == 0.60
    assert triplet_frame.loc[0, "independent_learning_scheme_auc_roc"] == 0.65
    assert triplet_frame.loc[0, "priority_function_curated_context_auc_roc"] == 0.72