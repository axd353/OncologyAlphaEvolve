from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from GenomicsHelpers.oracle_data_adapter import DEFAULT_ANCESTRY_FIELDS
from GenomicsHelpers.oracle_data_adapter import DEFAULT_LABEL_FIELD
from GenomicsHelpers.oracle_data_adapter import DOSAGE_COLUMN_PREFIX
from funsearch_pipeline.evaluation.procedure2 import _safe_roc_auc


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    dosage_columns = [
        str(column)
        for column in frame.columns
        if str(column).startswith(DOSAGE_COLUMN_PREFIX)
    ]
    if not dosage_columns:
        raise ValueError("Mixture Learning baseline requires at least one dosage column.")

    missing_ancestry_columns = [
        column_name for column_name in DEFAULT_ANCESTRY_FIELDS if column_name not in frame.columns
    ]
    if missing_ancestry_columns:
        raise ValueError(
            "Mixture Learning baseline requires ancestry columns "
            f"{missing_ancestry_columns}."
        )
    return dosage_columns + list(DEFAULT_ANCESTRY_FIELDS)


def _extract_features(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, _feature_columns(frame)].to_numpy(dtype=float)


def _extract_labels(frame: pd.DataFrame) -> np.ndarray:
    if DEFAULT_LABEL_FIELD not in frame.columns:
        raise ValueError(
            "Mixture Learning baseline requires the label column "
            f"{DEFAULT_LABEL_FIELD!r}."
        )
    return frame[DEFAULT_LABEL_FIELD].to_numpy(dtype=float)


def evaluate_mixture_learning(
    training_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
    heldout_data: pd.DataFrame,
    options: dict[str, Any],
) -> float:
    combined_training = pd.concat([training_data, calibration_data], ignore_index=True)
    alpha = float(options.get("alpha", 1.0))
    model = Ridge(alpha=alpha)
    model.fit(_extract_features(combined_training), _extract_labels(combined_training))
    heldout_scores = model.predict(_extract_features(heldout_data))
    return _safe_roc_auc(_extract_labels(heldout_data), heldout_scores)