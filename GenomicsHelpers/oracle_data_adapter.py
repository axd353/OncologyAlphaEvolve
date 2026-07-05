"""Dataset adapter for the oracle pipeline.

Keep the function signatures in this file stable. When a new dataset format
arrives, update only the bodies here so the rest of the pipeline can stay
unchanged.

Current default assumptions:
- training data is loaded from a pickle file or passed in memory
- the MEC ``train_AA*.pkl`` files are pandas DataFrames with these defaults:
    - label column: ``phenotype``
    - ancestry coordinates: ``PC1`` through ``PC16``
    - genotype dosage columns: names beginning with ``dosage__``
        - optional non-genetic covariates: the explicit fields listed in
            ``DEFAULT_COVARIATE_FIELDS``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_LABEL_FIELD = "phenotype"
DEFAULT_ANCESTRY_FIELDS = tuple(f"PC{index}" for index in range(1, 17))
DEFAULT_COVARIATE_FIELDS = (
    "bmi_cat",
    "current_smoking",
    "ever_smoking",
    "alcohol_intake_cat",
    "packyears_smoking_cat",
    "physical_activity_cat",
    "lycopene_density_cat",
    "percent_fat_cat",
    "calcium_density_cat",
)
DOSAGE_COLUMN_PREFIX = "dosage__"


def load_training_data(data_source: str | Path) -> Any:
    """Load a training dataset.

    Input:
        data_source: Path to a pickle file such as ``train_AA.pkl``.

    Output:
        The deserialized in-memory dataset object.
    """

    data_path = Path(data_source)
    with data_path.open("rb") as handle:
        return pickle.load(handle)


def iter_training_records(training_data: Any) -> Iterable[Any]:
    """Return an iterable of per-subject records.

    Input:
        training_data: Loaded dataset container.

    Output:
        An iterable whose elements each represent one subject record.
    """

    if hasattr(training_data, "itertuples") and hasattr(training_data, "columns"):
        return training_data.itertuples(index=False, name="TrainingRecord")
    if isinstance(training_data, Mapping) and "records" in training_data:
        return training_data["records"]
    if isinstance(training_data, Sequence) and not isinstance(
        training_data,
        (str, bytes, bytearray),
    ):
        return training_data
    raise TypeError("Unsupported training data container. Update iter_training_records().")


def read_record_field(record: Any, field_name: str) -> Any:
    """Read one field from a subject record.

    Input:
        record: One subject record.
        field_name: Field name to extract.

    Output:
        The raw field value.
    """

    if isinstance(record, Mapping):
        return record[field_name]
    return getattr(record, field_name)


def list_record_field_names(record: Any) -> list[str]:
    """Return the available field names for one subject record.

    Input:
        record: One subject record.

    Output:
        A list of field names that can be read from the record.
    """

    if isinstance(record, Mapping):
        return list(record.keys())
    if hasattr(record, "_fields"):
        return list(record._fields)
    if hasattr(record, "__dict__"):
        return list(vars(record).keys())
    return []


def read_label(record: Any) -> float:
    """Return the binary disease label for one subject record.

    Input:
        record: One subject record.

    Output:
        The label as a float.
    """

    return float(read_record_field(record, DEFAULT_LABEL_FIELD))


def read_ancestry_coordinate(record: Any) -> np.ndarray:
    """Return the ancestry coordinate vector for one subject record.

    Input:
        record: One subject record.

    Output:
        A one-dimensional float numpy array.
    """

    return np.asarray(
        [read_record_field(record, field_name) for field_name in DEFAULT_ANCESTRY_FIELDS],
        dtype=float,
    ).reshape(-1)


def read_variant_dosage(record: Any, target_variant: Any) -> float:
    """Return the dosage for one target variant in one subject record.

    Input:
        record: One subject record.
        target_variant: Variant index or key.

    Output:
        The target dosage as a float.
    """

    available_field_names = set(list_record_field_names(record))
    for field_name in candidate_variant_field_names(target_variant):
        if field_name in available_field_names:
            return float(read_record_field(record, field_name))

    raise KeyError(f"Could not find dosage column for variant {target_variant!r}.")


def read_optional_covariates(record: Any) -> np.ndarray | None:
    """Return optional non-genetic covariates for one subject record.

    Input:
        record: One subject record.

    Output:
        A one-dimensional float numpy array, or ``None`` if covariates are absent.
    """

    available_field_names = set(list_record_field_names(record))
    present_covariate_fields = [
        field_name for field_name in DEFAULT_COVARIATE_FIELDS if field_name in available_field_names
    ]
    if not present_covariate_fields:
        return None

    if len(present_covariate_fields) != len(DEFAULT_COVARIATE_FIELDS):
        missing_fields = [
            field_name for field_name in DEFAULT_COVARIATE_FIELDS if field_name not in available_field_names
        ]
        raise ValueError(
            "Record has only a partial covariate layout. Missing fields: "
            f"{missing_fields}."
        )

    return np.asarray(
        [read_record_field(record, field_name) for field_name in DEFAULT_COVARIATE_FIELDS],
        dtype=float,
    ).reshape(-1)


def candidate_variant_field_names(target_variant: Any) -> list[Any]:
    """Return the field names that may store one target variant dosage.

    Input:
        target_variant: Variant key supplied by the caller.

    Output:
        Candidate field names to try in order.
    """

    if not isinstance(target_variant, str):
        return [target_variant]

    candidates = [target_variant]
    if not target_variant.startswith(DOSAGE_COLUMN_PREFIX):
        candidates.append(f"{DOSAGE_COLUMN_PREFIX}{target_variant}")
    return candidates


