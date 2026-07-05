from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PriorityTrainingRecord:
	"""One normalized Oracle-Train sample exposed to a priority function."""

	label: float
	ancestry_coordinate: tuple[float, ...]
	variant_dosages: Mapping[str, float]
	covariates: Mapping[str, float] | None


@dataclass(frozen=True)
class PriorityTrainingData:
	"""Strict Oracle-Train payload passed into a priority function."""

	records: tuple[PriorityTrainingRecord, ...]
	variant_names: tuple[str, ...]
	variant_dosage_fields: tuple[str, ...]
	covariate_names: tuple[str, ...]
	sample_count: int
	ancestry_dimension: int
	has_additional_covariates: bool


@dataclass(frozen=True)
class PriorityAncestryCoordinate:
	"""Normalized target-subject ancestry location passed to a priority function."""

	values: tuple[float, ...]
	dimension: int


@dataclass(frozen=True)
class PriorityTargetVariant:
	"""Normalized target-variant payload passed to a priority function."""

	name: str
	dosage_field: str
	column_index: int


AncestryCoordinate = PriorityAncestryCoordinate
TargetVariant = PriorityTargetVariant
TrainingData = PriorityTrainingData
PriorityScore = float


__all__ = [
	"AncestryCoordinate",
	"PriorityAncestryCoordinate",
	"PriorityScore",
	"PriorityTargetVariant",
	"PriorityTrainingData",
	"PriorityTrainingRecord",
	"TargetVariant",
	"TrainingData",
]
