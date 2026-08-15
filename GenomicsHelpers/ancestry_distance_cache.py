from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd

from GenomicsHelpers.oracle_data_adapter import DEFAULT_ANCESTRY_FIELDS


_CACHE_SCHEMA_VERSION = 1
_DEFAULT_NOVELTY_PERCENTAGE = 10.0
_DEFAULT_NOVELTY_BASELINE_SAMPLE_SIZE = 100
_NOVELTY_RANDOM_SEED = 0
_ACTIVE_DISTANCE_CONTEXT: ContextVar[ActiveDistanceContext | None] = ContextVar(
    "active_ancestry_distance_context",
    default=None,
)


@dataclass(frozen=True)
class DistanceCacheArtifacts:
    manifest_path: str
    distances_path: str
    sorted_indices_path: str
    cache_name: str
    reference_hash: str
    target_hash: str
    reference_row_count: int
    target_row_count: int
    ancestry_dimension: int
    novelty_baseline_median: float


@dataclass(frozen=True)
class TargetDistanceView:
    target_index: int
    distances: np.ndarray
    sorted_reference_indices: np.ndarray

    def sorted_distances(self) -> np.ndarray:
        return np.asarray(self.distances[self.sorted_reference_indices], dtype=float)


@dataclass(frozen=True)
class ActiveDistanceContext:
    raw_training_data_id: int
    priority_training_data_id: int | None
    target_ancestry_values: tuple[float, ...]
    target_distance_view: TargetDistanceView
    novelty_baseline_median: float | None


@dataclass
class OpenedDistanceCache:
    artifacts: DistanceCacheArtifacts
    distances: np.ndarray
    sorted_indices: np.ndarray

    def row_view(self, target_index: int) -> TargetDistanceView:
        if target_index < 0 or target_index >= self.artifacts.target_row_count:
            raise IndexError(
                f"Target row index {target_index} is out of range for cached targets "
                f"[0, {self.artifacts.target_row_count})."
            )
        return TargetDistanceView(
            target_index=target_index,
            distances=self.distances[target_index],
            sorted_reference_indices=self.sorted_indices[target_index],
        )


def ensure_distance_cache(
    *,
    reference_data: Any,
    target_data: Any,
    cache_root: Path,
    cache_name: str,
    reference_source_paths: Sequence[str] = (),
    target_source_paths: Sequence[str] = (),
) -> DistanceCacheArtifacts | None:
    if not isinstance(reference_data, pd.DataFrame) or not isinstance(target_data, pd.DataFrame):
        return None

    reference_matrix = _extract_ancestry_matrix(reference_data)
    target_matrix = _extract_ancestry_matrix(target_data)
    reference_hash = _hash_ancestry_matrix(reference_matrix)
    target_hash = _hash_ancestry_matrix(target_matrix)
    cache_key = _cache_key(
        cache_name=cache_name,
        reference_hash=reference_hash,
        target_hash=target_hash,
        reference_row_count=reference_matrix.shape[0],
        target_row_count=target_matrix.shape[0],
        ancestry_dimension=reference_matrix.shape[1],
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / f"{cache_key}.manifest.json"
    distances_path = cache_root / f"{cache_key}.distances.npy"
    sorted_indices_path = cache_root / f"{cache_key}.sorted_indices.npy"

    if manifest_path.exists() and distances_path.exists() and sorted_indices_path.exists():
        artifacts = _load_artifacts_from_manifest(manifest_path)
        if artifacts is not None:
            return artifacts

    distances = _compute_distance_matrix(target_matrix, reference_matrix)
    sorted_indices = np.argsort(distances, axis=1, kind="mergesort").astype(np.int32, copy=False)
    novelty_baseline_median = _compute_novelty_baseline_median(reference_matrix)

    np.save(distances_path, distances)
    np.save(sorted_indices_path, sorted_indices)
    manifest_payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "cache_name": cache_name,
        "distances_path": str(distances_path),
        "sorted_indices_path": str(sorted_indices_path),
        "reference_hash": reference_hash,
        "target_hash": target_hash,
        "reference_row_count": int(reference_matrix.shape[0]),
        "target_row_count": int(target_matrix.shape[0]),
        "ancestry_dimension": int(reference_matrix.shape[1]),
        "novelty_baseline_median": float(novelty_baseline_median),
        "reference_source_paths": list(reference_source_paths),
        "target_source_paths": list(target_source_paths),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    return DistanceCacheArtifacts(
        manifest_path=str(manifest_path),
        distances_path=str(distances_path),
        sorted_indices_path=str(sorted_indices_path),
        cache_name=cache_name,
        reference_hash=reference_hash,
        target_hash=target_hash,
        reference_row_count=int(reference_matrix.shape[0]),
        target_row_count=int(target_matrix.shape[0]),
        ancestry_dimension=int(reference_matrix.shape[1]),
        novelty_baseline_median=float(novelty_baseline_median),
    )


def load_distance_cache(manifest_path: str | Path | None) -> OpenedDistanceCache | None:
    if manifest_path is None:
        return None
    artifacts = _load_artifacts_from_manifest(Path(manifest_path))
    if artifacts is None:
        return None
    distances = np.load(artifacts.distances_path, mmap_mode="r")
    sorted_indices = np.load(artifacts.sorted_indices_path, mmap_mode="r")
    if distances.shape != (artifacts.target_row_count, artifacts.reference_row_count):
        raise ValueError(
            "Distance cache shape does not match its manifest: "
            f"distances.shape={distances.shape} manifest="
            f"({artifacts.target_row_count}, {artifacts.reference_row_count})."
        )
    if sorted_indices.shape != distances.shape:
        raise ValueError(
            "Sorted-index cache shape does not match distance cache shape: "
            f"sorted_indices.shape={sorted_indices.shape} distances.shape={distances.shape}."
        )
    return OpenedDistanceCache(
        artifacts=artifacts,
        distances=distances,
        sorted_indices=sorted_indices,
    )


@contextmanager
def activate_distance_context(
    *,
    raw_training_data: Any,
    priority_training_data: Any | None,
    target_ancestry_values: Sequence[float],
    target_distance_view: TargetDistanceView,
    novelty_baseline_median: float | None,
) -> Iterator[None]:
    token = _ACTIVE_DISTANCE_CONTEXT.set(
        ActiveDistanceContext(
            raw_training_data_id=id(raw_training_data),
            priority_training_data_id=(id(priority_training_data) if priority_training_data is not None else None),
            target_ancestry_values=_normalize_ancestry_values(target_ancestry_values),
            target_distance_view=target_distance_view,
            novelty_baseline_median=novelty_baseline_median,
        )
    )
    try:
        yield
    finally:
        _ACTIVE_DISTANCE_CONTEXT.reset(token)


def get_active_priority_distance_context(
    training_data: Any,
    ancestry_coordinate: Any,
) -> ActiveDistanceContext | None:
    context = _ACTIVE_DISTANCE_CONTEXT.get()
    if context is None:
        return None
    if context.priority_training_data_id != id(training_data):
        return None
    if context.target_ancestry_values != _normalize_ancestry_values(_ancestry_values_from_object(ancestry_coordinate)):
        return None
    return context


def get_active_raw_distance_context(
    training_data: Any,
    ancestry_coordinate: Any,
) -> ActiveDistanceContext | None:
    context = _ACTIVE_DISTANCE_CONTEXT.get()
    if context is None:
        return None
    if context.raw_training_data_id != id(training_data):
        return None
    if context.target_ancestry_values != _normalize_ancestry_values(ancestry_coordinate):
        return None
    return context


def _load_artifacts_from_manifest(manifest_path: Path) -> DistanceCacheArtifacts | None:
    try:
        payload = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if int(payload.get("schema_version", 0)) != _CACHE_SCHEMA_VERSION:
        return None
    distances_path = Path(str(payload["distances_path"]))
    sorted_indices_path = Path(str(payload["sorted_indices_path"]))
    if not distances_path.exists() or not sorted_indices_path.exists():
        return None
    return DistanceCacheArtifacts(
        manifest_path=str(manifest_path),
        distances_path=str(distances_path),
        sorted_indices_path=str(sorted_indices_path),
        cache_name=str(payload["cache_name"]),
        reference_hash=str(payload["reference_hash"]),
        target_hash=str(payload["target_hash"]),
        reference_row_count=int(payload["reference_row_count"]),
        target_row_count=int(payload["target_row_count"]),
        ancestry_dimension=int(payload["ancestry_dimension"]),
        novelty_baseline_median=float(payload["novelty_baseline_median"]),
    )


def _extract_ancestry_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing_columns = [column for column in DEFAULT_ANCESTRY_FIELDS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing expected ancestry columns: {missing_columns}.")
    matrix = frame.loc[:, DEFAULT_ANCESTRY_FIELDS].to_numpy(dtype=float, copy=True)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional ancestry matrix, got shape {matrix.shape}.")
    return np.ascontiguousarray(matrix, dtype=float)


def _hash_ancestry_matrix(matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode("utf-8"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _cache_key(
    *,
    cache_name: str,
    reference_hash: str,
    target_hash: str,
    reference_row_count: int,
    target_row_count: int,
    ancestry_dimension: int,
) -> str:
    normalized_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in cache_name
    )
    return (
        f"{normalized_name}.ref_{reference_hash[:16]}.tgt_{target_hash[:16]}."
        f"rows_{reference_row_count}x{target_row_count}.dim_{ancestry_dimension}"
    )


def _compute_distance_matrix(
    target_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    target_count = target_matrix.shape[0]
    reference_squared = np.sum(reference_matrix * reference_matrix, axis=1, dtype=float)
    distances = np.empty((target_count, reference_matrix.shape[0]), dtype=float)
    for start_index in range(0, target_count, chunk_size):
        end_index = min(target_count, start_index + chunk_size)
        target_chunk = target_matrix[start_index:end_index]
        squared_distances = (
            np.sum(target_chunk * target_chunk, axis=1, dtype=float)[:, None]
            + reference_squared[None, :]
            - 2.0 * (target_chunk @ reference_matrix.T)
        )
        np.maximum(squared_distances, 0.0, out=squared_distances)
        distances[start_index:end_index] = np.sqrt(squared_distances)
    return distances


def _compute_novelty_baseline_median(reference_matrix: np.ndarray) -> float:
    sample_count = int(reference_matrix.shape[0])
    if sample_count <= 1:
        return 0.0

    target_count = max(1, math.floor(sample_count * _DEFAULT_NOVELTY_PERCENTAGE / 100.0))
    sampled_indices = list(range(sample_count))
    if sample_count > _DEFAULT_NOVELTY_BASELINE_SAMPLE_SIZE:
        sampled_indices = random.Random(_NOVELTY_RANDOM_SEED).sample(
            sampled_indices,
            k=_DEFAULT_NOVELTY_BASELINE_SAMPLE_SIZE,
        )

    baseline_radii: list[float] = []
    for record_index in sampled_indices:
        target_values = reference_matrix[record_index : record_index + 1]
        row_distances = _compute_distance_matrix(target_values, reference_matrix)[0]
        if row_distances.size <= 1:
            baseline_radii.append(0.0)
            continue
        without_self = np.delete(row_distances, record_index)
        without_self.sort()
        prefix_count = min(target_count, without_self.size)
        baseline_radii.append(_radius_for_prefix_count(without_self, prefix_count))

    if not baseline_radii:
        return 0.0
    return float(np.median(np.asarray(baseline_radii, dtype=float)))


def _radius_for_prefix_count(sorted_distances: np.ndarray, prefix_count: int) -> float:
    if prefix_count <= 0 or sorted_distances.size == 0:
        return 0.0
    lower_distance = float(sorted_distances[prefix_count - 1])
    return math.nextafter(lower_distance, math.inf)


def _ancestry_values_from_object(value: Any) -> Sequence[float]:
    if hasattr(value, "values"):
        return value.values
    return value


def _normalize_ancestry_values(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(component) for component in values)