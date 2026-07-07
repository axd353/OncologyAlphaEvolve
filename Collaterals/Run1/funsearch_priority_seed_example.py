from __future__ import annotations

from funsearch_pipeline.priority_tools.contracts import PriorityAncestryCoordinate
from funsearch_pipeline.priority_tools.contracts import PriorityTargetVariant
from funsearch_pipeline.priority_tools.contracts import PriorityTrainingData


def priority(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
) -> float:
    """Simple seed priority function used to bootstrap the program database."""

    ancestry_dimensionality = float(ancestry_coordinate.dimension)
    variant_name_length = float(len(target_variant.name))
    sample_count = float(training_data.sample_count)
    return ancestry_dimensionality * 0.01 + variant_name_length * 0.001 + sample_count * 0.0
