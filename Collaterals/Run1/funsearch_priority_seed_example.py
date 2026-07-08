from funsearch_pipeline.priority_tools import PriorityAncestryCoordinate
from funsearch_pipeline.priority_tools import PriorityTargetVariant
from funsearch_pipeline.priority_tools import PriorityTrainingData
from funsearch_pipeline.priority_tools import equal_count_interval_densities
from funsearch_pipeline.priority_tools import equal_count_intervals


def priority(
    training_data,
    ancestry_coordinate,
    target_variant,
) -> float:
    """Choose the interval boundary after the sharpest density drop-off."""

    densities = equal_count_interval_densities(training_data, ancestry_coordinate, 6)
    intervals = equal_count_intervals(training_data, ancestry_coordinate, 6)
    if len(densities) < 2:
        return intervals[0][0]
    drop_index = max(
        range(1, len(densities)),
        key=lambda index: densities[index - 1] - densities[index],
    )
    return intervals[drop_index][0]
