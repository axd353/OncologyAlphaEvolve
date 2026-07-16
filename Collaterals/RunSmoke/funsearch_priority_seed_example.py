def priority(
    training_data,
    ancestry_coordinate,
    target_variant,
) -> float:
    densities = equal_count_interval_densities(training_data, ancestry_coordinate, 6)
    intervals = equal_count_intervals(training_data, ancestry_coordinate, 6)
    if len(densities) < 2:
        return intervals[0][0]
    drop_index = max(
        range(1, len(densities)),
        key=lambda index: densities[index - 1] - densities[index],
    )
    return intervals[drop_index][0]
