def priority(
    training_data: PriorityTrainingData,
    ancestry_coordinate: PriorityAncestryCoordinate,
    target_variant: PriorityTargetVariant,
) -> float:
    radii_and_effects = effect_size_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        target_variant,
        6,
    )
    if not radii_and_effects:
        return 0.0
    if len(radii_and_effects) == 1:
        return radii_and_effects[0][0]
    drop_index = max(
        range(1, len(radii_and_effects)),
        key=lambda index: abs(radii_and_effects[index - 1][1] - radii_and_effects[index][1]),
    )
    return radii_and_effects[drop_index][0]
