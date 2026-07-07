def priority(
    training_data,
    ancestry_coordinate,
    target_variant,
) -> float:
    """Simple seed priority function used to bootstrap the program database."""

    ancestry_dimensionality = float(ancestry_coordinate.dimension)
    variant_name_length = float(len(target_variant.name))
    sample_count = float(training_data.sample_count)
    return ancestry_dimensionality * 0.01 + variant_name_length * 0.001 + sample_count * 0.0
