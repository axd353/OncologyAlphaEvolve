# label=cycle_0001
# source_kind=cycle_best
# cycle_index=1
# original_prio_function_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_052346/cycle_0001/best_prio.py
# discovery_snapshot_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_052346/cycle_0001/program_db_end.pkl
# discovery_source_island_id=3
# discovery_reduced_score=0.5738927824708068
# discovery_scores_per_test={"combined": 0.5738927824708068, "mean": 0.5738927824708068, "no_covariates_fold_1": 0.5718163485047247, "no_covariates_fold_2": 0.5759692164368888, "simplicity": -1228.0, "simplicity_bonus": 0.0}
# priority_source_hash=a9e8d2d77f9aed5d29f3ea025a433a9a431d12c8d776f536d3ad3ab0ed4df673
# reused_from_label=None
# selected_calibration_penalty=0.1
# heldout_subject_count=400
# heldout_auc_roc=0.547275

def priority(training_data, ancestry_coordinate, target_variant) -> float:
  import math

  def clean_radius(value, default=0.0):
    try:
      radius = float(value)
    except Exception:
      try:
        fallback = float(default)
      except Exception:
        return 0.0
      return fallback if math.isfinite(fallback) and fallback >= 0.0 else 0.0
    if math.isfinite(radius) and radius >= 0.0:
      return radius
    try:
      fallback = float(default)
    except Exception:
      return 0.0
    return fallback if math.isfinite(fallback) and fallback >= 0.0 else 0.0

  def radius_for_usable_percentage(percentage):
    try:
      pct = max(0.0, min(100.0, float(percentage)))
    except Exception:
      pct = 0.0
    try:
      radius, _ = minimum_radius_for_training_percentage(
        training_data,
        ancestry_coordinate,
        target_variant,
        pct,
      )
      return clean_radius(radius)
    except Exception:
      pass
    try:
      return clean_radius(radius_for_percentage(training_data, ancestry_coordinate, pct))
    except Exception:
      return 0.0

  def dosage_entropy(values):
    counts = [0, 0, 0]
    total = 0
    for value in values:
      if value < 0.5:
        counts[0] += 1
      elif value < 1.5:
        counts[1] += 1
      else:
        counts[2] += 1
      total += 1
    if total <= 0:
      return 0.0
    entropy = 0.0
    for count in counts:
      if count:
        p = count / total
        entropy -= p * math.log(p, 2.0)
    return entropy

  records = getattr(training_data, "records", ())
  try:
    sample_count = int(getattr(training_data, "sample_count", len(records)))
  except Exception:
    sample_count = len(records)

  if sample_count <= 0:
    return 0.0

  dosage_field = getattr(target_variant, "dosage_field", None)
  variant_name = getattr(target_variant, "name", None)
  dosages = []

  for record in records:
    variant_dosages = getattr(record, "variant_dosages", {}) or {}
    value = None
    if dosage_field in variant_dosages:
      value = variant_dosages.get(dosage_field)
    elif variant_name in variant_dosages:
      value = variant_dosages.get(variant_name)
    try:
      value = float(value)
    except Exception:
      continue
    if math.isfinite(value):
      dosages.append(value)

  usable_count = len(dosages)
  if usable_count < 2:
    return 0.0

  mean_dosage = sum(dosages) / usable_count
  dosage_variance = sum((value - mean_dosage) ** 2 for value in dosages) / usable_count
  if not math.isfinite(dosage_variance) or dosage_variance <= 1e-12:
    return radius_for_usable_percentage(100.0)

  entropy = dosage_entropy(dosages)
  basis_count = max(1, min(sample_count, usable_count))

  if basis_count < 40:
    min_percentage = 100.0
    base_percentage = 100.0
  elif basis_count < 100:
    min_percentage = 60.0
    base_percentage = 80.0
  elif basis_count < 250:
    min_percentage = 40.0
    base_percentage = 60.0
  else:
    min_percentage = 20.0
    base_percentage = 50.0

  if dosage_variance < 0.02 or entropy < 0.15:
    min_percentage = max(min_percentage, 50.0)
    base_percentage = max(base_percentage, 80.0)
  elif dosage_variance < 0.05 or entropy < 0.35:
    min_percentage = max(min_percentage, 35.0)
    base_percentage = max(base_percentage, 65.0)

  try:
    novelty = float(ancestry_novelty_score(training_data, ancestry_coordinate))
  except Exception:
    novelty = 1.0

  if math.isfinite(novelty):
    if novelty > 2.0:
      base_percentage = max(base_percentage, 85.0)
    elif novelty > 1.35:
      base_percentage = max(base_percentage, 65.0)
    elif novelty < 0.60 and entropy >= 0.25 and dosage_variance >= 0.03:
      base_percentage = min(base_percentage, 35.0)

  min_percentage = max(0.0, min(100.0, min_percentage))
  base_percentage = max(min_percentage, min(100.0, base_percentage))
  min_radius = radius_for_usable_percentage(min_percentage)

  if basis_count >= 240:
    num_intervals = 8
  elif basis_count >= 120:
    num_intervals = 6
  else:
    num_intervals = 4
  num_intervals = max(1, min(30, num_intervals, sample_count))

  min_samples = max(5, min(25, basis_count // max(1, 2 * num_intervals)))
  chosen_radius = None

  try:
    changes = standardized_effect_change_by_interval(
      training_data,
      ancestry_coordinate,
      target_variant,
      num_intervals,
      min_samples=min_samples,
    )
    finite_changes = []
    for index, change in enumerate(changes):
      try:
        change = float(change)
      except Exception:
        continue
      if math.isfinite(change):
        finite_changes.append((index, change))
  except Exception:
    finite_changes = []

  if finite_changes:
    max_index, max_change = max(finite_changes, key=lambda item: item[1])
    if max_change >= 2.5:
      try:
        intervals = equal_count_intervals(training_data, ancestry_coordinate, num_intervals)
        chosen_radius = max(clean_radius(intervals[max_index][1]), min_radius)
      except Exception:
        chosen_radius = min_radius
    elif max_change <= 0.75 and basis_count >= 250:
      base_percentage = max(base_percentage, 70.0)
    elif max_change >= 1.5:
      base_percentage = max(min_percentage, min(base_percentage, 50.0))

  if chosen_radius is None:
    chosen_radius = max(radius_for_usable_percentage(base_percentage), min_radius)

  try:
    cumulative_entropy = dosage_entropy_by_cumulative_radius(
      training_data,
      ancestry_coordinate,
      target_variant,
      num_intervals,
    )
    entropy_pairs = []
    for radius, local_entropy in cumulative_entropy:
      try:
        local_entropy = float(local_entropy)
      except Exception:
        continue
      if math.isfinite(local_entropy):
        entropy_pairs.append((clean_radius(radius), local_entropy))

    if entropy_pairs:
      global_entropy = max(0.0, entropy_pairs[-1][1])
      entropy_threshold = max(1e-6, min(0.15, 0.30 * global_entropy))
      if not any(
        radius <= chosen_radius and local_entropy >= entropy_threshold
        for radius, local_entropy in entropy_pairs
      ):
        for radius, local_entropy in entropy_pairs:
          if radius >= chosen_radius and local_entropy >= entropy_threshold:
            chosen_radius = radius
            break
  except Exception:
    pass

  return clean_radius(chosen_radius)

