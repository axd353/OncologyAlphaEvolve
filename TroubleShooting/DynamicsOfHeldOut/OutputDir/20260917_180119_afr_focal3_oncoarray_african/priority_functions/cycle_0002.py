# label=cycle_0002
# source_kind=cycle_best
# cycle_index=2
# original_prio_function_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_052346/cycle_0002/best_prio.py
# discovery_snapshot_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_052346/cycle_0002/program_db_end.pkl
# discovery_source_island_id=1
# discovery_reduced_score=0.5924817555394696
# discovery_scores_per_test={"combined": 0.5924817555394696, "mean": 0.6004817555394696, "no_covariates_fold_1": 0.6358733663546481, "no_covariates_fold_2": 0.5650901447242911, "simplicity": -3279.0, "simplicity_bonus": -0.008}
# priority_source_hash=b4236cc49317596b8f4f1e4afaf0aa481f28cb6b4a0789d0697ccdd56f9a0625
# reused_from_label=None
# selected_calibration_penalty=0.1
# heldout_subject_count=400
# heldout_auc_roc=0.581075

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

  def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count <= 0:
      return 0.0
    middle = count // 2
    if count % 2:
      return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])

  records = getattr(training_data, "records", ()) or ()
  try:
    sample_count = int(getattr(training_data, "sample_count", len(records)))
  except Exception:
    sample_count = len(records)

  if sample_count <= 0:
    return 0.0

  dosage_keys = []
  dosage_field = getattr(target_variant, "dosage_field", None)
  variant_name = getattr(target_variant, "name", None)

  def add_key(key):
    if key is not None and key not in dosage_keys:
      dosage_keys.append(key)

  add_key(dosage_field)
  add_key(variant_name)
  try:
    if variant_name is not None:
      text_name = str(variant_name)
      add_key("dosage__" + text_name)
  except Exception:
    pass
  try:
    if dosage_field is not None:
      text_field = str(dosage_field)
      if text_field.startswith("dosage__"):
        add_key(text_field[8:])
  except Exception:
    pass

  try:
    column_index = int(getattr(target_variant, "column_index", -1))
    fields = getattr(training_data, "variant_dosage_fields", ()) or ()
    names = getattr(training_data, "variant_names", ()) or ()
    if 0 <= column_index < len(fields):
      add_key(fields[column_index])
    if 0 <= column_index < len(names):
      add_key(names[column_index])
      try:
        add_key("dosage__" + str(names[column_index]))
      except Exception:
        pass
  except Exception:
    pass

  dosages = []
  labels = []
  label_low = 0
  label_high = 0
  for record in records:
    variant_dosages = getattr(record, "variant_dosages", {}) or {}
    value = None
    for key in dosage_keys:
      try:
        if key in variant_dosages:
          value = variant_dosages.get(key)
          break
      except Exception:
        continue
    try:
      value = float(value)
    except Exception:
      continue
    if math.isfinite(value):
      dosages.append(value)
      try:
        label = float(getattr(record, "label"))
        if math.isfinite(label):
          labels.append(label)
          if label <= 0.5:
            label_low += 1
          else:
            label_high += 1
      except Exception:
        pass

  usable_count = len(dosages)
  if usable_count < 2:
    return 0.0

  mean_dosage = sum(dosages) / usable_count
  dosage_variance = sum((value - mean_dosage) ** 2 for value in dosages) / usable_count
  if not math.isfinite(dosage_variance) or dosage_variance <= 1e-12:
    return clean_radius(radius_for_usable_percentage(100.0))

  entropy = dosage_entropy(dosages)
  basis_count = max(1, min(sample_count, usable_count))
  usable_fraction = usable_count / max(1.0, float(sample_count))

  if basis_count < 50:
    min_percentage = 100.0
    base_percentage = 100.0
  elif basis_count < 120:
    min_percentage = 60.0
    base_percentage = 82.0
  elif basis_count < 300:
    min_percentage = 35.0
    base_percentage = 60.0
  elif basis_count < 800:
    min_percentage = 20.0
    base_percentage = 45.0
  elif basis_count < 1500:
    min_percentage = 15.0
    base_percentage = 40.0
  else:
    min_percentage = 12.0
    base_percentage = 35.0

  clipped_sum = 0.0
  for value in dosages:
    if value <= 0.0:
      clipped = 0.0
    elif value >= 2.0:
      clipped = 2.0
    else:
      clipped = value
    clipped_sum += clipped
  allele_frequency = clipped_sum / max(1.0, 2.0 * usable_count)
  minor_frequency = min(max(allele_frequency, 0.0), max(0.0, 1.0 - allele_frequency))
  minor_copies = 2.0 * usable_count * minor_frequency

  if dosage_variance < 0.02 or entropy < 0.15 or minor_copies < 12.0:
    min_percentage = max(min_percentage, 65.0)
    base_percentage = max(base_percentage, 90.0)
  elif dosage_variance < 0.05 or entropy < 0.35 or minor_copies < 30.0:
    min_percentage = max(min_percentage, 45.0)
    base_percentage = max(base_percentage, 75.0)
  elif minor_copies < 60.0:
    min_percentage = max(min_percentage, 35.0)
    base_percentage = max(base_percentage, 65.0)

  if usable_fraction < 0.45:
    min_percentage = max(min_percentage, 55.0)
    base_percentage = max(base_percentage, 85.0)
  elif usable_fraction < 0.75:
    min_percentage = max(min_percentage, 35.0)
    base_percentage = max(base_percentage, 65.0)

  try:
    novelty = float(ancestry_novelty_score(training_data, ancestry_coordinate))
  except Exception:
    novelty = 1.0

  if math.isfinite(novelty):
    if novelty > 2.0:
      min_percentage = max(min_percentage, 55.0)
      base_percentage = max(base_percentage, 85.0)
    elif novelty > 1.35:
      base_percentage = max(base_percentage, 70.0)
    elif novelty < 0.60 and entropy >= 0.35 and dosage_variance >= 0.04 and minor_copies >= 60.0 and basis_count >= 250:
      base_percentage = min(base_percentage, 35.0)

  label_minor_global = None
  if len(labels) >= 2:
    label_mean = sum(labels) / len(labels)
    label_variance = sum((label - label_mean) ** 2 for label in labels) / len(labels)
    label_minor_global = min(label_low, label_high)
    if math.isfinite(label_variance):
      mean_based_minor = len(labels) * min(max(label_mean, 0.0), max(0.0, 1.0 - label_mean))
      label_minor = min(label_minor_global, mean_based_minor) if label_minor_global is not None else mean_based_minor
      if label_variance < 0.01 or label_minor < 8.0:
        min_percentage = max(min_percentage, 55.0)
        base_percentage = max(base_percentage, 82.0)
      elif label_minor < 20.0:
        min_percentage = max(min_percentage, 40.0)
        base_percentage = max(base_percentage, 68.0)

  min_percentage = max(0.0, min(100.0, min_percentage))
  base_percentage = max(min_percentage, min(100.0, base_percentage))
  min_radius = clean_radius(radius_for_usable_percentage(min_percentage))

  if basis_count >= 1500:
    num_intervals = 16
  elif basis_count >= 800:
    num_intervals = 14
  elif basis_count >= 500:
    num_intervals = 12
  elif basis_count >= 400:
    num_intervals = 10
  elif basis_count >= 240:
    num_intervals = 8
  elif basis_count >= 120:
    num_intervals = 6
  else:
    num_intervals = 4
  num_intervals = max(1, min(30, num_intervals, sample_count))

  min_samples = max(5, min(45, basis_count // max(1, 2 * num_intervals)))

  cap_radius = None
  heterogeneous = False
  stable_effects = False
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
      if math.isfinite(change) and change >= 0.0:
        finite_changes.append((index, change))

    if finite_changes:
      change_values = [change for _, change in finite_changes]
      max_index, max_change = max(finite_changes, key=lambda item: item[1])
      median_change = median(change_values)
      if max_change >= 2.35 or (max_change >= 2.05 and max_change >= median_change + 0.60):
        intervals = equal_count_intervals(training_data, ancestry_coordinate, num_intervals)
        trigger_level = max(1.85, 0.80 * max_change)
        trigger_index = max_index
        for index, change in finite_changes:
          if change >= trigger_level:
            trigger_index = index
            break
        cap_radius = max(clean_radius(intervals[trigger_index][1]), min_radius)
        heterogeneous = True
      elif max_change <= 0.60 and basis_count >= 250:
        base_percentage = max(base_percentage, 82.0)
        stable_effects = True
      elif max_change <= 0.90 and basis_count >= 250:
        base_percentage = max(base_percentage, 72.0)
        stable_effects = True
      elif max_change >= 1.40:
        base_percentage = max(min_percentage, min(base_percentage, 55.0))
  except Exception:
    pass

  if heterogeneous and cap_radius is not None:
    chosen_radius = cap_radius
  else:
    chosen_radius = max(clean_radius(radius_for_usable_percentage(base_percentage)), min_radius)

  se_pairs_raw = []
  try:
    se_pairs_raw = effect_size_standard_error_by_cumulative_radius(
      training_data,
      ancestry_coordinate,
      target_variant,
      num_intervals,
      min_samples=min_samples,
    )
    se_pairs = []
    for radius, se in se_pairs_raw:
      try:
        se = float(se)
      except Exception:
        continue
      radius = clean_radius(radius)
      if math.isfinite(se) and se >= 0.0 and radius >= min_radius:
        se_pairs.append((radius, se))

    if se_pairs:
      pool = se_pairs
      if cap_radius is not None:
        capped = [(radius, se) for radius, se in se_pairs if radius <= cap_radius + 1e-12]
        if capped:
          pool = capped
      best_se = min(se for _, se in pool)
      if minor_copies < 30.0:
        se_threshold = max(best_se * 1.20, best_se + 0.03)
      elif stable_effects:
        se_threshold = max(best_se * 1.45, best_se + 0.06)
      else:
        se_threshold = max(best_se * 1.30, best_se + 0.05)
      first_good = None
      for radius, se in pool:
        if se <= se_threshold:
          first_good = radius
          break
      if first_good is not None:
        if heterogeneous:
          chosen_radius = max(min_radius, first_good)
        elif stable_effects and math.isfinite(novelty) and novelty <= 1.1:
          chosen_radius = max(min_radius, min(chosen_radius, first_good))
        else:
          chosen_radius = max(chosen_radius, first_good)
  except Exception:
    pass

  try:
    if (
      not heterogeneous
      and basis_count >= 250
      and minor_copies >= 50.0
      and entropy >= 0.35
      and math.isfinite(novelty)
      and novelty <= 1.25
      and se_pairs_raw
    ):
      effect_pairs_raw = effect_size_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        target_variant,
        num_intervals,
        min_samples=min_samples,
      )
      effect_se_pairs = []
      limit = min(len(effect_pairs_raw), len(se_pairs_raw))
      for index in range(limit):
        radius, effect = effect_pairs_raw[index]
        _, se = se_pairs_raw[index]
        try:
          radius = clean_radius(radius)
          effect = float(effect)
          se = float(se)
        except Exception:
          continue
        if radius >= min_radius and math.isfinite(effect) and math.isfinite(se) and se >= 0.0:
          effect_se_pairs.append((radius, effect, se))
      if effect_se_pairs:
        reference_radius, reference_effect, reference_se = effect_se_pairs[-1]
        best_se = min(se for _, _, se in effect_se_pairs)
        first_converged = None
        for radius, effect, se in effect_se_pairs:
          combined = math.sqrt(se * se + reference_se * reference_se)
          tolerance = max(0.08, combined)
          if abs(effect - reference_effect) <= tolerance and se <= max(best_se * 1.50, best_se + 0.08):
            first_converged = radius
            break
        if first_converged is not None and first_converged < chosen_radius:
          if stable_effects or novelty < 0.85 or basis_count >= 500:
            chosen_radius = max(min_radius, first_converged)
  except Exception:
    pass

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
        entropy_pairs.append((clean_radius(radius), max(0.0, local_entropy)))

    if entropy_pairs:
      global_entropy = entropy_pairs[-1][1]
      if global_entropy > 0.0:
        if minor_copies < 30.0:
          entropy_fraction = 0.35
        else:
          entropy_fraction = 0.28
        entropy_threshold = max(0.04, min(0.35, entropy_fraction * global_entropy))
        current_entropy = None
        for radius, local_entropy in entropy_pairs:
          if radius <= chosen_radius + 1e-12:
            current_entropy = local_entropy
          else:
            break
        if current_entropy is None or current_entropy < entropy_threshold:
          for radius, local_entropy in entropy_pairs:
            if radius >= chosen_radius - 1e-12 and local_entropy >= entropy_threshold:
              if cap_radius is None or radius <= cap_radius + 1e-12:
                chosen_radius = radius
              break
  except Exception:
    pass

  try:
    label_entropy_pairs_raw = label_entropy_by_cumulative_radius(
      training_data,
      ancestry_coordinate,
      num_intervals,
    )
    label_entropy_pairs = []
    for radius, local_entropy in label_entropy_pairs_raw:
      try:
        local_entropy = float(local_entropy)
      except Exception:
        continue
      if math.isfinite(local_entropy):
        label_entropy_pairs.append((clean_radius(radius), max(0.0, local_entropy)))

    if label_entropy_pairs:
      global_label_entropy = label_entropy_pairs[-1][1]
      if global_label_entropy > 0.0:
        label_threshold = max(0.03, min(0.22, 0.22 * global_label_entropy))
        current_label_entropy = None
        for radius, local_entropy in label_entropy_pairs:
          if radius <= chosen_radius + 1e-12:
            current_label_entropy = local_entropy
          else:
            break
        if current_label_entropy is None or current_label_entropy < label_threshold:
          for radius, local_entropy in label_entropy_pairs:
            if radius >= chosen_radius - 1e-12 and local_entropy >= label_threshold:
              if cap_radius is None or radius <= cap_radius + 1e-12:
                chosen_radius = radius
              break
  except Exception:
    pass

  try:
    if not heterogeneous and basis_count >= 120:
      ld_pairs_raw = target_ld_similarity_by_cumulative_radius(
        training_data,
        ancestry_coordinate,
        target_variant,
        num_intervals,
      )
      ld_pairs = []
      for radius, similarity in ld_pairs_raw:
        try:
          similarity = float(similarity)
        except Exception:
          continue
        radius = clean_radius(radius)
        if math.isfinite(similarity) and 0.0 <= similarity <= 1.0:
          ld_pairs.append((radius, similarity))

      if ld_pairs:
        current_similarity = None
        for radius, similarity in ld_pairs:
          if radius <= chosen_radius + 1e-12:
            current_similarity = similarity
          else:
            break
        best_similarity = max(similarity for _, similarity in ld_pairs)
        needed_similarity = min(0.85, max(0.58, 0.88 * best_similarity))
        if current_similarity is None or current_similarity < needed_similarity:
          for radius, similarity in ld_pairs:
            if radius >= chosen_radius - 1e-12 and similarity >= needed_similarity:
              chosen_radius = radius
              break
  except Exception:
    pass

  return clean_radius(chosen_radius)

