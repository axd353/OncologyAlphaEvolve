# label=cycle_0004
# source_kind=cycle_best
# cycle_index=4
# original_prio_function_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_051522/cycle_0004/best_prio.py
# discovery_snapshot_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_051522/cycle_0004/program_db_end.pkl
# discovery_source_island_id=1
# discovery_reduced_score=0.5938545413033014
# discovery_scores_per_test={"combined": 0.5938545413033014, "mean": 0.5938545413033014, "no_covariates_fold_1": 0.5558413965284932, "no_covariates_fold_2": 0.5746320804306231, "simplicity": -1034.0, "simplicity_bonus": 0.0, "with_covariates_fold_1": 0.6352442884855207, "with_covariates_fold_2": 0.6097003997685687}
# priority_source_hash=54677c73d53c606ddfe4c6c51d62eb35075863fd4c8d464fd16fa9424478d0c1
# reused_from_label=cycle_0003
# selected_calibration_penalty=0.1
# heldout_subject_count=400
# heldout_auc_roc=0.5331750000000001

def priority(training_data, ancestry_coordinate, target_variant) -> float:
  import math

  def finite_radius(value, default=0.0):
      try:
          value = float(value)
      except Exception:
          return float(default)
      if math.isfinite(value) and value >= 0.0:
          return value
      return float(default)

  def full_radius():
      try:
          radius, _ = minimum_radius_for_training_percentage(
              training_data, ancestry_coordinate, target_variant, 100.0
          )
          return finite_radius(radius)
      except Exception:
          try:
              return finite_radius(radius_for_percentage(training_data, ancestry_coordinate, 100.0))
          except Exception:
              return 0.0

  try:
      records = getattr(training_data, "records")
      sample_count = int(getattr(training_data, "sample_count", len(records)))
  except Exception:
      return 0.0

  if sample_count <= 0:
      return 0.0

  dosage_keys = []
  for key in (getattr(target_variant, "dosage_field", None), getattr(target_variant, "name", None)):
      if key is not None and key not in dosage_keys:
          dosage_keys.append(key)

  try:
      column_index = int(getattr(target_variant, "column_index"))
      if 0 <= column_index < len(training_data.variant_dosage_fields):
          key = training_data.variant_dosage_fields[column_index]
          if key not in dosage_keys:
              dosage_keys.append(key)
      if 0 <= column_index < len(training_data.variant_names):
          key = training_data.variant_names[column_index]
          if key not in dosage_keys:
              dosage_keys.append(key)
  except Exception:
      pass

  dosages = []
  try:
      for record in records:
          value = None
          variant_dosages = getattr(record, "variant_dosages", {})
          for key in dosage_keys:
              if key in variant_dosages:
                  value = variant_dosages.get(key)
                  break
          try:
              value = float(value)
          except Exception:
              continue
          if math.isfinite(value):
              dosages.append(value)
  except Exception:
      return full_radius()

  if len(dosages) < 2 or max(dosages) - min(dosages) <= 1e-12:
      return full_radius()

  if sample_count < 10:
      return full_radius()

  n_balls = max(3, min(12, int(math.sqrt(sample_count))))
  min_samples = max(5, min(25, sample_count // n_balls))

  heterogeneity = 0.0
  first_break = None
  try:
      changes = standardized_effect_change_by_interval(
          training_data,
          ancestry_coordinate,
          target_variant,
          n_balls,
          min_samples,
      )
      for index, change in enumerate(changes):
          try:
              change = float(change)
          except Exception:
              continue
          if not math.isfinite(change):
              continue
          heterogeneity = max(heterogeneity, change)
          if first_break is None and change >= 2.0:
              first_break = index
  except Exception:
      pass

  if first_break is not None:
      percentage = max(15.0, 100.0 * float(first_break + 1) / float(n_balls))
  elif heterogeneity >= 1.5:
      percentage = 40.0
  elif heterogeneity >= 0.75:
      percentage = 55.0
  else:
      percentage = 75.0

  try:
      novelty = float(ancestry_novelty_score(training_data, ancestry_coordinate))
      if math.isfinite(novelty):
          if novelty >= 2.0:
              percentage += 20.0
          elif novelty >= 1.25:
              percentage += 10.0
          elif novelty <= 0.6:
              percentage -= 10.0
  except Exception:
      pass

  min_percentage = min(35.0, 100.0 * float(min(30, sample_count)) / float(sample_count))
  percentage = max(min_percentage, min(95.0, percentage))

  try:
      target_radius, _ = minimum_radius_for_training_percentage(
          training_data,
          ancestry_coordinate,
          target_variant,
          percentage,
      )
      target_radius = finite_radius(target_radius)
  except Exception:
      target_radius = full_radius()

  try:
      entropy_pairs = dosage_entropy_by_cumulative_radius(
          training_data,
          ancestry_coordinate,
          target_variant,
          n_balls,
      )
      se_pairs = effect_size_standard_error_by_cumulative_radius(
          training_data,
          ancestry_coordinate,
          target_variant,
          n_balls,
          min_samples,
      )

      first_entropy_radius = None
      candidates = []

      for index, pair in enumerate(entropy_pairs):
          radius, entropy = pair
          radius = finite_radius(radius)
          try:
              entropy = float(entropy)
          except Exception:
              entropy = 0.0

          if radius + 1e-12 < target_radius:
              continue
          if not math.isfinite(entropy) or entropy <= 1e-9:
              continue

          if first_entropy_radius is None:
              first_entropy_radius = radius

          se = math.inf
          if index < len(se_pairs):
              try:
                  se = float(se_pairs[index][1])
              except Exception:
                  se = math.inf

          if math.isfinite(se):
              candidates.append((radius, se))

      if candidates:
          best_se = min(se for _, se in candidates)
          if heterogeneity >= 1.5:
              tolerance = 2.0
          elif heterogeneity >= 0.75:
              tolerance = 1.6
          else:
              tolerance = 1.3
          se_limit = best_se * tolerance + 0.25

          for radius, se in candidates:
              if se <= se_limit:
                  return finite_radius(radius)

          return finite_radius(candidates[-1][0])

      if first_entropy_radius is not None:
          return finite_radius(first_entropy_radius)
  except Exception:
      pass

  return finite_radius(target_radius)

