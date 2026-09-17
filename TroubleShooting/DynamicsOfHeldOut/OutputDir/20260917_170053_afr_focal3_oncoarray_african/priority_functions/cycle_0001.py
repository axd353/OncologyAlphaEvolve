# label=cycle_0001
# source_kind=cycle_best
# cycle_index=1
# original_prio_function_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_051522/cycle_0001/best_prio.py
# discovery_snapshot_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_051522/cycle_0001/program_db_end.pkl
# discovery_source_island_id=3
# discovery_reduced_score=0.5907376985263634
# discovery_scores_per_test={"combined": 0.5907376985263634, "mean": 0.5891376985263633, "no_covariates_fold_1": 0.5517848003621829, "no_covariates_fold_2": 0.5738772161223209, "simplicity": -826.0, "simplicity_bonus": 0.0015999999999999999, "with_covariates_fold_1": 0.6293218092490871, "with_covariates_fold_2": 0.6015669683718625}
# priority_source_hash=3420b77907748394a7611a0b185efe9d4a97c8fd3e9c8098b8c7dc69ecbb705b
# reused_from_label=None
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
      sample_count = int(getattr(training_data, "sample_count", len(training_data.records)))
  except Exception:
      return 0.0

  if sample_count <= 0:
      return 0.0

  dosages = []
  dosage_field = getattr(target_variant, "dosage_field", None)
  variant_name = getattr(target_variant, "name", None)

  try:
      for record in training_data.records:
          value = None
          if dosage_field is not None:
              value = record.variant_dosages.get(dosage_field)
          if value is None and variant_name is not None:
              value = record.variant_dosages.get(variant_name)
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
          if math.isfinite(change) and change > heterogeneity:
              heterogeneity = change
          if first_break is None and math.isfinite(change) and change >= 2.0:
              first_break = index
  except Exception:
      changes = []

  if first_break is not None:
      percentage = 100.0 * float(first_break + 1) / float(n_balls)
      percentage = max(15.0, percentage)
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

  best_entropy_radius = None
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

      for index, pair in enumerate(entropy_pairs):
          radius, entropy = pair
          radius = finite_radius(radius)
          try:
              entropy = float(entropy)
          except Exception:
              entropy = 0.0

          if radius + 1e-12 < target_radius or not math.isfinite(entropy) or entropy <= 1e-9:
              continue

          if best_entropy_radius is None:
              best_entropy_radius = radius

          se = math.inf
          if index < len(se_pairs):
              try:
                  se = float(se_pairs[index][1])
              except Exception:
                  se = math.inf

          if math.isfinite(se):
              return finite_radius(radius)

      if best_entropy_radius is not None:
          return finite_radius(best_entropy_radius)
  except Exception:
      pass

  return finite_radius(target_radius)

