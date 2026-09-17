# label=cycle_0002
# source_kind=cycle_best
# cycle_index=2
# original_prio_function_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_051522/cycle_0002/best_prio.py
# discovery_snapshot_path=/lustre/isaac24/scratch/adas23/AlphaEvolve_prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_051522/cycle_0002/program_db_end.pkl
# discovery_source_island_id=2
# discovery_reduced_score=0.5918005264200448
# discovery_scores_per_test={"combined": 0.5918005264200448, "mean": 0.5855783041978225, "no_covariates_fold_1": 0.5646581733326885, "no_covariates_fold_2": 0.5684941592167061, "simplicity": -357.0, "simplicity_bonus": 0.006222222222222223, "with_covariates_fold_1": 0.5994429070260022, "with_covariates_fold_2": 0.6097179772158933}
# priority_source_hash=fe9df9d518230d76413e6966a5a54af5dcab23cbfdbf5922c6ff8e2a6b7e3cc7
# reused_from_label=None
# selected_calibration_penalty=0.1
# heldout_subject_count=400
# heldout_auc_roc=0.5585

def priority(training_data, ancestry_coordinate, target_variant) -> float:
  import math

  def finite_nonnegative(value, fallback=0.0):
    try:
      value = float(value)
    except Exception:
      return float(fallback)
    if math.isfinite(value) and value >= 0.0:
      return value
    return float(fallback)

  records = getattr(training_data, "records", ())
  dosage_keys = []

  def add_key(key):
    if key is not None and key not in dosage_keys:
      dosage_keys.append(key)

  add_key(getattr(target_variant, "dosage_field", None))
  add_key(getattr(target_variant, "name", None))

  try:
    column_index = int(getattr(target_variant, "column_index", -1))
  except Exception:
    column_index = -1

  if column_index >= 0:
    try:
      add_key(getattr(training_data, "variant_dosage_fields", ())[column_index])
    except Exception:
      pass
    try:
      add_key(getattr(training_data, "variant_names", ())[column_index])
    except Exception:
      pass

  usable = []
  for record in records:
    dosages = getattr(record, "variant_dosages", {})
    value = None
    for key in dosage_keys:
      try:
        if key in dosages:
          value = dosages[key]
          break
      except Exception:
        continue
    try:
      value = float(value)
    except Exception:
      continue
    if math.isfinite(value):
      usable.append(value)

  if len(usable) < 2:
    return 0.0

  if max(usable) - min(usable) <= 1e-12:
    return 0.0

  try:
    radius, effective_percentage = minimum_radius_for_training_percentage(
      training_data,
      ancestry_coordinate,
      target_variant,
      100.0,
    )
    if finite_nonnegative(effective_percentage) > 0.0:
      return finite_nonnegative(radius)
  except Exception:
    pass

  try:
    return finite_nonnegative(radius_for_percentage(training_data, ancestry_coordinate, 100.0), 4.0)
  except Exception:
    return 4.0

