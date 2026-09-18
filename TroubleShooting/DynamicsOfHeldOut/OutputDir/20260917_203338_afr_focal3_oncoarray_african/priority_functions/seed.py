# label=seed
# source_kind=seed
# cycle_index=None
# original_prio_function_path=/nfs/home/adas23/projects/AlphaEvolve/Collaterals/RunAFRFocal3/funsearch_priority_seed_constant4.py
# discovery_snapshot_path=/nfs/home/adas23/projects/AlphaEvolve/prio_func_disc_runs/oracle_priorityAfrFocal3_SimpleSeed_20260916_052346/program_db/bootstrap.pkl
# discovery_source_island_id=0
# discovery_reduced_score=0.5623413435306454
# discovery_scores_per_test={"combined": 0.5623413435306454, "mean": 0.5623413435306454, "no_covariates_fold_1": 0.5655101700903142, "no_covariates_fold_2": 0.5591725169709765, "simplicity": -9.0, "simplicity_bonus": 0.0}
# priority_source_hash=9ccd06ac08b86206dcff73f20dfa64f8d113cac41e6bacdcb81c9299e304e167
# reused_from_label=None
# selected_calibration_penalty=1.0
# heldout_subject_count=400
# heldout_auc_roc=0.5690500000000001

def priority(training_data, ancestry_coordinate, target_variant) -> float:
  return 4.0

