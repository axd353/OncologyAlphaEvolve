# Evaluate Best Priority Functions Across All Cycles

This flow evaluates `best_prio.py` across all completed `cycle_XXXX/` directories in one FunSearch run.

Implementation entry point:

- [PostProcesingData/evaluate_priofunction_all_cycles.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction_all_cycles.py)

Run it from the repository root:

```bash
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
PYTHONPATH=$PWD python -m PostProcesingData.evaluate_priofunction_all_cycles \
  PostProcesingData/evaluate_priofunction_all_cycles.example.json
```

Required config fields:

- `run_dir`: FunSearch run directory containing `cycle_XXXX/best_prio.py` files.
- `target_ancestry_group`: one ancestry code from `supported_ancestry_groups`, for example `AA`.

Behavior:

- The command scans all completed cycle directories under `run_dir` and evaluates each `best_prio.py`.
- If multiple cycles contain identical `best_prio.py` contents, the first result is reused and later cycles are not reevaluated.
- Calibration uses only rows from `calibrating_pickle_path` whose ancestry matches `target_ancestry_group`.
- Heldout scoring uses only rows from `heldout_pickle_path` whose ancestry matches `target_ancestry_group`.
- The training reference pool for calibration still uses all rows from `training_pickle_path`.
- The training reference pool for heldout scoring uses all rows from `training_pickle_path` plus all rows from `calibrating_pickle_path`, even though only target-ancestry heldout rows are scored.
- Configured baselines are run once for the same target-ancestry heldout slice and written into the run-level report.

Outputs:

- A run-level JSON report saved in `run_dir`, default file name `best_prio.all_cycles.<TARGET>.evaluation_report.json`.
- A progress log under the configured or default distance-cache directory.

Example config:

- [PostProcesingData/evaluate_priofunction_all_cycles.example.json](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction_all_cycles.example.json)