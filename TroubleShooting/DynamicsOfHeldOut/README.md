# Dynamics Of Heldout

This helper evaluates the initial seed priority function plus every `cycle_XXXX/best_prio.py` in a FunSearch run on a heldout dataset. It reuses the same evaluation mechanics as `PostProcesingData.evaluate_priofunction`:

- calibrate each priority function on the configured `calibrating_pickle_path`
- evaluate heldout scores using `training_pickle_path + calibrating_pickle_path` as the reference set
- reuse the same ancestry-distance cache format and multiprocessing partition controls

Run it with:

```bash
PYTHONPATH=$PWD python -m TroubleShooting.DynamicsOfHeldOut.evaluate_run_dynamics TroubleShooting/DynamicsOfHeldOut/example_config.json
```

Inputs come from the JSON config:

- the FunSearch `run_dir`
- training, calibration, and heldout evaluator pickles
- `output_row_tracking.pkl` for ancestry assignment
- optional `heldout_target_ancestry_group` to score only one ancestry within heldout
- optional `distance_cache_dir` to reuse a cache from an earlier single-priority heldout evaluation

Outputs are written to one timestamped subdirectory under `TroubleShooting/DynamicsOfHeldOut/OutputDir`:

- `config.used.json`: resolved copy of the troubleshooting config
- `priority_functions/`: one copied `.py` file per seed or cycle-best priority function plus a `.metadata.json` sidecar describing where it was discovered in the original FunSearch run
- `heldout_priority_scores.csv`: one row per seed or cycle-best priority function and score columns
- `heldout_priority_scores.json`: machine-readable copy of the results
- `summary.md`: short human-readable summary
- `dynamics_of_heldout.progress.log`: progress log for the run

Repeated cycle winners are still listed as separate rows in the CSV. When two rows have identical priority-function source, the tool computes the heldout metrics once and marks later rows with `reused_from_label`, while preserving the original discovery-time metadata columns such as `discovery_score[...]`, including fold scores and simplicity.