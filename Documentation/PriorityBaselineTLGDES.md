# TL-GDES Baseline

This baseline is implemented in [PostProcesingData/prio_func_eval_baselines/tl_gdes.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/tl_gdes.py) with shared helpers in [PostProcesingData/prio_func_eval_baselines/transfer_learning_common.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/transfer_learning_common.py).

The implementation is adapted from the GPTL prototype code, but it is integrated to match the priority-function evaluator contract rather than copied as notebook-style analysis code.

## Method summary

For each heldout ancestry group:

- The source pool is all training and calibration rows from the other ancestry groups.
- A source logistic model is fit on that pooled non-target source set.
- If the target ancestry has enough labeled rows, the source coefficients are adapted toward the target ancestry with gradient-descent early stopping.
- If the target ancestry is too small or lacks both classes, the source-only model is used as a fallback.

The final output is one heldout score per row, followed by overall and per-ancestry ROC AUC in the shared evaluation report.

## Supported config options

Example baseline entry:

```json
{
  "name": "TL-GDES",
  "enabled": true,
  "max_iter": 100,
  "learning_rate": 0.05,
  "source_n_iter": 3000,
  "source_learning_rate": 0.05,
  "source_l2": 0.0001,
  "target_l2": 0.0,
  "cal_fraction": 0.25,
  "min_target_n": 20,
  "min_class_count": 2,
  "class_weight": "balanced",
  "center_dosages": true,
  "scale_dosages": false,
  "seed": 0
}
```

Important options:

- `max_iter`: maximum target-adaptation iterations tested during early stopping
- `source_n_iter`: source-model training iterations
- `min_target_n`: minimum target-ancestry row count required before adaptation is attempted
- `min_class_count`: minimum examples per class required within the target ancestry
- `class_weight`: may be `"balanced"` to reduce class-imbalance effects

## Data expectations

TL-GDES expects evaluator-ready DataFrames with:

- the phenotype label column
- one or more dosage columns whose names use the shared dosage prefix
- ancestry-group assignments supplied by the main evaluator from `output_row_tracking.pkl`

The baseline does not require extra files beyond the normal evaluation config.