# TL-PR Baseline

This baseline is implemented in [PostProcesingData/prio_func_eval_baselines/tl_pr.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/tl_pr.py) with shared helpers in [PostProcesingData/prio_func_eval_baselines/transfer_learning_common.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/transfer_learning_common.py).

The implementation is adapted from the GPTL prototype code, but only the reusable fitting logic was brought over. Notebook orchestration, plotting, and unrelated comparison code were left out.

## Method summary

For each heldout ancestry group:

- The source pool is all training and calibration rows from the other ancestry groups.
- A source logistic model is fit on that pooled non-target source set.
- If the target ancestry has enough labeled rows, TL-PR fits an elastic-net penalized logistic model around the source coefficients.
- Cross-validation over `alpha` and `lambda` chooses the target-ancestry adaptation strength.
- If the target ancestry is too small or lacks both classes, the source-only model is used as a fallback.

The final output is one heldout score per row, followed by overall and per-ancestry ROC AUC in the shared evaluation report.

## Supported config options

Example baseline entry:

```json
{
  "name": "TL-PR",
  "enabled": true,
  "alpha_grid": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "n_lambdas": 30,
  "lambda_min_ratio": 0.01,
  "ridge_grid_max": 10000.0,
  "ridge_grid_min": 1.0,
  "cv_folds": 2,
  "max_iter": 600,
  "learning_rate": 0.05,
  "tol": 0.000001,
  "source_n_iter": 3000,
  "source_learning_rate": 0.05,
  "source_l2": 0.0001,
  "min_target_n": 20,
  "min_class_count": 2,
  "class_weight": "balanced",
  "center_dosages": true,
  "scale_dosages": false,
  "seed": 0
}
```

Important options:

- `alpha_grid`: elastic-net mixing values searched during target-ancestry adaptation
- `n_lambdas`: number of shrinkage values searched for each `alpha`
- `cv_folds`: stratified folds used to select `alpha` and `lambda`
- `max_iter`: optimization iterations for each TL-PR fit
- `min_target_n`: minimum target-ancestry row count required before adaptation is attempted

## Data expectations

TL-PR expects evaluator-ready DataFrames with:

- the phenotype label column
- one or more dosage columns whose names use the shared dosage prefix
- ancestry-group assignments supplied by the main evaluator from `output_row_tracking.pkl`

The baseline does not require extra files beyond the normal evaluation config.