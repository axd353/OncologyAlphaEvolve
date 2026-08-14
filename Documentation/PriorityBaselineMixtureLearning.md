# Mixture Learning Baseline

This baseline is implemented in [PostProcesingData/prio_func_eval_baselines/mixture_learning.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/mixture_learning.py).

## Method

The Mixture Learning baseline:

1. Concatenates the training and calibration datasets into one supervised training dataset.
2. Uses all dosage columns plus the ancestry coordinate columns as features.
3. Fits one ridge-regression model on that combined training set.
4. Applies the fitted model to all heldout subjects.
5. Reports overall heldout ROC AUC and per-ancestry heldout ROC AUC.

This is a single pooled model across ancestries. It does not train separate models per ancestry group.

## Config options

- `alpha`: ridge penalty strength, default `1.0`

Example baseline config entry:

```json
{
  "name": "Mixture Learning",
  "enabled": true,
  "alpha": 1.0
}
```

## Data requirements

The current implementation requires:

- at least one dosage column,
- the default ancestry columns,
- the default label column.