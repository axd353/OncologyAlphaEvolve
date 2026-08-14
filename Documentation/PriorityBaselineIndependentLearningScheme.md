# Independent Learning Scheme Baseline

This baseline is implemented in [PostProcesingData/prio_func_eval_baselines/independent_learning_scheme.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/independent_learning_scheme.py).

## Method

The Independent Learning Scheme baseline follows this ancestry-specific workflow:

1. Identify the unique ancestry groups present in the heldout set.
2. Concatenate the training and calibration datasets into one supervised training dataset.
3. For each discovered ancestry group `anc`, keep only the supervised training samples with ancestry `anc`.
4. Fit one ridge-regression model for that ancestry-specific training subset.
5. For each heldout subject, look up that subject's ancestry group and score the subject using the corresponding ancestry-specific model.
6. Report overall heldout ROC AUC and per-ancestry heldout ROC AUC.

The ancestry groups are recovered from the dataset lineage in `output_row_tracking.pkl`, not inferred from the standardized PC values themselves.

## Config options

- `alpha`: ridge penalty strength for each ancestry-specific ridge model, default `1.0`

Example baseline config entry:

```json
{
  "name": "Independent Learning Scheme",
  "enabled": true,
  "alpha": 1.0
}
```

## Failure mode

If the heldout set contains ancestry group `anc` but the combined training plus calibration data contains no samples from that same ancestry group, the baseline raises an error instead of silently falling back to another model.