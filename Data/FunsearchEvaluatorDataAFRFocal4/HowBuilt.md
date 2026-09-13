## AFRFocal4 build summary

The pickles in this folder are built by
`Data/FunsearchEvaluatorDataAFRFocal4/build_funsearch_evaluator_data_afr_focal.py`.

They are derived from `Data/FunsearchEvaluatorDataAFRFocal2`, but they are not a direct copy.

## What was reused from AFRFocal2

For each of these six outputs:

- `no_covariates_heldout.pkl`
- `no_covariates_test.pkl`
- `no_covariates_train.pkl`
- `with_covariates_heldout.pkl`
- `with_covariates_test.pkl`
- `with_covariates_train.pkl`

AFRFocal4 reuses the exact same subjects, the exact same split membership, and the exact same row order as AFRFocal2.

This is done by reading `Data/FunsearchEvaluatorDataAFRFocal2/output_row_tracking.pkl`, which records for every AFRFocal2 output row:

- the source raw OncoArray pickle name
- the source row number inside that pickle

Using that tracking file, the builder reloads the corresponding raw rows from `Data/RawDataOncoArray`.

Because of this, for the same subject, these fields are the same in AFRFocal2 and AFRFocal4:

- phenotype
- dosage columns
- covariates, when present

## Why AFRFocal4 is different

The difference is in the ancestry coordinates `PC1` through `PC16`.

AFRFocal4 does not reuse previously standardized coordinates. Instead, for each condition it recomputes standardization from the raw OncoArray ancestry coordinates using the subjects that appear in that condition's `train`, `test`, and `heldout` outputs.

The two conditions are handled separately:

- `no_covariates`
- `with_covariates`

Each condition therefore has its own ancestry transform, shared across that condition's `train`, `test`, and `heldout` pickles.

## Coordinate-wise ancestry standardization

For one condition, the builder pools together every selected raw row from all source shards that feed that condition's outputs and stacks `PC1` through `PC16`.

It then computes:

- `a_star`: the 16-dimensional mean ancestry vector across the pooled rows
- `r`: a 16-dimensional vector where `r_i` is the smallest radius such that at least 95% of pooled samples satisfy `|PC_i - a_star_i| <= r_i`

For every output pickle row with original raw ancestry vector `a_old`, the builder rewrites ancestry coordinates coordinate-wise as:

`a_new_i = (a_old_i - a_star_i) / r_i`

for `i = 1, 2, ..., 16`.

This differs from AFRFocal3, which used one scalar radius shared across all 16 coordinates.

## Extra outputs

The fitted constants are written to `transformations.txt` in this folder.

For each condition, that file records:

- `sample_count`
- `a_star`
- `r`

This folder also includes one raw-ancestry PCA boxplot PNG for each ancestry group:

- `AFRICAN_ANCESTRY.png`
- `ASIAN.png`
- `EUROPEAN.png`

Each of those plots is built from the raw non-covariate OncoArray train and test shards for that ancestry group. The `_add_covs` raw files are not used for the plots so the same subjects are not counted twice.