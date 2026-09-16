## AFRFocal3 build summary

The pickles in this folder were built by
`Data/FunsearchEvaluatorDataAFRFocal3/build_funsearch_evaluator_data_afr_focal.py`.

They are derived from `Data/FunsearchEvaluatorDataAFRFocal2`, but they are not a direct copy.

## What was reused from AFRFocal2

For each of these six outputs:

- `no_covariates_heldout.pkl`
- `no_covariates_test.pkl`
- `no_covariates_train.pkl`
- `with_covariates_heldout.pkl`
- `with_covariates_test.pkl`
- `with_covariates_train.pkl`

AFRFocal3 reuses the exact same subjects, the exact same split membership, and the exact same row order as AFRFocal2.

This was done by reading `Data/FunsearchEvaluatorDataAFRFocal2/output_row_tracking.pkl`, which records for every AFRFocal2 output row:

- the source raw OncoArray pickle name
- the source row number inside that pickle

Using that tracking file, the builder reloaded the corresponding raw rows from `Data/RawDataOncoArray`.

Because of this, for the same subject, these fields are the same in AFRFocal2 and AFRFocal3:

- phenotype
- dosage columns
- covariates, when present

## Why AFRFocal3 is different from AFRFocal2

The difference is in the ancestry coordinates `PC1` through `PC16`.

AFRFocal2 inherited already-standardized ancestry coordinates from the prepared OncoArray evaluator pickles in `Data/FunsearchEvaluatorDataOncoArray`.

AFRFocal3 does not reuse those standardized coordinates. Instead, it recomputes standardization from raw OncoArray ancestry coordinates, using only the subjects that appear in the AFRFocal3 outputs for that condition.

So AFRFocal3 is derived from AFRFocal2 in subject selection, but derived from `Data/RawDataOncoArray` for the actual ancestry-coordinate values.

## How the new ancestry coordinates were computed

The builder handled the two conditions separately:

- `no_covariates`
- `with_covariates`

For one condition, it pooled together all rows that appear in that condition's `train`, `test`, and `heldout` outputs, using the raw unstandardized `PC1` to `PC16` values from `Data/RawDataOncoArray`.

Then it fit one condition-wide transform:

- `a_star`: the 16-dimensional mean ancestry vector over the pooled raw rows
- `r`: the Euclidean radius around `a_star` that covers 95% of the pooled raw rows

Each subject ancestry vector `a` was then rewritten as:

`(a - a_star) / r`

This means:

- `no_covariates` has its own `a_star` and `r`
- `with_covariates` has its own `a_star` and `r`
- the same transform is shared within each condition across that condition's `train`, `test`, and `heldout` pickles

## Constants and extra outputs

The fitted standardization constants are written to `transformations.txt` in this folder.

That file records, for each condition:

- `sample_count`
- `a_star`
- `r`

This folder also includes two histogram PNG files showing the Euclidean distance between each subject's standardized ancestry coordinate in AFRFocal2 and the same subject's standardized ancestry coordinate in AFRFocal3, one histogram per condition.

## Subject counts by ancestry and phenotype

The counts below are taken from the pickle contents in this folder, with ancestry labels reconstructed from `output_row_tracking.pkl` source shard names.

### `no_covariates_heldout.pkl`

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| AFRICAN_ANCESTRY | 200 | 200 | 400 |
| ASIAN | 0 | 0 | 0 |
| EUROPEAN | 0 | 0 | 0 |

### `no_covariates_test.pkl`

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| AFRICAN_ANCESTRY | 200 | 200 | 400 |
| ASIAN | 10 | 10 | 20 |
| EUROPEAN | 30 | 30 | 60 |

### `no_covariates_train.pkl`

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| AFRICAN_ANCESTRY | 55 | 55 | 110 |
| ASIAN | 50 | 50 | 100 |
| EUROPEAN | 1400 | 1400 | 2800 |

### `with_covariates_heldout.pkl`

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| AFRICAN_ANCESTRY | 200 | 200 | 400 |
| ASIAN | 0 | 0 | 0 |
| EUROPEAN | 0 | 0 | 0 |

### `with_covariates_test.pkl`

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| AFRICAN_ANCESTRY | 200 | 200 | 400 |
| ASIAN | 10 | 10 | 20 |
| EUROPEAN | 30 | 30 | 60 |

### `with_covariates_train.pkl`

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| AFRICAN_ANCESTRY | 55 | 55 | 110 |
| ASIAN | 50 | 50 | 100 |
| EUROPEAN | 1400 | 1400 | 2800 |