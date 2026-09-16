## AFRFocal5 build summary

The pickles in this folder were built by
`Data/FunsearchEvaluatorDataAFRFocal5/build_funsearch_evaluator_data_afr_focal.py`.

They are derived from `Data/FunsearchEvaluatorDataAFRFocal3`, but they are not a direct copy.

## What was reused from AFRFocal3

For each of these six outputs:

- `no_covariates_heldout.pkl`
- `no_covariates_test.pkl`
- `no_covariates_train.pkl`
- `with_covariates_heldout.pkl`
- `with_covariates_test.pkl`
- `with_covariates_train.pkl`

AFRFocal5 reuses the exact same subjects and the exact same row order as AFRFocal3 for:

- both `train` outputs
- both `heldout` outputs
- the original rows already present in both `test` outputs

This was done by reading `Data/FunsearchEvaluatorDataAFRFocal3/output_row_tracking.pkl`, which records for every AFRFocal3 output row:

- the source raw OncoArray pickle name
- the source row number inside that pickle

Using that tracking file, the builder reloaded the corresponding raw rows from `Data/RawDataOncoArray`.

Because of this, for every reused subject, these fields are the same in AFRFocal3 and AFRFocal5:

- phenotype
- dosage columns
- covariates, when present

## What changed relative to AFRFocal3

AFRFocal5 enlarges each `test` output by 320 European rows:

- 160 controls
- 160 cases

The two conditions were handled independently:

- `no_covariates`
- `with_covariates`

For one condition, the builder first collected every raw source row already used anywhere in that condition's AFRFocal3 `train`, `test`, and `heldout` outputs.

It then scanned the unused European raw rows in this deterministic order:

1. `train_European*.pkl`
2. `test_European*.pkl`

within each source file's natural row order.

From that ordered pool, it took:

- the first 160 unused controls
- the first 160 unused cases

and appended them to the end of that condition's `test` output.

In the actual build that created this folder, all 320 added rows came from the `train_European` shard for that condition because that first source already contained enough unused controls and cases.

Specifically:

- `no_covariates_test.pkl` received 160 controls and 160 cases from `train_European.pkl`
- `with_covariates_test.pkl` received 160 controls and 160 cases from `train_European_add_covs.pkl`

No tracked source row is reused twice within a condition in AFRFocal5.

## How the ancestry coordinates were recomputed

Like AFRFocal3, AFRFocal5 does not reuse previously standardized ancestry coordinates. Instead, for each condition it reloads the selected raw OncoArray rows and refits one condition-wide ancestry transform from those raw `PC1` through `PC16` values.

For one condition, the pooled fitting rows are all rows appearing in that condition's:

- `train`
- `heldout`
- expanded `test`

including the 320 added European `test` rows.

The fitted transform is the same scalar-radius transform used by AFRFocal3:

- `a_star`: the 16-dimensional mean ancestry vector over the pooled raw rows
- `r`: the Euclidean radius around `a_star` that covers 95% of the pooled raw rows

Each ancestry vector `a` was rewritten as:

`(a - a_star) / r`

This means:

- `no_covariates` has its own `a_star` and `r`
- `with_covariates` has its own `a_star` and `r`
- the extra European `test` rows are included when fitting those parameters

The fitted constants are written to `transformations.txt` in this folder.

## Files written by the build

The builder writes:

- `no_covariates_heldout.pkl`
- `no_covariates_test.pkl`
- `no_covariates_train.pkl`
- `with_covariates_heldout.pkl`
- `with_covariates_test.pkl`
- `with_covariates_train.pkl`
- `output_row_tracking.pkl`
- `transformations.txt`
- `build_funsearch_evaluator_data.log`

`output_row_tracking.pkl` contains one row per output row and records:

- `output_pickle_name`
- `output_row_number`
- `source_pickle_name`
- `source_pickle_path`
- `source_row_number`

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
| EUROPEAN | 190 | 190 | 380 |

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
| EUROPEAN | 190 | 190 | 380 |

### `with_covariates_train.pkl`

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| AFRICAN_ANCESTRY | 55 | 55 | 110 |
| ASIAN | 50 | 50 | 100 |
| EUROPEAN | 1400 | 1400 | 2800 |