# AFR Focal 2 Discovery Cohort Composition

This discovery cohort is built by pooling the prepared, already-standardized OncoArray evaluator pickles in `Data/FunsearchEvaluatorDataOncoArray` and resampling them into AFR-focused splits for the two evaluator conditions.

Compared with `Data/FunsearchEvaluatorDataAFRFocal`, this second AFR focal cohort uses explicit case/control quotas in every split.

## Source pool

The pooled OncoArray source available per condition is:

| Ancestry | Count |
| --- | ---: |
| African_Ancestry | 1465 |
| Asian | 629 |
| European | 29892 |
| Total | 31986 |

These totals are identical for the `no_covariates` and `with_covariates` conditions because the additional-covariates pickles are row-aligned versions of the same source subjects.

## Raw OncoArray data composition

The source raw OncoArray data in `Data/RawDataOncoArray` is split into cases and controls as follows:

| Ancestry | Controls | Cases | Total |
| --- | ---: | ---: | ---: |
| African_Ancestry | 471 | 994 | 1465 |
| Asian | 242 | 387 | 629 |
| European | 11110 | 18782 | 29892 |

## Split design

The AFR focal 2 builder writes three primary outputs per condition in `Data/FunsearchEvaluatorDataAFRFocal2`.

- Heldout: 400 rows total, with 200 African_Ancestry controls and 200 African_Ancestry cases.
- Test: 480 rows total, with 200 African_Ancestry controls, 200 African_Ancestry cases, 30 European controls, 30 European cases, 10 Asian controls, and 10 Asian cases.
- Train: 3010 rows total, with 55 African_Ancestry controls, 55 African_Ancestry cases, 1400 European controls, 1400 European cases, 50 Asian controls, and 50 Asian cases.

The heldout set is shared across both conditions at the logical sample level. Test and train are drawn independently per condition and are disjoint within each condition, but they are not forced to be disjoint across conditions.

## Realized composition

### no_covariates

| Split | African_Ancestry | Asian | European | Total |
| --- | ---: | ---: | ---: | ---: |
| heldout | 400 | 0 | 0 | 400 |
| test | 400 | 20 | 60 | 480 |
| train | 110 | 100 | 2800 | 3010 |

Detailed case/control composition:

- heldout: African_Ancestry 200 controls and 200 cases.
- test: African_Ancestry 200 controls and 200 cases, Asian 10 controls and 10 cases, European 30 controls and 30 cases.
- train: African_Ancestry 55 controls and 55 cases, Asian 50 controls and 50 cases, European 1400 controls and 1400 cases.

Source-shard composition:

- heldout: test_African_Ancestry 39, train_African_Ancestry 361.
- test: test_African_Ancestry 46, train_African_Ancestry 354, test_Asian 2, train_Asian 18, test_European 6, train_European 54.
- train: test_African_Ancestry 10, train_African_Ancestry 100, test_Asian 5, train_Asian 95, test_European 264, train_European 2536.

### with_covariates

| Split | African_Ancestry | Asian | European | Total |
| --- | ---: | ---: | ---: | ---: |
| heldout | 400 | 0 | 0 | 400 |
| test | 400 | 20 | 60 | 480 |
| train | 110 | 100 | 2800 | 3010 |

Detailed case/control composition:

- heldout: African_Ancestry 200 controls and 200 cases.
- test: African_Ancestry 200 controls and 200 cases, Asian 10 controls and 10 cases, European 30 controls and 30 cases.
- train: African_Ancestry 55 controls and 55 cases, Asian 50 controls and 50 cases, European 1400 controls and 1400 cases.

Source-shard composition:

- heldout: test_African_Ancestry_add_covs 39, train_African_Ancestry_add_covs 361.
- test: test_African_Ancestry_add_covs 34, train_African_Ancestry_add_covs 366, test_Asian_add_covs 1, train_Asian_add_covs 19, test_European_add_covs 10, train_European_add_covs 50.
- train: test_African_Ancestry_add_covs 14, train_African_Ancestry_add_covs 96, test_Asian_add_covs 11, train_Asian_add_covs 89, test_European_add_covs 309, train_European_add_covs 2491.

## Additional derived heldout subsets

Later analysis work also added three `no_covariates` helper pickles in this directory.

### `no_covariates_heldout_cal.pkl`

- Total rows: 100.
- Composition: African_Ancestry only, with 50 controls and 50 cases.
- Source shards: test_African_Ancestry 10, train_African_Ancestry 90.

### `no_covariates_heldout_test.pkl`

- Total rows: 300.
- Composition: African_Ancestry only, with 150 controls and 150 cases.
- Source shards: test_African_Ancestry 29, train_African_Ancestry 271.

### `no_covariates_heldout_cal2.pkl`

- Total rows: 220.
- Composition: African_Ancestry 200, European 15, Asian 5.
- Case/control breakdown: African_Ancestry 100 controls and 100 cases, European 8 controls and 7 cases, Asian 2 controls and 3 cases.
- This file is sampled from `no_covariates_test.pkl`, not from `no_covariates_heldout.pkl`.
- Source shards: test_African_Ancestry 22, train_African_Ancestry 178, test_European 2, train_European 13, train_Asian 5.

## Cross-condition overlap note

Because the two conditions are row-aligned views of the same logical subjects, overlap should be measured after stripping the `_add_covs` suffix from source pickle names.

For the current regenerated AFR focal 2 build, the logical-key overlap is:

- shared heldout overlap: 400 rows
- total discovery overlap: 626 rows
- same-split test overlap: 196 rows
- same-split train overlap: 302 rows
- `no_covariates` train versus `with_covariates` test overlap: 57 rows
- `no_covariates` test versus `with_covariates` train overlap: 71 rows

So the heldout split is identical across conditions by design, while the discovery splits are independent draws with substantial but not complete cross-condition overlap.