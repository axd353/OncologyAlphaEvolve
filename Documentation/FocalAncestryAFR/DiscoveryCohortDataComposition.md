# AFR Focal Discovery Cohort Composition

This discovery cohort is built by pooling the prepared, already-standardized OncoArray evaluator pickles in Data/FunsearchEvaluatorDataOncoArray and resampling them into AFR-focused splits for the two evaluator conditions.

## Source pool

The pooled OncoArray source available per condition is:

| Ancestry | Count |
| --- | ---: |
| African_Ancestry | 1465 |
| Asian | 629 |
| European | 29892 |
| Total | 31986 |

These totals are identical for the no_covariates and with_covariates conditions because the additional-covariates pickles are row-aligned versions of the same source subjects.

## Split design

The AFR focal builder writes three outputs per condition in Data/FunsearchEvaluatorDataAFRFocal.

- Heldout: 500 African_Ancestry rows.
- Test: 540 rows total, with 375 African_Ancestry rows and 165 random non-African_Ancestry rows.
- Train: 3000 rows total, with 400 African_Ancestry rows, 200 Asian rows, and 2400 European rows.

The heldout set is shared across both conditions at the sample level.

## Realized composition

### no_covariates

| Split | African_Ancestry | Asian | European | Total |
| --- | ---: | ---: | ---: | ---: |
| heldout | 500 | 0 | 0 | 500 |
| test | 375 | 0 | 165 | 540 |
| train | 400 | 200 | 2400 | 3000 |

Source-shard composition:

- heldout: test_African_Ancestry 51, train_African_Ancestry 449.
- test: test_African_Ancestry 40, train_African_Ancestry 335, test_European 20, train_European 145.
- train: test_African_Ancestry 38, train_African_Ancestry 362, test_Asian 18, train_Asian 182, test_European 222, train_European 2178.

### with_covariates

| Split | African_Ancestry | Asian | European | Total |
| --- | ---: | ---: | ---: | ---: |
| heldout | 500 | 0 | 0 | 500 |
| test | 375 | 0 | 165 | 540 |
| train | 400 | 200 | 2400 | 3000 |

Source-shard composition:

- heldout: test_African_Ancestry 51, train_African_Ancestry 449.
- test: test_African_Ancestry 31, train_African_Ancestry 344, test_European 18, train_European 147.
- train: test_African_Ancestry 43, train_African_Ancestry 357, test_Asian 23, train_Asian 177, test_European 258, train_European 2142.

For experiemnts on East Asian ancestry we had

Heldout: AA 57, JA 19, LA 57
Test: AA 165, JA 76, LA 114
Train: AA 1165, JA 51, LA 706

## Constraint note

The request to make train and test fully different across the two conditions is not feasible for African_Ancestry at these quotas. The pooled OncoArray source contains 1465 African_Ancestry rows total, while fully disjoint cross-condition AFR allocations would require 2050 unique AFR rows:

- 500 shared heldout rows
- 375 no_covariates test rows
- 400 no_covariates train rows
- 375 with_covariates test rows
- 400 with_covariates train rows

Because of that supply constraint, the current AFR focal builder keeps heldout identical across conditions and minimizes discovery-split overlap greedily in shared logical-key space. In the current regenerated build, discovery overlap is 585 rows total, with 0 same-split overlap in test and 25 same-split overlap in train.