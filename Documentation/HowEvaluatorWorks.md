# How the Evaluator Works

This document describes the Procedure 2 evaluator used by the FunSearch pipeline to score evolved priority functions.

The main implementation lives in [funsearch_pipeline/evaluation/procedure2.py](funsearch_pipeline/evaluation/procedure2.py). The two entry points that matter most are:

- `prepare(...)`, which builds or loads the preprocessed oracle datasets.
- `evaluate_candidate(...)`, which loads a candidate priority function, scores it on each configured dataset pair, and returns the values registered in the program database.

## What the evaluator scores

For each candidate priority function, the evaluator produces one score per configured dataset pair. In the current pipeline, there are two pairs:

- the no-additional-covariates condition
- the additional-covariates condition

Each pair gets its own score because the same priority function is evaluated under both data conditions. The final value used by upstream FunSearch is the mean of the pair scores, stored under `mean` as the last score entry.

The evaluator also records an auxiliary simplicity score for the candidate function body. This is stored before `mean`, so it is visible in the program signature but does not control island ranking.

## The three oracle parts

The oracle is split into three pieces:

1. Priority function
2. Effect size calculator
3. Calibration / scoring model

The priority function chooses a radius for a target subject and target variant. The effect size calculator uses that radius to estimate a local marginal effect. The evaluator then turns those effect sizes into calibrated risk scores and computes ROC AUC.

### 1. Priority function

Signature:

```python
def priority(training_data, ancestry_coordinate, target_variant) -> float:
    ...
```

The evaluator passes strict contract objects into this function, not raw pandas DataFrames or adapter helpers:

- `training_data` is a normalized Oracle-Train payload containing the training records, variant names, variant dosage fields, covariate names, and sample counts.
- `ancestry_coordinate` is the normalized target ancestry vector.
- `target_variant` is the normalized target variant object, including the logical name, dosage field, and column index.

The priority function returns a non-negative radius.

### 2. Effect size calculator

Signature:

```python
effect_size_calculator(training_data, ancestry_coordinate, target_variant, radius) -> float
```

For each scoring subject and each variant, the evaluator calls the priority function to get a radius, then passes that radius into `effect_size_calculator(...)` to estimate the local marginal effect size.

That effect size is the estimate of `\hat b_j(a)` for the current ancestry point `a` and target variant `j`.

### 3. Calibration / scoring model

After the effect sizes are computed, the evaluator forms oracle-derived features:

```python
X_ij = G_ij * \hat b_j(a_i)
```

where `G_ij` is the dosage for subject `i` and variant `j`.

The evaluator fits a ridge-penalized logistic regression on the calibration split:

```python
Pr(y_i = 1) = sigmoid(alpha + gamma^T c_i + sum_j lambda_j X_ij)
```

Then it scores the held-out scoring set with the fitted coefficients and computes ROC AUC.

## How `prepare(...)` works

`prepare(...)` creates the artifacts that the evaluator needs later in `evaluate_candidate(...)`.

For each configured dataset pair, it does the following:

1. Concatenate the raw training pickle shards.
2. Split the combined training set into two parts using `evaluator.oracle_train_fraction`.
3. Write the first part to `oracle_train.pkl`.
4. Write the second part to `calibration.pkl`.
5. Concatenate the raw testing pickle shards and write them to `scoring.pkl`.

If the evaluator is configured with already prepared paths (`oracle_train_pickle`, `calibration_pickle`, `scoring_pickle`), those are reused instead of rebuilding the split.

In normal runs, worker-local evaluators also call `prepare(...)`, but they reuse the same prepared artifacts if the manifest or pickles already exist. The evaluator only creates the three pickles the first time a pair is materialized in a run.

The first time a pair is actually materialized, the evaluator logs:

- the raw source pickle paths
- the output artifact paths
- the sample counts written to each artifact

### Missing-value imputation during preparation

Before the prepared pickles are written, the evaluator imputes missing feature values in the raw pandas DataFrames.

- dosage columns are mean-imputed
- additional covariate columns are mode-imputed

This happens in `prepare(...)`, not during candidate scoring, so the persisted `oracle_train.pkl`, `calibration.pkl`, and `scoring.pkl` artifacts already contain the cleaned values used by later evaluation steps. The covariate columns are also cast to float at this stage so pandas nullable `NA` values do not leak into downstream feature extraction.

## How `evaluate_candidate(...)` works

`evaluate_candidate(...)` is where the candidate priority function is actually used.

For each prepared dataset pair, it does this in order:

1. Load `oracle_train.pkl`, `calibration.pkl`, and `scoring.pkl`.
2. Extract the dosage column names from the oracle-train data.
3. Build the strict priority-function contract objects.
4. For each scoring subject and each variant:
   - call the priority function to get a radius
   - call `effect_size_calculator(...)` with that radius
   - multiply the estimated effect size by the subject dosage to get an oracle contribution
5. Fit the ridge-penalized logistic calibration model on the calibration split.
6. Apply the fitted model to the scoring split.
7. Compute ROC AUC on the scoring labels.
8. Bootstrap the scoring set AUC and keep the median bootstrap AUC as the pair score.

The evaluator returns one pair score per dataset pair, plus:

- `simplicity`, an auxiliary score based on AST size of the priority function body. This is not the core ranking signal; it is stored before `mean` for diagnostics and diversity, while `mean` is what upstream FunSearch uses for ranking.
- `mean`, the mean of the pair scores, which is the final value used by upstream FunSearch for ranking and best-program tracking

When the evaluator moves from calibration to scoring, it uses the combined `oracle_train.pkl + calibration.pkl` data to estimate marginal effect sizes for the scoring subjects. The scoring split itself is only used for the final personalized-risk scoring and AUC calculation.

## Where each score comes from

### Pair score

The pair score is the median ROC AUC from bootstrapping the held-out scoring set. It measures how well the calibrated personalized risk scores align with the disease labels for that pair.

### Simplicity score

The simplicity score is a negative structural complexity score computed from the candidate function's AST. Smaller functions get a better simplicity value.

### Mean score

The mean score is the average of the dataset-pair scores. It is stored last under `mean`, and that is the value upstream FunSearch uses for island ranking.

## Data flow summary

The flow for one candidate is:

```mermaid
flowchart TD
    A[Prepared oracle-train/calibration/scoring pickles] --> B[Load candidate priority function]
    B --> C[Build strict oracle contract objects]
    C --> D[Call priority_function(training_data, ancestry_coordinate, target_variant)]
    D --> E[Call effect_size_calculator with returned radius]
    E --> F[Build oracle contribution G_ij * b_hat_j(a_i)]
    F --> G[Fit ridge logistic calibration]
    G --> H[Score held-out scoring set]
    H --> I[Bootstrap ROC AUC and take median]
    I --> J[Pair score]
    J --> K[Mean across pairs]
```

In short: `prepare(...)` creates the data layout, and `evaluate_candidate(...)` uses that layout to score the priority function under each configured pair.