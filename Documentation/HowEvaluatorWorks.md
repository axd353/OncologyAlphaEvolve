# How the Evaluator Works

This document describes the Procedure 2 evaluator used by the FunSearch pipeline to score evolved priority functions.

The main implementation lives in [funsearch_pipeline/evaluation/procedure2.py](funsearch_pipeline/evaluation/procedure2.py). The two entry points that matter most are:

- `prepare(...)`, which builds or loads the preprocessed oracle datasets.
- `evaluate_candidate(...)`, which loads a candidate priority function, scores it on each configured dataset pair, and returns the values registered in the program database.

## What the evaluator scores

For each candidate priority function, the evaluator produces one score per configured dataset pair. In the current pipeline, there are two pairs:

- the no-additional-covariates condition
- the additional-covariates condition

Each pair gets its own score because the same priority function is evaluated under both data conditions. The evaluator stores their average under `mean`, then the program database appends a registration-time `combined` score as the last entry. That final `combined` value is what upstream FunSearch uses for island ranking, best-program tracking, and prompt-function ordering.

The evaluator also records an auxiliary simplicity score for the candidate function body. `simplicity` is defined as the negative AST node count of the candidate function, so values closer to zero are simpler and more negative values are more complex. During registration into an island, the program database compares that simplicity against up to 100 existing functions from the destination island, computes a bounded `simplicity_bonus`, and appends `combined = mean + simplicity_bonus` as the final ranking score.

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

Because the priority-function contract objects are built from those prepared artifacts, direct and helper tools under `funsearch_pipeline/priority_tools/` should not normally encounter missing dosage values during evaluator-driven runs.

### from db-gap mec to funcsearch

The repository includes a standalone builder at `Data/build_funsearch_evaluator_data.py` that converts the raw MEC dbGaP shards under `Data/RawData/` into evaluator-ready pickles under `Data/FunsearchEvaluatorData/` without modifying the raw inputs.

For each condition (`no_covariates` and `with_covariates`), the builder first combines all six raw shards for that condition (`train_AA`, `train_JA`, `train_LA`, `test_AA`, `test_JA`, `test_LA`) and computes a shared ancestry transform on `PC1` through `PC16`:

- `a*` is the mean ancestry vector across all samples in that condition.
- `r` is the smallest radius such that at least 95% of those samples lie within Euclidean distance `r` of `a*`.
- Every output pickle replaces the original ancestry coordinates with `(a - a*) / r` while keeping the same column names.

The builder then writes three non-overlapping datasets per condition:

- `*_heldout.pkl`: 20% of `train_JA`, plus `M_ho` times as many samples from each of `train_AA` and `train_LA`. Because the raw `*_add_covs.pkl` files preserve the same row order as the corresponding base pickles, the builder reuses the same heldout row indices across `no_covariates` and `with_covariates`, so the two heldout outputs represent the same subjects.
- `*_test.pkl`: all of the raw test shards for that condition, plus an optional `P_add%` sample from the remaining `train_JA` rows and matched counts from the remaining `train_AA` and `train_LA` rows.
- `*_train.pkl`: every training row left after the heldout and optional test augmentation draws.

The same run also writes `transformations.txt`, which records the condition-specific ancestry center and radius, `build_funsearch_evaluator_data.log`, which records the supplied arguments, the output row counts, and the non-zero source-pickle contributions for every generated dataset, and `output_row_tracking.pkl`, which maps each output row back to its source raw-data pickle path and source row number before ancestry standardization.

When wiring these artifacts into a FunSearch run, the current `procedure2` backend can use `*_train.pkl` as its training source and `*_test.pkl` as its scoring source. The `*_heldout.pkl` files remain an explicit reserve split outside the current `procedure2` train/calibration/scoring flow.

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

- `simplicity`, an auxiliary score based on AST size of the priority function body.
- `mean`, the mean of the pair scores.

When that evaluated candidate is registered into an island, the program database appends:

- `simplicity_bonus`, computed from the candidate's simplicity rank relative to up to 100 baseline functions already registered in that island.
- `combined`, the final ranking score used by upstream FunSearch, defined as `mean + simplicity_bonus`.

When the evaluator moves from calibration to scoring, it uses the combined `oracle_train.pkl + calibration.pkl` data to estimate marginal effect sizes for the scoring subjects. The scoring split itself is only used for the final personalized-risk scoring and AUC calculation.

## Where each score comes from

### Pair score

The pair score is the median ROC AUC from bootstrapping the held-out scoring set. It measures how well the calibrated personalized risk scores align with the disease labels for that pair.

### Simplicity score

The simplicity score is computed as:

`simplicity = -(number of AST nodes in the candidate priority function)`

That means the direction is inverted relative to a raw size metric:

- higher `simplicity` values are simpler functions because they are less negative and closer to zero
- lower `simplicity` values are more complicated functions because they have more AST nodes and therefore become more negative

Examples from this run: a compact seed can have `simplicity = -93`, while a much larger evolved function can have `simplicity = -1144`. So `-93` is simpler than `-1144`.

This score is auxiliary only. It is stored in the program signature for diagnostics and diversity, but it is not the value upstream FunSearch uses to rank islands.

### Mean score

The mean score is the average of the dataset-pair scores. It is stored under `mean` for diagnostics and as the base input to the final ranking score, but it is no longer the last stored value once the candidate is registered into an island.

### Combined score

The combined score is the final scalar upstream FunSearch ranks on. It is appended last in the score signature during island registration:

`combined = mean + simplicity_bonus`

The simplicity bonus is computed by comparing the candidate's `simplicity` score against up to 100 deterministically sampled baseline functions already registered in the destination island.

- If the island is empty and this is the first registered function, `simplicity_bonus = 0.0`.
- Otherwise the candidate is placed on a simplicity rank scale relative to the baseline sample.
- The interpolation range is `[-Y, Y]`, where `Y = program_database.simplicity_bonus_max` from the config file.
- A uniquely simplest candidate gets `+Y`, a uniquely most complex candidate gets `-Y`, and intermediate ranks interpolate linearly between those endpoints.

Because `combined` is stored last, it is the value upstream FunSearch uses for island ranking, for choosing island founders during resets, and for biasing which prior functions are sampled into prompts.

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
    K --> L[Compare simplicity to destination-island baselines]
    L --> M[Append simplicity_bonus and combined]
```

In short: `prepare(...)` creates the data layout, and `evaluate_candidate(...)` uses that layout to score the priority function under each configured pair.