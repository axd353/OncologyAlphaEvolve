# How the Evaluator Works

This document describes the Procedure 2 evaluator used by the FunSearch pipeline to score evolved priority functions.

The main implementation lives in [funsearch_pipeline/evaluation/procedure2.py](funsearch_pipeline/evaluation/procedure2.py). The two entry points that matter most are:

- `prepare(...)`, which builds or loads the preprocessed oracle datasets.
- `evaluate_candidate(...)`, which loads a candidate priority function, scores it on each configured dataset pair, and returns the values registered in the program database.

## What the evaluator scores

For each candidate priority function, the evaluator produces one score per prepared fold. In the current pipeline, there are typically two configured dataset pairs:

- the no-additional-covariates condition
- the additional-covariates condition

Each pair contributes `num_folds` fold scores because the same priority function is evaluated repeatedly under fold-specific calibration/scoring splits for that condition. The evaluator stores those per-fold scores first, then stores the average of all fold scores under `mean` as the final entry. That final `mean` value is what upstream FunSearch uses for island ranking, best-program tracking, and default prompt-function ordering.

So for one candidate, the stored score signature is effectively:

- one scalar per fold, for example `no_covariates_fold_1`, `no_covariates_fold_2`, `with_covariates_fold_1`, `with_covariates_fold_2`
- `simplicity`
- `mean`

The fold-score vector is therefore stored in the program database semantically, but as separate named scalar entries rather than as one vector-valued field.

The evaluator also records an auxiliary simplicity score for the candidate function body. `simplicity` is defined as the negative AST node count of the candidate function, so values closer to zero are simpler and more negative values are more complex. That simplicity score is stored alongside the fold scores for diagnostics and prompt ordering, but it does not change the persisted ranking score. Fold scores are computed once, at evaluation time before first registration, and later island resets copy those stored scores instead of recomputing them.

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
2. Impute missing dosage and covariate values in that combined training data.
3. Write the full combined result to `oracle_train.pkl`.
4. Concatenate the raw testing pickle shards.
5. Impute missing dosage and covariate values in that combined testing data.
6. Randomly split the combined testing data into `evaluator.num_folds` disjoint folds using a deterministic seed derived from `experiment.random_seed` and the pair name.
7. For each fold `i`, write `scoring_i.pkl` as fold `i` and `calibration_i.pkl` as the concatenation of all remaining folds.

If the evaluator is configured with already prepared paths (`oracle_train_pickle`, `calibration_pickle`, `scoring_pickle`), those are treated as base paths for the persisted artifacts. The evaluator reuses the first materialized `oracle_train.pkl` plus numbered `calibration_i.pkl` and `scoring_i.pkl` files on later `prepare(...)` calls.

In normal runs, worker-local evaluators also call `prepare(...)`, but they reuse the same prepared artifacts if the manifest or pickles already exist. The evaluator only creates these artifacts the first time a pair is materialized in a run.

The first time a pair is actually materialized, the evaluator logs:

- the raw source pickle paths
- the output artifact paths
- the sample counts written to each artifact

### Missing-value imputation during preparation

Before the prepared pickles are written, the evaluator imputes missing feature values in the raw pandas DataFrames.

- dosage columns are mean-imputed
- additional covariate columns are mode-imputed

This happens in `prepare(...)`, not during candidate scoring, so the persisted `oracle_train.pkl`, `calibration_i.pkl`, and `scoring_i.pkl` artifacts already contain the cleaned values used by later evaluation steps. The covariate columns are also cast to float at this stage so pandas nullable `NA` values do not leak into downstream feature extraction.

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

For each prepared dataset pair, it does this for every fold-specific `calibration_i.pkl` and `scoring_i.pkl` pair:

1. Load `oracle_train.pkl`, `calibration_i.pkl`, and `scoring_i.pkl`.
2. Extract the dosage column names from the oracle-train data.
3. Build the strict priority-function contract objects.
4. For each scoring subject and each variant:
    - call the priority function to get a radius
    - call `effect_size_calculator(...)` with that radius
    - multiply the estimated effect size by the subject dosage to get an oracle contribution
5. Fit the ridge-penalized logistic calibration model on the calibration split.
6. Apply the fitted model to the scoring split.
7. Compute ROC AUC on the scoring labels.
8. Bootstrap the scoring set AUC and keep the median bootstrap AUC as the fold score.

The evaluator returns one fold score per prepared fold, plus:

- one stored scalar per fold, such as `pair_name_fold_1`, `pair_name_fold_2`, and so on.
- `simplicity`, an auxiliary score based on AST size of the priority function body.
- `mean`, the mean of all fold scores across all configured dataset pairs.

When that evaluated candidate is registered into an island, the program database stores exactly those fold scores, the auxiliary `simplicity`, and the final `mean`. It does not append any registration-time bonus or alternate combined score.

When a prompt contains two prior functions, the default ordering is still lower-mean first (`priority_v0`) and higher-mean second (`priority_v1`). There is one override: if the lower-mean function is simpler and has a higher score on at least one corresponding fold, it is treated as the more desirable mutation and is shown second as `priority_v1`.

## How prior mutations are presented to the LLM

When the sampler prepares an LLM prompt for an island, it shows one or two prior priority functions from that island and then asks for the next version.

With two prior functions, the default presentation is:

- the lower-mean function is shown first as `priority_v0`
- the higher-mean function is shown second as `priority_v1`

In that default case, the bridge text tells the LLM that `priority_v1` is a higher-scored improvement over `priority_v0` and suggests improving further in that direction or trying a novel direction.

There is one special override. If both of the following hold:

- the lower-mean function has a higher `simplicity` score, meaning it is simpler
- the lower-mean function beats the higher-mean function on at least one corresponding stored fold score

then the lower-mean function is treated as the more desirable mutation for prompting purposes. In that case:

- the higher-mean function is shown first as `priority_v0`
- the simpler lower-mean function is shown second as `priority_v1`

The bridge text also changes in that override case. Instead of calling `priority_v1` a higher-scored improvement, it explicitly tells the LLM that `priority_v1` is the preferred mutation because it is simpler and wins at least one fold score even though it does not have the higher mean score.

When the evaluator moves from calibration to scoring, it uses the combined `oracle_train.pkl + calibration_i.pkl` data to estimate marginal effect sizes for the scoring subjects. The scoring split itself is only used for the final personalized-risk scoring and AUC calculation.

## Repeated ancestry-distance work and proposed caching

This section documents a verified performance opportunity and a proposed implementation. The cache is not implemented yet.

For every target row, `_build_oracle_feature_matrix(...)` iterates over every dosage variant. For each variant it calls the candidate priority function and then `effect_size_calculator(...)`. The current effect-size path scans every reference record and recomputes the Euclidean ancestry distance to the target row. Consequently, the same target-to-reference distances are recomputed once per variant even though only the chosen radius and dosage column are variant-specific.

Candidate priority functions can add further repeated work. Priority helper functions that form ancestry intervals, cumulative balls, or novelty scores independently scan and sort the same target-to-reference distances. A reusable sorted order is therefore useful to both the final effect-size calculation and the supported priority helpers.

### Exact reference data by phase

Distance caches must preserve the evaluator's current ordered data composition:

1. Calibration for pair `p`, fold `i`: targets are `calibration_i.pkl`; references are `oracle_train.pkl`.
2. Scoring for pair `p`, fold `i`: targets are `scoring_i.pkl`; references are `oracle_train.pkl` followed by `calibration_i.pkl` in that exact order.

Preparation concatenates source pickle shards in config order using `pandas.concat(..., ignore_index=True)`. It creates folds from a deterministic seeded permutation. Each selected fold resets its DataFrame index but preserves the order of the selected row indices. Evaluation later reads those persisted artifacts without reordering them. Therefore row order is stable across candidate evaluations within a run, but it differs by pair and fold and must be part of cache identity.

The prepared artifacts are reused for every priority-function candidate in the experiment. Distances can therefore be computed once per `(pair, fold, phase)` and reused across all islands, cycles, candidates, and variants. A fold cache must never be reused for another fold solely because its arrays have the same shape.

### Implemented artifact layout

The pipeline stores cache artifacts under the prepared pair directory by default, adjacent to the corresponding `oracle_train.pkl`, `calibration_i.pkl`, and `scoring_i.pkl` files. You can override the root with `evaluator.distance_cache_dir`, or disable the feature with `evaluator.distance_cache_enabled = false`. For each fold, it persists files of the form:

```text
calibration_i.distances.npy
calibration_i.sorted_indices.npy
scoring_i.distances.npy
scoring_i.sorted_indices.npy
distance_cache_manifest.json
```

The distance arrays are two-dimensional because ancestry distance does not have a variant axis. `float64` preserves the current scalar-distance precision. Sorted indices use `int32` today. Partition workers memory-map these read-only arrays rather than pickle and copy them into every process.

The manifest records hashes of the ordered target and reference ancestry matrices, ancestry dimension, cache schema version, and the novelty baseline median needed by `ancestry_novelty_score`. Calibration references are distinct from scoring references because scoring uses the ordered concatenation `oracle_train.pkl + calibration_i.pkl`.

### CPU versus GPU

Vectorized NumPy is the current implementation. For the smoke and current production dimensions, ancestry has only 16 columns; CPU vectorization avoids Python record loops and is fast enough to justify the extra artifact write once per fold. It also avoids adding PyTorch as a required evaluator dependency.

GPU construction should be optional rather than automatic based only on installation. A CUDA path can use chunked `torch.cdist` when `torch.cuda.is_available()` is true, but it must write the same CPU `.npy` format and produce equivalent neighborhood membership. GPU transfer, initialization, shared-cluster policy, and contention from multiple pair workers can outweigh the small amount of arithmetic. Only a compute-node benchmark should decide whether CUDA is beneficial.

### Compatibility and validation requirements

The optimized path retains an uncached fallback. When `distance_cache_enabled` is `false`, the evaluator behaves as before. The test suite runs cached and uncached comparisons on smoke data and checks:

- fold construction and ordered row identities
- direct distances and sorted indices
- radius-boundary neighborhood membership
- calibration and scoring oracle feature matrices
- selected ridge penalty
- direct AUC, bootstrap median AUC, every fold score, and mean score

For larger verification on real compute nodes, use [verify_distance_cache_equivalence.py](/nfs/home/adas23/projects/AlphaEvolve/verify_distance_cache_equivalence.py). It compares cached and uncached fold scores for the Procedure 2 evaluator slice, and cached and uncached heldout AUC values for the standalone heldout evaluator.

## Where each score comes from

### Fold score

The fold score is the median ROC AUC from bootstrapping the held-out scoring fold. It measures how well the calibrated personalized risk scores align with the disease labels for that fold.

### Simplicity score

The simplicity score is computed as:

`simplicity = -(number of AST nodes in the candidate priority function)`

That means the direction is inverted relative to a raw size metric:

- higher `simplicity` values are simpler functions because they are less negative and closer to zero
- lower `simplicity` values are more complicated functions because they have more AST nodes and therefore become more negative

Examples from this run: a compact seed can have `simplicity = -93`, while a much larger evolved function can have `simplicity = -1144`. So `-93` is simpler than `-1144`.

This score is auxiliary only. It is stored in the program signature for diagnostics and diversity, but it is not the value upstream FunSearch uses to rank islands.

### Mean score

The mean score is the average of all fold scores across all configured conditions. With two conditions and `num_folds = N`, the evaluator averages `2N` fold scores. It is stored under `mean`, it remains the last stored value when the candidate is registered into an island, and it is the scalar upstream FunSearch uses for island ranking and resets.

The earlier fold-level values are still stored individually in the program database and are used by the prompt-order override described above.

## Data flow summary

The flow for one candidate is:

```mermaid
flowchart TD
    A[Prepared oracle-train plus fold-specific calibration/scoring pickles] --> B[Load candidate priority function]
    B --> C[Build strict oracle contract objects]
    C --> D[Call priority_function(training_data, ancestry_coordinate, target_variant)]
    D --> E[Call effect_size_calculator with returned radius]
    E --> F[Build oracle contribution G_ij * b_hat_j(a_i)]
    F --> G[Fit ridge logistic calibration]
    G --> H[Score held-out scoring set]
    H --> I[Bootstrap ROC AUC and take median]
    I --> J[Fold score]
    J --> K[Mean across all folds]
    J --> L[Store per-fold scores]
    K --> M[Store mean as final ranking score]
    K --> N[Use simplicity plus fold vector for prompt-order override only]
```

In short: `prepare(...)` creates the data layout, and `evaluate_candidate(...)` uses that layout to score the priority function under each configured pair and fold.