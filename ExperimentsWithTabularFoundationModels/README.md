# Tabular Foundation Model Experiments

This document describes the implemented experiment runner for testing whether a locked FunSearch-discovered borrowing scheme can improve exemplar selection for off-the-shelf tabular foundation models.

## Scientific Aim

The scientific object being discovered in this repository is not a disease classifier by itself. The core discovery target is an interpretable, patient- and variant-adaptive borrowing rule:

- input: a labeled reference cohort, one target patient's 16-dimensional ancestry coordinate, and one target variant
- output: a non-negative radius saying how far the evaluator should borrow reference subjects around that ancestry location for that variant

The grant narrative and evaluator documentation show the larger goal:

- discover a locked borrowing rule that can adapt local variant-effect estimation across ancestry drift
- validate that locked rule on heldout and study-independent cohorts
- release the locked rule as reusable executable code that no longer depends on an LLM after discovery

In the current Procedure 2 pipeline, that borrowing rule is used to estimate local per-variant effects and then a logistic calibration model turns those oracle-derived features into a prostate-cancer risk score.

The experiment defined here asks a different downstream question:

- can the same locked borrowing rule curate better in-context exemplars for off-the-shelf tabular foundation models than ancestry-agnostic or ancestry-matched random context selection?

## Papers, Packages, and Model Access

### Recommended installation into the current project environment

The current code should use these package versions in `/nfs/home/adas23/python_environments/OcologyAlphaEvolve`:

- `tabpfn==8.0.7`
- `tabicl==2.0.2`

Installation command:

```bash
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
python -m pip install tabpfn==8.0.7 tabicl==2.0.2
```

Notes for the reserved V100 node:

- the existing `torch 2.8.0+cu126` build in the environment is already CUDA-enabled
- no extra Flash Attention setup is required for the first pass
- TabICL's published docs describe Flash Attention 3 as primarily relevant to newer Hopper GPUs such as H100, so it should not be treated as a requirement for the V100 run

## Datasets in Scope

The first implementation should use the exact no-covariate prepared pickles named by the user.

### Dataset pair 1: MEC-style prepared data

- training data: `Data/FunsearchEvaluatorData/no_covariates_train.pkl`
- calibration data: `Data/FunsearchEvaluatorData/no_covariates_test.pkl`
- heldout data: `Data/FunsearchEvaluatorData/no_covariates_heldout.pkl`
- row lineage: `Data/FunsearchEvaluatorData/output_row_tracking.pkl`
- heldout evaluation ancestry: `JA`

Observed schema in this workspace:

- label column: `phenotype`
- feature columns: `110` dosage columns with prefix `dosage__`
- ancestry columns: `PC1` through `PC16`
- total predictor count for the no-covariate tables: `126`

Observed sample counts relevant to this experiment:

- train plus calibration candidate pool: `2277`
- same-ancestry `JA` candidate pool inside train plus calibration: `127`
- heldout `JA` targets to evaluate: `19`

### Dataset pair 2: OncoArray prepared data

- training data: `Data/FunsearchEvaluatorDataOncoArray/no_covariates_train.pkl`
- calibration data: `Data/FunsearchEvaluatorDataOncoArray/no_covariates_test.pkl`
- heldout data: `Data/FunsearchEvaluatorDataOncoArray/no_covariates_heldout.pkl`
- row lineage: `Data/FunsearchEvaluatorDataOncoArray/output_row_tracking.pkl`
- heldout evaluation ancestry: `ASIAN`

Observed schema in this workspace:

- label column: `phenotype`
- feature columns: `73` dosage columns with prefix `dosage__`
- ancestry columns: `PC1` through `PC16`
- total predictor count for the no-covariate tables: `89`

Observed sample counts relevant to this experiment:

- train plus calibration candidate pool: `28786`
- same-ancestry `ASIAN` candidate pool inside train plus calibration: `566`
- heldout `ASIAN` targets to evaluate: `63`

## Features Sent to the Tabular Foundation Models

For the first pass, the tabular foundation models should receive exactly the no-covariate prepared predictors already present in the chosen pickle.

### Predictor set

- all columns whose names start with `dosage__`
- all ancestry columns `PC1` through `PC16`

### Label

- `phenotype`

### Excluded columns

- no row-tracking fields go into the model
- no extra covariates go into the first pass because the requested data triplets are the `no_covariates_*` pickles

This means the feature matrix is:

- MEC: `110 dosage + 16 PCs = 126` predictors
- OncoArray: `73 dosage + 16 PCs = 89` predictors

## Evaluation Population

Only a filtered subset of the heldout pickle is scored for each dataset pair.

- MEC experiment: evaluate only heldout rows whose lineage ancestry group is `JA`
- OncoArray experiment: evaluate only heldout rows whose lineage ancestry group is `ASIAN`

The ancestry filter must be derived from `output_row_tracking.pkl`, not inferred from PC values directly.

All train and calibration rows remain eligible exemplar candidates unless a scheme explicitly restricts them.

## Models to Compare

Each model will be tested separately.

### TabPFN

- package: `tabpfn`
- first-pass estimator: `TabPFNClassifier`
- mode: off-the-shelf pretrained classifier
- intended use in this experiment: call `fit(X_context, y_context)` on the selected exemplar set and then `predict_proba(X_target)` for the heldout target rows

### TabICL

- package: `tabicl`
- first-pass estimator: `TabICLClassifier`
- mode: off-the-shelf pretrained classifier
- intended use in this experiment: call `fit(X_context, y_context)` on the selected exemplar set and then `predict_proba(X_target)` for the heldout target rows

Important interpretation:

- in both packages, the estimator wrapper is fitted to the current context set, but the pretrained model weights are not retrained from scratch on your cohort
- this is why the current experiment naturally combines train and calibration into one exemplar pool instead of repeating Procedure 2's separate logistic calibration stage

### Model-spec context cap used by the runner

The implemented runner does not leave context size uncapped.

The preferred way to run fixed experiments is to put the published cap directly into `max_context_rows` in the config. The runner still knows how to derive a fallback cap when `max_context_rows` is `null`, but that is only a safety default.

The published constraints used here are:

- `tabpfn`: use the documented TabPFN-3 row-by-feature operating points. The runner applies `1,000,000` rows when the feature count is at most `200`, `100,000` rows when the feature count is at most `2,000`, and `1,000` rows when the feature count is at most `20,000`.
- `tabicl`: use a conservative cap of `48,000` rows from the TabICL FAQ statement that the model was pre-trained on datasets with `300` to `48K` training samples. The runner also records warnings if the experiment lies outside TabICL's FAQ pre-training range of `2` to `100` columns, even though the broader README states that larger tables can still be handled.

For the current datasets, the fixed config values should be:

- `tabpfn`: `1000000`, because both current feature sets are at most `200` columns
- `tabicl`: `48000`, because that is the explicit upper end of the documented pre-training regime for released zero-shot checkpoints

For the current datasets, the reference-cohort sizes are below these row caps:

- MEC train plus calibration: `2277` rows
- OncoArray train plus calibration: `28786` rows

So the auto cap does not truncate the pooled reference cohorts for these runs. The target scheme remains distinct because it keeps only reference rows with positive priority support and truncates further only if that selected subset exceeds the model-spec cap.

## Context Selection Schemes

Each model and dataset pair should compare three exemplar-selection schemes.

### Baseline 1: Mixture Learning

- candidate pool: all rows in `training + calibration`
- context rule: use `max_context_rows` from config. If it is `null`, fall back to the runner's published-model cap. If eligible rows exceed the applied cap, take a reproducible random subsample.
- ancestry restriction: none

### Baseline 2: Independent Learning Scheme

- candidate pool: rows in `training + calibration` whose ancestry group matches the target heldout subject's ancestry group
- context rule: use `max_context_rows` from config. If it is `null`, fall back to the runner's published-model cap. If eligible rows exceed the applied cap, take a reproducible random subsample.
- ancestry restriction: yes, same broad ancestry group only

### Target scheme: Priority-function-curated context

Because the discovered priority function is variant-specific but TabPFN and TabICL need one row-level exemplar set, the implementation should use the following adapter.

For one heldout target subject with ancestry coordinate $a_i$ and a candidate exemplar row $r$:

1. Use the full `training + calibration` pool as the `training_data` contract for the locked priority function.
2. For every dosage feature / target variant $v$, evaluate the priority function and obtain a radius $R_v(a_i)$.
3. Compute the Euclidean ancestry distance $d(r, a_i)$ between the candidate exemplar row and the heldout target using `PC1` through `PC16`.
4. Define the candidate row support score as

$$
S(r \mid a_i) = \frac{1}{|V|} \sum_{v \in V} \mathbf{1}\{d(r, a_i) \le R_v(a_i)\}
$$

where $V$ is the set of dosage variants.
5. Rank candidate exemplar rows by descending $S(r \mid a_i)$.
6. Break ties by smaller ancestry distance.
7. Keep all candidate rows with strictly positive support score.
8. If that selected subset exceeds the applied context cap, keep the top rows up to the cap.
9. If no row has positive support score, fall back to the single closest ancestry neighbor.

This means rows that are not selected for any variant of the heldout target are excluded from the priority-curated context, even when `max_context_rows` is very large.

Interpretation:

- a row scores highly if it lies inside the discovered borrowing neighborhood for many variants of the target patient
- this converts the variant-wise borrowing scheme into a single exemplar set that can be consumed by a tabular foundation model
- when the model-spec cap does not bind, this scheme can still differ from Mixture Learning because it discards rows that are outside the discovered borrowing neighborhoods for all variants of the target patient

Operationally, the priority scheme is not "rank every reference row and then keep them all because the cap is large". It first filters to rows with $S(r \mid a_i) > 0$, then ranks that filtered set, then applies the cap only if needed.

This adapter is the default planned implementation because it is deterministic, interpretable, and uses the locked priority function without modifying its contract.

## Planned Procedure

For one experiment entry:

1. Load the train, calibration, and heldout pickles.
2. Concatenate train and calibration into one exemplar candidate pool.
3. Load `output_row_tracking.pkl` and recover ancestry groups for train, calibration, and heldout rows.
4. Filter the heldout table to the configured target ancestry group.
5. Build the predictor matrix from all `dosage__*` columns plus `PC1` through `PC16`.
6. For each comparison scheme, select a context set for each heldout subject or heldout ancestry batch.
7. Run the chosen off-the-shelf model on that context and predict heldout labels.
8. Compute ROC AUC on the filtered heldout subset.
9. Persist per-subject predictions, context audit information, and aggregate metrics.

Expected fitting granularity:

- Mixture Learning baseline can reuse one sampled context per experiment and therefore one fitted estimator per model and dataset pair.
- Independent Learning Scheme can reuse one fitted estimator per target ancestry group.
- Priority-function-curated context is target-subject-specific, so the first implementation should assume one context selection and one fitted estimator per heldout target subject.

Given the current heldout target counts, that is still tractable:

- MEC `JA`: `19` target-specific fits per model for the priority scheme
- OncoArray `ASIAN`: `63` target-specific fits per model for the priority scheme

## Outputs

`output_root_dir` is required.

For each invocation, the implementation should create one timestamped run directory inside `output_root_dir`, for example:

```text
ExperimentsWithTabularFoundationModels/output_results/
  20260912_021500_tabular_foundation_models/
```

All outputs for that invocation, including reusable caches, live inside that run directory.

The runner produces at least these files:

Before any model fitting starts, the runner prepares the `priority_radius_cache` for every experiment in the config. When two experiments in the same invocation have the same priority function and the same reference and heldout cohorts, the later experiment reuses the already-prepared cache from the earlier one inside that run instead of recomputing it.

- `run_config.used.json`: fully resolved config used for the run
- `run_events.log`: timestamped event log with start, completion, failure, and elapsed-seconds records for the run, each experiment, cache preparation, and each scheme execution
- `summary.json`: experiment-level metrics and metadata
- `scheme_metrics.csv`: one row per model and scheme with ROC AUC, target count, context-size summary, and timing summary
- `heldout_predictions.pkl`: one row per scored heldout subject per model, scheme, and dataset pair
- `selected_context_audit.pkl`: audit rows for every selected exemplar. For target-specific priority contexts, rows are recorded per heldout subject. For shared baseline contexts, the heldout subject id is null and the audit row states that the same context applies to all filtered heldout subjects.
- `priority_radius_cache/`: reusable per-heldout-subject, per-variant radius artifacts and metadata
- `figures/`: PNG figures matching the style of grant Figure 2 and Figure 3A for each tabular model
- `README_run_notes.md`: human-readable summary of what was run and which model checkpoints were used

### `heldout_predictions.pkl` contents

Each row should contain at least:

- dataset pair name
- model name
- scheme name
- heldout subject index within the filtered heldout slice
- heldout output pickle name and row number
- source lineage fields from `output_row_tracking.pkl`
- ancestry group
- actual disease label
- predicted disease probability
- predicted no-disease probability
- correct-label probability
- predicted label

For binary labels, `correct-label probability` means:

- disease probability if the true label is disease
- `1 - disease probability` if the true label is no disease

### `priority_radius_cache/` contents

For every heldout subject and every dosage variant, the implementation should compute and persist the radius returned by the locked priority function.

The cache is computed only for the heldout rows that survive the configured `heldout_target_ancestry_group` filter. It does not store radii for heldout ancestry groups that are outside that experiment's requested evaluation slice.

This cache must be stored inside the run directory so that later runs can inspect or reuse it directly. It should be written using the same contract objects and as much of the existing `PostProcesingData.evaluate_priofunction` and ancestry-distance-cache flow as possible.

The cache depends on the priority function and the dataset triplet, not on whether the downstream model is TabPFN or TabICL. A completed MEC cache can therefore be reused across `mec_ja_tabpfn` and `mec_ja_tabicl`, while OncoArray requires its own separate cache because the reference and heldout cohorts are different.

Recommended layout:

```text
<run_dir>/priority_radius_cache/
  manifest.json
  radius_matrix.npy
  radii_by_subject_variant.pkl
  reference_tracking_rows.pkl
  heldout_tracking_rows.pkl
  variant_metadata.json
  distance_cache/
```

`radii_by_subject_variant.pkl` is the easiest file to inspect directly. It is a pandas DataFrame in long form with one row per heldout target subject and dosage variant. Each row records at least:

- dataset pair name
- heldout subject index within the filtered heldout ancestry slice
- heldout output pickle name
- heldout output row number
- ancestry group
- variant index
- variant name
- dosage field name
- computed radius returned by the locked priority function
- reference cohort size used when the radius was computed

`reference_tracking_rows.pkl` is a pandas DataFrame that maps each pooled reference-cohort row, meaning training plus calibration, back to lineage metadata from `output_row_tracking.pkl`.

`heldout_tracking_rows.pkl` is the analogous pandas DataFrame for the filtered heldout target rows that actually participate in that experiment. It includes the local heldout-subject indexing used by `radii_by_subject_variant.pkl` and `radius_matrix.npy`.

`manifest.json` should include at least:

- full resolved path to the priority-function file used
- priority-function SHA256 so later runs can verify that the executable rule is identical
- function name
- full resolved paths to all training pickle inputs combined into the reference cohort
- full resolved paths to all calibration pickle inputs combined into the reference cohort
- full resolved path to the heldout pickle
- full resolved path to `output_row_tracking.pkl`
- feature specification used for variant enumeration and ancestry distance
- supported ancestry groups
- heldout target ancestry group
- random seed
- radius-cache schema version

Each stored radius record should include at least:

- dataset pair name
- heldout subject identifier
- variant name
- dosage field name
- computed radius
- reference cohort size used when the radius was computed

### `figures/` contents

For each tabular model, the implementation should write PNG figures modeled after the grant figures:

- an OncoArray ROC-AUC comparison figure in the style of Figure 2
- a heldout MEC ROC-AUC comparison figure in the style of Figure 3A

Recommended file names:

- `figures/oncoarray_roc_auc_tabpfn.png`
- `figures/oncoarray_roc_auc_tabicl.png`
- `figures/mec_roc_auc_tabpfn.png`
- `figures/mec_roc_auc_tabicl.png`

The intended style is:

- horizontal bar chart
- one bar per scheme
- x-axis is ROC-AUC on the requested heldout focal-ancestry subjects
- the Priority Function Curated Context bar is highlighted separately from the two baselines
- OncoArray figures use the configured bootstrap confidence interval, matching the Figure 2 style
- MEC figures omit confidence intervals when the config sets `plot_ci_level` to `null`, matching the Figure 3A style

`summary.json` should include at least:

- package versions for `tabpfn`, `tabicl`, `torch`, `numpy`, `pandas`, `scikit-learn`
- model checkpoint identifiers actually used
- dataset paths
- target ancestry group
- feature column list or compact feature specification
- random seed
- context-cap setting
- overall ROC AUC by scheme
- heldout target count by scheme

## Input Config

The runner accepts one JSON config file describing one or more experiments.

### Planned config semantics

- `training_pickle_path`, `calibrating_pickle_path`, and `heldout_pickle_path` may each be either one path or a JSON array of paths, matching the pattern already used elsewhere in this repository.
- `output_root_dir` is required and the implementation should create one timestamped run directory inside it. All results and caches for that invocation live under that generated run directory.
- `supported_ancestry_groups` defines the allowed ancestry labels recoverable from lineage names in `output_row_tracking.pkl`.
- `heldout_target_ancestry_group` is the only heldout group whose AUC is reported for that experiment.
- `model_name` must be either `tabpfn` or `tabicl` in the first implementation.
- `model_init_kwargs` passes estimator-specific constructor arguments directly to the underlying package.
- `priority_radius_cache_path` is an optional path to an existing `priority_radius_cache` directory or its `manifest.json`. If it matches the current priority function and dataset inputs, the runner links reusable artifacts into the new run directory instead of recomputing them. If the source cache is only partial but contains a valid distance cache, the runner links the distance cache and recomputes the radius files locally.
- within one invocation, the runner also auto-reuses a cache that it already prepared earlier in the same run when a later experiment has the same priority function and dataset triplet. This is why `mec_ja_tabpfn` and `mec_ja_tabicl` share one MEC cache, while the OncoArray experiments prepare or reuse a different cache.
- `max_context_rows` is the requested cap and should normally be set explicitly from published model guidance. If `null`, the runner applies its model-spec cap automatically. If it is a positive integer, the runner still clamps it to the model-spec cap.
- `predict_proba_batch_size` is an inference-throughput knob, not a context-window limit.
- `schemes` defaults to the three schemes defined in this document.

## Example Config With Inline Documentation

```json
{
  "_description": "Planned config for testing off-the-shelf tabular foundation models with baseline and priority-function-curated exemplar selection.",
  "_field_docs": {
    "prio_function_path": "Path to the locked FunSearch-discovered priority function. The function is used only for the priority-curated context scheme.",
    "function_name": "Function name inside prio_function_path. Usually priority.",
    "random_seed": "Global reproducibility seed used for random subsampling baselines and deterministic tie-breaking.",
    "output_root_dir": "Required root directory. The implementation creates one timestamped run directory inside it and writes all results there, including reusable radius caches.",
    "should_overwrite": "If false, the future implementation should refuse to overwrite a completed experiment directory.",
    "experiments": "List of experiment entries. Each entry binds one dataset triplet to one tabular foundation model."
  },
  "prio_function_path": "prio_func_disc_runs/oracle_priority_20260810_032011/cycle_0004/best_prio.py",
  "function_name": "priority",
  "random_seed": 7,
  "output_root_dir": "ExperimentsWithTabularFoundationModels/output_results",
  "should_overwrite": false,
  "experiments": [
    {
      "_description": "MEC-style no-covariate experiment, scoring only heldout JA subjects with TabPFN.",
      "_field_docs": {
        "name": "Unique experiment name. Also used as the result-directory name.",
        "model_name": "One of tabpfn or tabicl.",
        "training_pickle_path": "One .pkl path or a list of .pkl paths. All listed frames are concatenated into the exemplar training pool before train-plus-calibration pooling.",
        "calibrating_pickle_path": "One .pkl path or a list of .pkl paths. These rows are appended to training rows to create the full exemplar candidate pool.",
        "heldout_pickle_path": "One .pkl path or a list of .pkl paths. Only rows from heldout_target_ancestry_group are scored.",
        "output_row_tracking_path": "Path to output_row_tracking.pkl used to infer ancestry groups and carry row lineage into outputs.",
        "supported_ancestry_groups": "Allowed ancestry labels matched against lineage names in output_row_tracking.pkl.",
        "heldout_target_ancestry_group": "The only ancestry group evaluated for this experiment.",
        "label_column": "Binary response column to predict.",
        "dosage_prefix": "Prefix identifying dosage feature columns.",
        "ancestry_columns": "Columns used both as model features and for ancestry-distance calculations in the priority-curated scheme.",
        "additional_feature_columns": "Optional extra predictors to append after dosage and ancestry features. Empty in the first no-covariate pass.",
        "priority_radius_cache_path": "Optional path to an existing priority_radius_cache directory or manifest.json. If it matches the current priority function and datasets, the runner links reusable artifacts into the new run directory.",
        "max_context_rows": "Maximum number of exemplar rows passed to the tabular foundation model. Prefer an explicit published cap in the config; use null only to let the runner derive the same cap automatically.",
        "predict_proba_batch_size": "Optional target-batch size for prediction throughput only. This is not the context-window size.",
        "model_init_kwargs": "Package-specific constructor kwargs forwarded to TabPFNClassifier or TabICLClassifier.",
        "schemes": "Selection schemes to run. If omitted, run mixture_learning, independent_learning_scheme, and priority_function_curated_context."
      },
      "name": "mec_ja_tabpfn",
      "model_name": "tabpfn",
      "training_pickle_path": "Data/FunsearchEvaluatorData/no_covariates_train.pkl",
      "calibrating_pickle_path": "Data/FunsearchEvaluatorData/no_covariates_test.pkl",
      "heldout_pickle_path": "Data/FunsearchEvaluatorData/no_covariates_heldout.pkl",
      "output_row_tracking_path": "Data/FunsearchEvaluatorData/output_row_tracking.pkl",
      "supported_ancestry_groups": ["AA", "JA", "LA"],
      "heldout_target_ancestry_group": "JA",
      "label_column": "phenotype",
      "dosage_prefix": "dosage__",
      "ancestry_columns": [
        "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8",
        "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16"
      ],
      "additional_feature_columns": [],
      "priority_radius_cache_path": null,
      "max_context_rows": 1000000,
      "predict_proba_batch_size": null,
      "model_init_kwargs": {
        "device": "cuda",
        "model_path": "auto"
      },
      "schemes": [
        {
          "name": "mixture_learning",
          "type": "mixture_random"
        },
        {
          "name": "independent_learning_scheme",
          "type": "same_ancestry_random"
        },
        {
          "name": "priority_function_curated_context",
          "type": "priority_support_fraction",
          "tie_breaker": "smaller_ancestry_distance"
        }
      ]
    },
    {
      "name": "oncoarray_asian_tabicl",
      "model_name": "tabicl",
      "training_pickle_path": "Data/FunsearchEvaluatorDataOncoArray/no_covariates_train.pkl",
      "calibrating_pickle_path": "Data/FunsearchEvaluatorDataOncoArray/no_covariates_test.pkl",
      "heldout_pickle_path": "Data/FunsearchEvaluatorDataOncoArray/no_covariates_heldout.pkl",
      "output_row_tracking_path": "Data/FunsearchEvaluatorDataOncoArray/output_row_tracking.pkl",
      "supported_ancestry_groups": ["AFRICAN_ANCESTRY", "ASIAN", "EUROPEAN"],
      "heldout_target_ancestry_group": "ASIAN",
      "label_column": "phenotype",
      "dosage_prefix": "dosage__",
      "ancestry_columns": [
        "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8",
        "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16"
      ],
      "additional_feature_columns": [],
      "priority_radius_cache_path": null,
      "max_context_rows": 48000,
      "predict_proba_batch_size": null,
      "plot_ci_level": 80,
      "model_init_kwargs": {
        "device": "cuda",
        "allow_auto_download": true,
        "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt"
      }
    }
  ]
}
```

## Notes for the Future Implementation

- The implementation should reuse the repository's existing ancestry-group recovery logic from row lineage rather than inventing a new encoding.
- The implementation should reuse the existing priority-function loader and strict contract objects where possible.
- The implementation should cache ancestry-distance calculations because the priority-curated scheme repeatedly compares one heldout target against the same train-plus-calibration pool across many variants.
- The implementation should reuse existing code paths from `PostProcesingData.evaluate_priofunction` and `GenomicsHelpers.ancestry_distance_cache` where possible instead of rebuilding those mechanics independently.
- The implementation should record the exact package versions and resolved checkpoint identifiers for reproducibility.

## How to Run

From the repository root on the compute node:

```bash
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PYTHONPATH=$PWD python -m ExperimentsWithTabularFoundationModels \
  --config ExperimentsWithTabularFoundationModels/tabular_foundation_models.example.json
```

### One-time TabPFN license acceptance on a headless node

If an experiment includes `tabpfn`, the first local checkpoint download may pause for one-time license acceptance. On a cluster node without a browser, the `xdg-open` errors are expected and can be ignored. The package is falling back to manual authentication.

Use this flow once per account or environment:

1. Open `https://ux.priorlabs.ai/account` in a browser on your laptop or desktop and log in.
2. Accept the TabPFN license at `https://ux.priorlabs.ai/account/licenses`.
3. Copy the API key from the account page.
4. On the cluster, either paste that key directly into the terminal prompt when TabPFN asks for it, or export it before the run:

```bash
export TABPFN_TOKEN="<your-api-key>"
```

TabPFN caches the token after a successful check in `~/.cache/tabpfn/auth_token`, so later runs should not prompt again unless the token is missing or invalid.

The runner writes one timestamped directory under `ExperimentsWithTabularFoundationModels/output_results/`.

Inside that run directory, `run_events.log` records timestamped `started`, `completed`, and `failed` events with elapsed seconds for the run, each experiment, data loading, cache preparation, and each scheme execution.

The full example config currently defines four experiments:

- MEC `JA` with `tabpfn`
- MEC `JA` with `tabicl`
- OncoArray `ASIAN` with `tabpfn`
- OncoArray `ASIAN` with `tabicl`

The plots written by the run are:

- `mec_roc_auc_tabpfn.png`
- `mec_roc_auc_tabicl.png`
- `oncoarray_roc_auc_tabpfn.png`
- `oncoarray_roc_auc_tabicl.png`

Each plot contains one horizontal bar per scheme:

- Priority Function Curated Context
- Mixture Learning
- Independent Learning Scheme

The x-axis is ROC AUC on the requested heldout focal-ancestry subjects only. The bar labels show the exact AUC values. OncoArray plots include confidence-interval error bars because the example config requests an 80% bootstrap CI. MEC plots omit those error bars because the example config sets `plot_ci_level` to `null`, matching the Figure 3A presentation.

### Restarting With Cache Reuse

If a previous run already produced a matching `priority_radius_cache/`, point a later experiment at that cache with `priority_radius_cache_path`.

Example reuse path for the completed MEC cache from the interrupted run:

```json
"priority_radius_cache_path": "../ExperimentsWithTabularFoundationModels/output_results/20260912_025350_tabular_foundation_models/mec_ja_tabpfn/priority_radius_cache"
```

The runner verifies the priority function, function name, train paths, calibration paths, heldout paths, ancestry columns, and heldout target ancestry group before linking it into the new run directory. If the cache is complete, it symlinks the full cache. If only the distance-cache subtree is available, it symlinks that subtree and recomputes the radius files in the new run directory.