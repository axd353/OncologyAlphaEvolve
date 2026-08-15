# Priority-Function Evaluation Against Baselines

This document explains how to evaluate one produced priority function on heldout data, how that main evaluation method works, which alternate baselines are currently available, how to run the code, and what the command produces.

The implementation entry point is [PostProcesingData/evaluate_priofunction.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction.py).

## What the main method is

The main method is the repository's Procedure 2 heldout evaluation flow applied to one concrete priority-function file.

At a high level, the script does four things:

1. Load a produced priority function from a `.py` file.
2. Build oracle-derived features for that priority function on a calibration split.
3. Fit the best calibration model over a configured set of penalties.
4. Score the heldout split and report heldout ROC AUC.

Operationally, the script treats its input datasets like this:

- `training_pickle_path`: oracle-training data. This may be one `.pkl` path or a JSON array of `.pkl` paths. If multiple paths are supplied, the DataFrames are concatenated to form the full training dataset.
- `calibrating_pickle_path`: calibration data used to fit the final linear calibration model. This may be one `.pkl` path or a JSON array of `.pkl` paths. If multiple paths are supplied, the DataFrames are concatenated to form the full calibration dataset.
- `heldout_pickle_path`: final evaluation data used only for the reported heldout ROC AUC. This may be one `.pkl` path or a JSON array of `.pkl` paths. If multiple paths are supplied, the DataFrames are concatenated to form the full heldout dataset.
- `output_row_tracking_path`: path to `output_row_tracking.pkl`, used to map heldout rows back to their source shards so the final report can compute per-ancestry heldout ROC AUC values.

The priority function itself is not directly compared on raw dosage values alone. Instead, the Procedure 2 evaluator builds oracle feature matrices from the priority function, calibrates those features on the calibration split, and then applies the fitted model to heldout subjects.

In code, this flow is driven by [PostProcesingData/evaluate_priofunction.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction.py) and the Procedure 2 helpers in [funsearch_pipeline/evaluation/procedure2.py](/nfs/home/adas23/projects/AlphaEvolve/funsearch_pipeline/evaluation/procedure2.py).

## Proposed ancestry-distance cache

This section describes the implemented ancestry-distance cache used by the heldout evaluator.

Oracle feature construction now caches ancestry distances for each target subject. For one target subject, the priority-function radius may differ by variant, but the Euclidean distance from that subject to each reference subject is independent of the variant. The same distance vector can therefore be shared by all variants without changing the evaluator's mathematical definition.

The postprocessing flow needs two separate reference/target combinations:

1. Calibration distances: rows from `calibrating_pickle_path` are targets and rows from `training_pickle_path` are references.
2. Heldout distances: rows from `heldout_pickle_path` are targets and the ordered concatenation `training_pickle_path + calibrating_pickle_path` is the reference data.

The cache artifacts are two-dimensional arrays, not variant tensors:

```text
calibration_distances.shape = (number of calibration rows, number of training rows)
heldout_distances.shape = (number of heldout rows, number of training + calibration rows)
```

An aligned matrix of sorted reference-row indices is also persisted. A priority helper can then reuse one sorted ancestry order for every variant of the target subject. Neighborhood membership remains variant-specific because each priority-function call can choose a different radius.

### Cache location and opt-out

Input pickle files do not need to be in the same directory. The heldout evaluator accepts these optional config fields:

- `distance_cache_enabled`: defaults to `true`; set it to `false` to force the old uncached behavior
- `distance_cache_dir`: optional path override; by default caches are written under `prio_function_path.parent / distance_cache`

Cache placement therefore does not depend on finding a common parent directory among the input files.

Each cache entry must be keyed by the ordered reference and target datasets, not just their file names. The cache manifest should record:

- schema and implementation version
- distance dtype and ancestry-column order
- ordered, resolved source paths
- source file sizes, modification times, and content fingerprints
- ordered row counts and row-identity fingerprints after concatenation and imputation
- reference composition, including the fact that heldout uses training rows followed by calibration rows

In the current implementation, the file names themselves are keyed by hashes of the ordered reference and target ancestry matrices plus the phase name. If the rows or their order change, a different cache file name is produced and a fresh cache is built. This prevents silently applying distances to the wrong subjects when files are moved, replaced, reordered, or regenerated.

### CPU and GPU construction

The current implementation uses vectorized NumPy on CPU. The current datasets have only 16 ancestry dimensions, so the arithmetic per distance is small and transferring data to a GPU may cost more than the calculation itself. The persistent cache also means distance construction happens once while feature evaluation is repeated many times.

An optional PyTorch CUDA builder may still be considered later, with all of these conditions:

- select CUDA only when `torch.cuda.is_available()` is true
- use chunked `torch.cdist` so device memory is bounded
- copy the final array to the same CPU cache format used by the NumPy path
- fall back to NumPy when CUDA is absent or initialization fails
- demonstrate a wall-time improvement on the compute node before becoming the default

Multiple partition workers must not independently construct the same cache on one GPU. Cache construction should happen once before partition workers start, with atomic publication of the completed artifact.

### Verification

The repository now includes cached-versus-uncached equivalence tests on smoke data, and a standalone verifier for larger compute-node checks. Verification compares:

- calibration and heldout distance matrices against direct scalar distance calculations
- neighborhood membership at radius boundaries
- oracle feature matrices
- selected calibration penalty
- per-fold direct and bootstrap ROC AUC values
- final fold-score ordering and mean score

The same candidate source, random seeds, folds, row order, bootstrap seeds, and floating-point dtype must be used on both paths. Fold scores should be exactly equal when the cached distances are computed with equivalent operations; otherwise the test must use a documented tight tolerance and explicitly check boundary memberships.

For larger runs, use [verify_distance_cache_equivalence.py](/nfs/home/adas23/projects/AlphaEvolve/verify_distance_cache_equivalence.py). It compares cached and uncached outputs for the Procedure 2 evaluator slice and for the standalone heldout evaluator.

## What counts as the comparison target

The reported number for the produced priority function is:

- heldout ROC AUC after Procedure 2 calibration and scoring.

If baselines are configured, the same heldout split is also evaluated with each configured baseline model, and their heldout ROC AUC values are printed alongside the priority-function result.

The final summary report also breaks down the priority-function heldout ROC AUC by ancestry group. Those ancestry groups are inferred from the heldout rows' source shard names through `output_row_tracking.pkl`, for example `train_AA.pkl`, `test_JA.pkl`, or `train_LA_add_covs.pkl`.

## Available alternate baselines

Baseline implementations are registered in [PostProcesingData/prio_func_eval_baselines/__init__.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/__init__.py).

The baseline catalog and method-specific documentation live here:

- [Documentation/PriorityFunctionBaselines.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityFunctionBaselines.md)
- [Documentation/PriorityBaselineMixtureLearning.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineMixtureLearning.md)
- [Documentation/PriorityBaselineIndependentLearningScheme.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineIndependentLearningScheme.md)
- [Documentation/PriorityBaselineTLGDES.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineTLGDES.md)
- [Documentation/PriorityBaselineTLPR.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineTLPR.md)

## How to run the evaluation

From the repository root:

```bash
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
PYTHONPATH=$PWD python -m PostProcesingData.evaluate_priofunction <config.json>
```

Example:

```bash
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
PYTHONPATH=$PWD python -m PostProcesingData.evaluate_priofunction \
  PostProcesingData/my_config.json
```

An example config file already exists at [PostProcesingData/evaluate_priofunction.oracle_priority_20260717_141059.cycle_0006.best_prio.json](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction.oracle_priority_20260717_141059.cycle_0006.best_prio.json).

## Config format

The config file is a JSON object.

Required field:

- `prio_function_path`: path to the `.py` file containing the produced priority function.

Common optional fields:

- `training_pickle_path`: defaults to `Data/FunsearchEvaluatorData/no_covariates_train.pkl`; may be a string path or a JSON array of string paths
- `calibrating_pickle_path`: defaults to `Data/FunsearchEvaluatorData/no_covariates_test.pkl`; may be a string path or a JSON array of string paths
- `heldout_pickle_path`: defaults to `Data/FunsearchEvaluatorData/no_covariates_heldout.pkl`; may be a string path or a JSON array of string paths
- `output_row_tracking_path`: defaults to `Data/FunsearchEvaluatorData/output_row_tracking.pkl`; used to recover ancestry groups for the heldout rows
- `report_file_name`: defaults to the priority-function file name with `.evaluation_report.json`, for example `best_prio.evaluation_report.json`; this file is written in the same directory as `prio_function_path`
- `supported_ancestry_groups`: defaults to `[
  "AA", "JA", "LA"
]`; these are the ancestry codes accepted when parsing `source_pickle_name` values from `output_row_tracking.pkl`
- `should_overwrite`: defaults to `true`; if the configured report file already exists, `true` forces a fresh overwrite and `false` reuses that file and computes only configured baselines that are still missing from it. If the configured report file does not exist, the full analysis runs fresh regardless.
- `function_name`: defaults to `priority`
- `calibration_penalties`: defaults to `[0.1, 1.0, 10.0]`
- `calibration_partitions`: positive integer worker count or `"auto"`
- `scoring_partitions`: positive integer worker count or `"auto"`
- `distance_cache_enabled`: defaults to `true`; set to `false` to disable persistent ancestry-distance caches
- `distance_cache_dir`: optional path override for the cache root directory
- `baselines`: list of baseline entries to run in addition to the produced priority function

Each baseline entry may be either:

- a string such as `"Mixture Learning"`
- an object such as `{ "name": "Mixture Learning", "alpha": 1.0 }`

Each baseline object may also include:

- `enabled`: if `false`, that baseline entry is ignored

Example config:

```json
{
  "_description": "Evaluate one produced priority function on concatenated train/calibration/heldout datasets and compare against optional baselines.",
  "_field_docs": {
    "prio_function_path": "Python file containing the produced priority function to evaluate.",
    "training_pickle_path": "One .pkl path or a JSON array of .pkl paths. All listed DataFrames are concatenated to form the oracle-training dataset.",
    "calibrating_pickle_path": "One .pkl path or a JSON array of .pkl paths. All listed DataFrames are concatenated to form the calibration dataset used to fit the final linear model.",
    "heldout_pickle_path": "One .pkl path or a JSON array of .pkl paths. All listed DataFrames are concatenated to form the heldout evaluation dataset.",
    "output_row_tracking_path": "Path to output_row_tracking.pkl. This links heldout rows back to source shard names so the report can compute per-ancestry heldout ROC AUC.",
    "report_file_name": "File name for the saved clean JSON report. It is written in the same directory as prio_function_path.",
    "supported_ancestry_groups": "Allowed ancestry codes when parsing source shard names from output_row_tracking.pkl.",
    "should_overwrite": "If the configured report file already exists: true overwrites it with a fresh run; false reuses it and computes only configured baselines that are missing from it. If the configured report file does not exist, the full analysis runs fresh.",
    "function_name": "Function name to load from prio_function_path. Usually priority.",
    "calibration_penalties": "Penalty values searched when fitting the final calibration model.",
    "calibration_partitions": "Worker count for calibration oracle feature construction. Use auto to detect visible CPUs.",
    "scoring_partitions": "Worker count for heldout scoring oracle feature construction. Use auto to detect visible CPUs.",
    "baselines": "Optional baseline models to run on the same heldout dataset."
  },
  "prio_function_path": "../prio_func_disc_runs/oracle_priority_20260717_141059/cycle_0006/best_prio.py",
  "training_pickle_path": [
    "../Data/FunsearchEvaluatorData/no_covariates_train.pkl"
  ],
  "calibrating_pickle_path": [
    "../Data/FunsearchEvaluatorData/no_covariates_test.pkl"
  ],
  "heldout_pickle_path": [
    "../Data/FunsearchEvaluatorData/no_covariates_heldout.pkl"
  ],
  "output_row_tracking_path": "../Data/FunsearchEvaluatorData/output_row_tracking.pkl",
  "report_file_name": "best_prio.evaluation_report.json",
  "supported_ancestry_groups": ["AA", "JA", "LA"],
  "should_overwrite": false,
  "function_name": "priority",
  "calibration_penalties": [0.1, 1.0, 10.0],
  "calibration_partitions": "auto",
  "scoring_partitions": "auto",
  "distance_cache_enabled": true,
  "baselines": [
    {
      "name": "Mixture Learning",
      "enabled": true,
      "alpha": 1.0
    },
    {
      "name": "Independent Learning Scheme",
      "enabled": true,
      "alpha": 1.0
    },
    {
      "name": "TL-GDES",
      "enabled": true,
      "max_iter": 100,
      "source_n_iter": 3000,
      "min_target_n": 20
    },
    {
      "name": "TL-PR",
      "enabled": true,
      "max_iter": 600,
      "source_n_iter": 3000,
      "n_lambdas": 30,
      "cv_folds": 2
    }
  ]
}
```

## What the command produces

The script produces both terminal output and a saved clean report file.

There are two output layers:

1. Progress logs with timestamps, for example dataset loading, feature-matrix construction, calibration fitting, and baseline execution.
2. A final plain-text summary report.
3. A clean JSON report file written in the same directory as the evaluated priority function.

The final summary report has this shape:

```text
prio_function_path=/absolute/path/to/best_prio.py
heldout_auc_roc=0.912345
heldout_subject_count[AA]=152
heldout_auc_roc[AA]=0.901234
heldout_subject_count[JA]=32
heldout_auc_roc[JA]=0.934567
heldout_subject_count[LA]=88
heldout_auc_roc[LA]=0.889012
baseline_auc_roc[Mixture Learning]=0.887654
baseline_auc_roc[Mixture Learning][AA]=0.876543
baseline_auc_roc[Mixture Learning][JA]=0.901234
baseline_auc_roc[Mixture Learning][LA]=0.854321
```

The saved JSON report path is determined by two config values:

Example:

- if the evaluated function is `cycle_0006/best_prio.py`
- and `report_file_name` is `best_prio.evaluation_report.json`
- then the clean report file is written as `cycle_0006/best_prio.evaluation_report.json`

If you set `report_file_name` to some other file name, that new file name is used in the same directory as the priority function.

- If the configured report file does not exist, the script runs the full evaluation fresh and writes a new report there.
- If `should_overwrite` is `true` and the configured report file already exists, rerunning overwrites that file with a fresh evaluation.
- If `should_overwrite` is `false` and the configured report file already exists, the script checks whether any configured baselines are missing from the existing report. Missing baselines are evaluated and appended into the report. If no configured baselines are missing, the existing report is reused.

If no baselines are configured, only the first two lines are printed.

The JSON report includes:

- the resolved config values used for the evaluation,
- the overall heldout ROC AUC,
- the per-ancestry heldout ROC AUC breakdown,
- the baseline results, including per-ancestry heldout ROC AUC when ancestry groups are available,
- the final plain-text summary as `summary_text`.

The command still does not write a pickle output, plot, or new artifact directory.

If you also want a shell-captured text file containing the progress logs plus stdout summary, you can still redirect stdout manually.

## Data and model expectations

The current baseline and main evaluation flow assume evaluator-ready pandas DataFrames.

For baseline-specific requirements, see:

- [Documentation/PriorityBaselineMixtureLearning.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineMixtureLearning.md)
- [Documentation/PriorityBaselineIndependentLearningScheme.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineIndependentLearningScheme.md)
- [Documentation/PriorityBaselineTLGDES.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineTLGDES.md)
- [Documentation/PriorityBaselineTLPR.md](/nfs/home/adas23/projects/AlphaEvolve/Documentation/PriorityBaselineTLPR.md)

The priority function is also validated before scoring. The script will fail early if:

- the config is malformed,
- the priority function cannot be loaded,
- the priority-function signature is invalid,
- required dosage columns are missing,
- or a baseline name is unsupported.

## How to add more baselines

To add another comparison method:

1. Implement a baseline evaluator under [PostProcesingData/prio_func_eval_baselines](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines).
2. Register it in [PostProcesingData/prio_func_eval_baselines/__init__.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/__init__.py).
3. Add a config entry under `baselines`.

Unsupported names currently raise a `ValueError` that lists the supported baseline names.