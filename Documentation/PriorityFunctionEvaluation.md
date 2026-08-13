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

Operationally, the script treats its three input datasets like this:

- `training_pickle_path`: oracle-training data.
- `calibrating_pickle_path`: calibration data used to fit the final linear calibration model.
- `heldout_pickle_path`: final evaluation data used only for the reported heldout ROC AUC.

The priority function itself is not directly compared on raw dosage values alone. Instead, the Procedure 2 evaluator builds oracle feature matrices from the priority function, calibrates those features on the calibration split, and then applies the fitted model to heldout subjects.

In code, this flow is driven by [PostProcesingData/evaluate_priofunction.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction.py) and the Procedure 2 helpers in [funsearch_pipeline/evaluation/procedure2.py](/nfs/home/adas23/projects/AlphaEvolve/funsearch_pipeline/evaluation/procedure2.py).

## What counts as the comparison target

The reported number for the produced priority function is:

- heldout ROC AUC after Procedure 2 calibration and scoring.

If baselines are configured, the same heldout split is also evaluated with each configured baseline model, and their heldout ROC AUC values are printed alongside the priority-function result.

## Available alternate baselines

Baselines are registered in [PostProcesingData/prio_func_eval_baselines/__init__.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/__init__.py).

At present, the only supported alternate baseline is:

- `Mixture Learning`

Accepted baseline names for the same implementation are:

- `Mixture Learning`
- `mixture_learning`

The implementation lives in [PostProcesingData/prio_func_eval_baselines/mixture_learning.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/prio_func_eval_baselines/mixture_learning.py).

### What Mixture Learning does

The current Mixture Learning baseline:

- concatenates the training and calibration datasets,
- extracts all dosage columns plus ancestry columns,
- fits a ridge-regression model,
- predicts scores for the heldout dataset,
- reports heldout ROC AUC.

The currently supported option is:

- `alpha`: ridge penalty strength, default `1.0`

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
  PostProcesingData/evaluate_priofunction.oracle_priority_20260717_141059.cycle_0006.best_prio.json
```

An example config file already exists at [PostProcesingData/evaluate_priofunction.oracle_priority_20260717_141059.cycle_0006.best_prio.json](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction.oracle_priority_20260717_141059.cycle_0006.best_prio.json).

## Config format

The config file is a JSON object.

Required field:

- `prio_function_path`: path to the `.py` file containing the produced priority function.

Common optional fields:

- `training_pickle_path`: defaults to `Data/FunsearchEvaluatorData/no_covariates_train.pkl`
- `calibrating_pickle_path`: defaults to `Data/FunsearchEvaluatorData/no_covariates_test.pkl`
- `heldout_pickle_path`: defaults to `Data/FunsearchEvaluatorData/no_covariates_heldout.pkl`
- `function_name`: defaults to `priority`
- `calibration_penalties`: defaults to `[0.1, 1.0, 10.0]`
- `calibration_partitions`: positive integer worker count or `"auto"`
- `scoring_partitions`: positive integer worker count or `"auto"`
- `baselines`: list of baseline entries to run in addition to the produced priority function

Each baseline entry may be either:

- a string such as `"Mixture Learning"`
- an object such as `{ "name": "Mixture Learning", "alpha": 1.0 }`

Each baseline object may also include:

- `enabled`: if `false`, that baseline entry is ignored

Example config:

```json
{
  "prio_function_path": "../prio_func_disc_runs/oracle_priority_20260717_141059/cycle_0006/best_prio.py",
  "training_pickle_path": "../Data/FunsearchEvaluatorData/no_covariates_train.pkl",
  "calibrating_pickle_path": "../Data/FunsearchEvaluatorData/no_covariates_test.pkl",
  "heldout_pickle_path": "../Data/FunsearchEvaluatorData/no_covariates_heldout.pkl",
  "function_name": "priority",
  "calibration_penalties": [0.1, 1.0, 10.0],
  "calibration_partitions": "auto",
  "scoring_partitions": "auto",
  "baselines": [
    {
      "name": "Mixture Learning",
      "enabled": true,
      "alpha": 1.0
    }
  ]
}
```

## What the command produces

The script produces terminal output, not a saved report file.

There are two output layers:

1. Progress logs with timestamps, for example dataset loading, feature-matrix construction, calibration fitting, and baseline execution.
2. A final plain-text summary report.

The final summary report has this shape:

```text
prio_function_path=/absolute/path/to/best_prio.py
heldout_auc_roc=0.912345
baseline_auc_roc[Mixture Learning]=0.887654
```

If no baselines are configured, only the first two lines are printed.

The command does not currently write:

- a JSON report,
- a pickle output,
- a plot,
- or a new artifact directory.

If you want a saved report, redirect stdout to a file at the shell level.

Example:

```bash
PYTHONPATH=$PWD python -m PostProcesingData.evaluate_priofunction my_config.json \
  > my_priority_function_eval.txt
```

## Data and model expectations

The current baseline and main evaluation flow assume evaluator-ready pandas DataFrames.

For the current Mixture Learning baseline in particular, the data must contain:

- at least one dosage column,
- the default ancestry columns,
- the default label column.

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