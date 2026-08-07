# Broad Overview

This document ties together three things that are otherwise spread across the repository:

- how raw MEC data becomes evaluator-ready pickles under `Data/FunsearchEvaluatorData`
- what additional preprocessing the evaluator does before a priority function is called
- how to run a full priority-function discovery pipeline end to end

For implementation detail, also see [Documentation/HowEvaluatorWorks.md](Documentation/HowEvaluatorWorks.md), [Documentation/PYTHON_ENV_SETUP.md](Documentation/PYTHON_ENV_SETUP.md), and [Documentation/FunsearchRunPostProcessing.md](Documentation/FunsearchRunPostProcessing.md).

## End-to-end flow

At a high level, the current pipeline is:

1. Start from raw MEC pickle shards under `Data/RawData/`.
2. Run `Data/build_funsearch_evaluator_data.py` to create evaluator-ready datasets under `Data/FunsearchEvaluatorData/`.
3. Point `evaluator.dataset_pairs[*].training_pickles` and `testing_pickles` at those outputs in a FunSearch config such as `Collaterals/Run1/funsearch_pipeline.example.json`.
4. Run `python -m funsearch_pipeline --config ...` to launch the multi-cycle discovery run.
5. Optionally post-process the run directory to summarize funnel counts and plots.

## What `training_pickles` means during evaluation

The `training_pickles` entries in the FunSearch config are input sources for evaluator preparation. They are not passed directly and unchanged into the candidate priority function.

For the `procedure2` evaluator backend, the path is:

1. Concatenate all configured `training_pickles` for the dataset pair.
2. If the combined object is a pandas DataFrame, impute missing dosage columns by column mean.
3. If additional covariate columns are present, impute those by column mode and cast them to float.
4. Split the combined training rows into `oracle_train.pkl` and `calibration.pkl` using `evaluator.oracle_train_fraction`.
5. Build a strict `PriorityTrainingData` contract object from `oracle_train.pkl` when calibrating a candidate.
6. Build a strict `PriorityTrainingData` contract object from `oracle_train + calibration` when scoring the held-out testing set.

So the priority function sees a normalized contract object, not the original pickle container directly.

Just as important, the evaluator does not perform ancestry standardization itself. If your `training_pickles` already contain standardized `PC1` through `PC16`, then the priority function sees those standardized coordinates. If they do not, the evaluator will use the coordinates as provided.

The only built-in evaluator-side transformations are:

- concatenation of the listed pickle shards
- missing-value imputation for dosage and configured covariate columns
- deterministic oracle-train versus calibration split
- conversion into strict contract objects before the priority function is called

## How raw MEC data is transformed into `Data/FunsearchEvaluatorData`

The standalone builder lives at `Data/build_funsearch_evaluator_data.py`.

It reads the raw MEC shards from `Data/RawData/` for two conditions:

- `no_covariates`
- `with_covariates`

For each condition, it loads all six source shards:

- `train_AA`
- `train_JA`
- `train_LA`
- `test_AA`
- `test_JA`
- `test_LA`

### Standardization rule

The ancestry standardization is condition-wide, not file-by-file.

For one condition, the builder stacks `PC1` through `PC16` from all six shards and computes:

- $a^*$: the mean ancestry vector across all samples in that condition
- $r$: the smallest Euclidean radius such that at least 95% of samples lie within distance $r$ of $a^*$

It then rewrites every ancestry vector $a$ in every output pickle as:

$$
\frac{a - a^*}{r}
$$

The column names stay the same: `PC1` through `PC16`.

This means:

- the transform is shared across train, heldout, and test outputs within a condition
- the transform is also shared across ancestries AA, JA, and LA within a condition
- `no_covariates` and `with_covariates` each get their own independently fit transform

If you want to reuse this standardization on another dataset, the rule to preserve is:

1. decide the population of samples that defines the condition
2. pool all samples for that condition
3. compute one mean vector over `PC1` through `PC16`
4. compute one 95% coverage Euclidean radius from that same pooled set
5. apply the same $(a-a^*)/r$ transform to every split derived from that condition
6. record the fitted center and radius so future runs can reproduce the exact transform

### Dataset partitioning rule

After ancestry standardization, the builder writes three non-overlapping outputs per condition:

- `*_heldout.pkl`
- `*_test.pkl`
- `*_train.pkl`

The split logic is:

- heldout: 20% of `train_JA`, plus `M_ho` times that many rows from each of `train_AA` and `train_LA`
- test: all original test shards, plus an optional `P_add%` sample from the remaining `train_JA` rows and matched counts from the remaining `train_AA` and `train_LA` rows
- train: every remaining training row after heldout and optional test augmentation draws

For the `with_covariates` condition, the builder reuses the same heldout row indices drawn from the corresponding `no_covariates` training shards so both conditions refer to the same heldout subjects.

### Builder outputs

The builder writes:

- `Data/FunsearchEvaluatorData/no_covariates_train.pkl`
- `Data/FunsearchEvaluatorData/no_covariates_test.pkl`
- `Data/FunsearchEvaluatorData/no_covariates_heldout.pkl`
- `Data/FunsearchEvaluatorData/with_covariates_train.pkl`
- `Data/FunsearchEvaluatorData/with_covariates_test.pkl`
- `Data/FunsearchEvaluatorData/with_covariates_heldout.pkl`
- `Data/FunsearchEvaluatorData/transformations.txt`
- `Data/FunsearchEvaluatorData/build_funsearch_evaluator_data.log`

`transformations.txt` is the authoritative record of the fitted ancestry centers and radii.

## Commands

### 1. Set up the environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.funsearch_pipeline.txt
```

Set API keys before LLM-backed sampling runs:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 2. Build evaluator-ready datasets

From the repo root:

```bash
PYTHONPATH=$PWD python Data/build_funsearch_evaluator_data.py
```

Optional arguments:

```bash
PYTHONPATH=$PWD python Data/build_funsearch_evaluator_data.py --p-add 0 --m-ho 3 --random-seed 7
```

This populates `Data/FunsearchEvaluatorData/` and refreshes `transformations.txt` and `build_funsearch_evaluator_data.log`.

### 3. Confirm the FunSearch config points at the built pickles

In `Collaterals/Run1/funsearch_pipeline.example.json`, the current example already points `training_pickles` and `testing_pickles` at:

- `Data/FunsearchEvaluatorData/no_covariates_train.pkl`
- `Data/FunsearchEvaluatorData/no_covariates_test.pkl`
- `Data/FunsearchEvaluatorData/with_covariates_train.pkl`
- `Data/FunsearchEvaluatorData/with_covariates_test.pkl`

### 4. Run the full multi-cycle discovery pipeline

From the repo root:

```bash
PYTHONPATH=$PWD python -m funsearch_pipeline --config Collaterals/Run1/funsearch_pipeline.example.json > "prio_func_disc_runs/logger_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

For the smoke example:

```bash
PYTHONPATH=$PWD python -m funsearch_pipeline --config Collaterals/RunSmoke/funsearch_pipeline.example.json > "prio_func_disc_runs/logger_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

The runner creates a timestamped experiment directory under `prio_func_disc_runs/` and copies the resolved config there as `config.used.json`.

### 5. Post-process a completed run

```bash
PYTHONPATH=$PWD python -m PostProcesingData.funsearch_run_postprocess prio_func_disc_runs/oracle_priority_YYYYMMDD_HHMMSS
```

That step moves the matching shell log into the run directory and writes the funnel-count summaries used by the plotting utilities.

## Reuse on other datasets

If you want to apply the same ancestry standardization scheme to another dataset, keep these boundaries clear:

- the builder-stage standardization rule is external to the evaluator
- the evaluator will not infer or fit a new ancestry transform for you
- whatever `PC1` through `PC16` values appear in the configured pickles are the values the priority function and effect-size calculator will use

So for a new dataset, first produce standardized train and test pickles with the same schema, then point `training_pickles` and `testing_pickles` at those files in the FunSearch config.