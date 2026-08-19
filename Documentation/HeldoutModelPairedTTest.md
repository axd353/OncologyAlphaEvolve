# Heldout Model Paired t-test

This document explains what the paired-t-test postprocessing script does, how it matches heldout subjects across models, what goes in the config file, and what the command produces.

The implementation entry point is [PostProcesingData/heldout_model_paired_ttest.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/heldout_model_paired_ttest.py).

## What the test does

The script compares `Scheme Discovered by LLM` against each alternate baseline model for one chosen ancestry group.

It reads the existing heldout prediction artifacts from:

- `precomputed_directory/heldout_model_predictions_by_model/*.pkl`, when that directory exists
- `precomputed_directory/heldout_model_predictions.pkl`, otherwise

For each heldout subject in the chosen ancestry group, the script computes a per-subject correctness score from the saved `risk_probability` column:

- if `label == 1`, use `risk_probability`
- if `label == 0`, use `1 - risk_probability`

So each subject gets the probability that the model assigned to the subject's correct label.

The script then compares `Scheme Discovered by LLM` against each alternate baseline with a one-sided paired t-test.

For each baseline, the hypotheses are:

- null hypothesis: the mean correct-label probability is equal between `Scheme Discovered by LLM` and the baseline
- alternative hypothesis: `Scheme Discovered by LLM` has a higher mean correct-label probability than the baseline

The p-value is computed from the paired differences:

- `correct_label_probability_llm - correct_label_probability_baseline`

The decision rule is:

- reject the null when `p_value <= p_value_threshold`

## How subject matching works

This is a paired test, so each baseline subject row must be matched to the corresponding `Scheme Discovered by LLM` subject row.

The script validates pairing using the identity columns already written by [PostProcesingData/evaluate_priofunction.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction.py):

- `heldout_subject_index`
- `heldout_output_pickle_name`
- `heldout_output_row_number`
- `source_pickle_name`
- `source_row_number`
- `ancestry_group`
- `label`

If these identity columns do not align exactly between the LLM-discovered scheme and a baseline, the script stops with an error instead of silently mismatching subjects.

## How to run it

From the repository root:

```bash
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
PYTHONPATH=$PWD python -m PostProcesingData.heldout_model_paired_ttest <config.json>
```

Example:

```bash
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
PYTHONPATH=$PWD python -m PostProcesingData.heldout_model_paired_ttest \
  PostProcesingData/heldout_model_paired_ttest.example.json
```

The example config file lives at [PostProcesingData/heldout_model_paired_ttest.example.json](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/heldout_model_paired_ttest.example.json).

## Config format

The config file is a JSON object.

Required fields:

- `precomputed_directory`: directory containing `heldout_model_predictions_by_model/` or `heldout_model_predictions.pkl`
- `target_ancestry_group`: ancestry label to test, matched case-insensitively after removing punctuation
- `p_value_threshold`: significance threshold for rejecting the equal-performance null, for example `0.05`

Optional field:

- `output_file_name`: markdown file name to write under `precomputed_directory`; must end with `.md`

Example config:

```json
{
  "_description": "Compare Scheme Discovered by LLM against each alternate baseline using a one-sided paired t-test on heldout correct-label probability within one ancestry group.",
  "_fields": {
    "precomputed_directory": "Directory that already contains heldout_model_predictions_by_model/ or heldout_model_predictions.pkl.",
    "target_ancestry_group": "Requested ancestry label, matched case-insensitively after removing punctuation.",
    "p_value_threshold": "Decision threshold for rejecting the equal-performance null, for example 0.05.",
    "output_file_name": "Optional. Markdown output file name written under precomputed_directory. Must end with .md and must be a file name only."
  },
  "precomputed_directory": "../prio_func_disc_runs/oracle_priority_20260810_032011/cycle_0004/distance_cache_oncoarray",
  "target_ancestry_group": "ASIAN",
  "p_value_threshold": 0.05,
  "output_file_name": "heldout_paired_ttest_asian.md"
}
```

## What the command produces

The script writes one new markdown file under `precomputed_directory`.

It does not modify or delete the existing model prediction pickles. If the requested output markdown file already exists, the script stops with `FileExistsError` instead of overwriting it.

The output markdown contains:

- the selected ancestry group
- the matched heldout subject count
- the definition of the per-subject score
- the null and alternative hypotheses
- one table row per alternate baseline

The result table has these columns:

- `Alternate baseline`
- `Matched subjects`
- `Mean diff (LLM - baseline)`
- `p-value`
- `Result`

The `Result` column reports whether the equal-performance null is rejected at the configured `p_value_threshold`.

## Notes

- The test uses the saved `risk_probability` values exactly as requested.
- For ridge-style baselines, `risk_probability` is a sigmoid transform of `risk_score`, so this test is on that bounded convenience value rather than on a separately calibrated absolute probability.