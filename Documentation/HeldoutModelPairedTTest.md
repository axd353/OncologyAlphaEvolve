# Heldout Model Paired t-test

This document explains what the paired-t-test postprocessing script does, how it matches heldout subjects across models, what goes in the config file, and what the command produces.

The implementation entry point is [PostProcesingData/heldout_model_paired_ttest.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/heldout_model_paired_ttest.py).

## What the test does

The script compares `Scheme Discovered by LLM` against each alternate baseline model for either:

- one chosen ancestry group from one precomputed directory
- a list of chosen ancestry groups pooled across multiple precomputed directories

For each configured heldout slice, it reads the existing prediction artifacts from:

- `precomputed_directory/heldout_model_predictions_by_model/*.pkl`, when that directory exists
- `precomputed_directory/heldout_model_predictions.pkl`, otherwise

For each heldout subject in the chosen ancestry group for that slice, the script computes a per-subject correctness score from the saved `risk_probability` column:

- if `label == 1`, use `risk_probability`
- if `label == 0`, use `1 - risk_probability`

So each subject gets the probability that the model assigned to the subject's correct label.

The script then compares `Scheme Discovered by LLM` against each alternate baseline with a one-sided paired t-test.

For each baseline, the hypotheses are:

- null hypothesis: the mean correct-label probability difference is less than or equal to `effect_size`
- alternative hypothesis: `Scheme Discovered by LLM` mean correct-label probability minus baseline mean correct-label probability is greater than `effect_size`

The p-value is computed from the paired differences:

- `correct_label_probability_llm - correct_label_probability_baseline`

The t-test is centered at the configured effect-size margin, so the tested quantity is:

- `correct_label_probability_llm - correct_label_probability_baseline - effect_size`

The decision rule is:

- reject the null when `p_value <= p_value_threshold`

If `effect_size` is omitted from the config, it defaults to `0.0`, which recovers the old behavior.

If multiple precomputed directories are configured, the script performs that subject-level pairing independently within each directory, then concatenates the paired subject differences across all configured slices before running the t-test for each baseline.

## How subject matching works

This is a paired test, so each baseline subject row must be matched to the corresponding `Scheme Discovered by LLM` subject row inside the same precomputed directory and ancestry slice.

The script validates pairing using the identity columns already written by [PostProcesingData/evaluate_priofunction.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/evaluate_priofunction.py):

- `heldout_subject_index`
- `heldout_output_pickle_name`
- `heldout_output_row_number`
- `source_pickle_name`
- `source_row_number`
- `ancestry_group`
- `label`

If these identity columns do not align exactly between the LLM-discovered scheme and a baseline, the script stops with an error instead of silently mismatching subjects.

For multi-directory configs, the script also requires the set of alternate baselines to match across all configured precomputed directories.

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

Example config files:

- [PostProcesingData/heldout_model_paired_ttest.example.json](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/heldout_model_paired_ttest.example.json) for the original single-directory format
- [PostProcesingData/heldout_model_paired_ttest.multi_directory.example.json](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/heldout_model_paired_ttest.multi_directory.example.json) for the pooled multi-directory format

## Config format

The script accepts two config variants.

### Single-directory config

Required fields:

- `precomputed_directory`: directory containing `heldout_model_predictions_by_model/` or `heldout_model_predictions.pkl`
- `target_ancestry_group`: ancestry label to test, matched case-insensitively after removing punctuation
- `p_value_threshold`: significance threshold for rejecting the equal-performance null, for example `0.05`

Optional field:

- `effect_size`: minimum improvement margin in the alternative hypothesis; defaults to `0.0`
- `output_file_name`: markdown output path; if it is only a file name, it is written under `precomputed_directory`. A full path is also accepted.

Example config:

```json
{
  "_description": "Compare Scheme Discovered by LLM against each alternate baseline using a one-sided paired t-test on heldout correct-label probability within one ancestry group.",
  "_fields": {
    "precomputed_directory": "Directory that already contains heldout_model_predictions_by_model/ or heldout_model_predictions.pkl.",
    "target_ancestry_group": "Requested ancestry label, matched case-insensitively after removing punctuation.",
    "p_value_threshold": "Decision threshold for rejecting the equal-performance null, for example 0.05.",
    "effect_size": "Optional. Minimum improvement margin for the alternative hypothesis. Defaults to 0.0.",
    "output_file_name": "Optional. Markdown output file name. For single-directory configs, this may be just a file name written under precomputed_directory, or a full path."
  },
  "precomputed_directory": "../prio_func_disc_runs/oracle_priority_20260810_032011/cycle_0004/distance_cache_MEC_HO",
  "target_ancestry_group": "JA",
  "p_value_threshold": 0.05,
  "effect_size": 0.0,
  "output_file_name": "heldout_paired_ttest_ja.md"
}
```

### Multi-directory config

Required fields:

- `precomputed_directories`: JSON array of precomputed directories
- `target_ancestry_group`: JSON array of ancestry labels with the same length as `precomputed_directories`
- `p_value_threshold`: significance threshold for rejecting the equal-performance null, for example `0.05`
- `output_file_name`: full absolute path to the markdown file to create

Optional field:

- `effect_size`: minimum improvement margin in the alternative hypothesis; defaults to `0.0`

Each `(precomputed_directories[i], target_ancestry_group[i])` pair defines one heldout slice. The script validates subject pairing within each slice, computes the per-subject correct-label probabilities there, and then pools all paired subject differences across the slices before running the one-sided paired t-test for each alternate baseline.

Example config:

```json
{
  "_description": "Compare Scheme Discovered by LLM against each alternate baseline using a one-sided paired t-test on heldout correct-label probability pooled across multiple heldout slices.",
  "_fields": {
    "precomputed_directories": "JSON array of directories. Each directory must already contain heldout_model_predictions_by_model/ or heldout_model_predictions.pkl.",
    "target_ancestry_group": "JSON array of ancestry labels with the same length as precomputed_directories. Each entry selects the ancestry slice to use in the corresponding directory.",
    "p_value_threshold": "Decision threshold for rejecting the equal-performance null, for example 0.05.",
    "effect_size": "Optional. Minimum improvement margin for the alternative hypothesis. Defaults to 0.0.",
    "output_file_name": "Required for multi-directory configs. Must be a full absolute path to the markdown file to create."
  },
  "precomputed_directories": [
    "../prio_func_disc_runs/oracle_priority_20260810_032011/cycle_0004/distance_cache_MEC_HO",
    "../prio_func_disc_runs/oracle_priority_20260810_032011/cycle_0004/distance_cache_oncoarray"
  ],
  "target_ancestry_group": [
    "JA",
    "ASIAN"
  ],
  "p_value_threshold": 0.05,
  "effect_size": 0.0,
  "output_file_name": "/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/heldout_model_paired_ttest.multi_directory.example.output.md"
}
```

## What the command produces

For single-directory configs, the script writes one new markdown file either under `precomputed_directory` or at the explicitly supplied full path.

For multi-directory configs, the script writes one new markdown file at the full path given by `output_file_name`.

It does not modify or delete the existing model prediction pickles. If the requested output markdown file already exists, the script stops with `FileExistsError` instead of overwriting it.

The output markdown contains:

- the matched heldout subject count
- the list of configured input slices when multiple directories are used
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