# FunSearch Run Post-Processing Outputs

The post-processing entry point is [PostProcesingData/funsearch_run_postprocess.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/funsearch_run_postprocess.py). Run it with a FunSearch experiment directory such as:

```bash
PYTHONPATH=$PWD python -m PostProcesingData.funsearch_run_postprocess prio_func_disc_runs/oracle_priority_20260716_050704
```

The script first looks for a sibling `logger_*` shell-capture log whose contents mention the run directory and moves that file into the run directory. It then writes four pickle files into the same run directory.

## 1. `sampler_completed_priority_function_counts.pkl`

Stored DataFrame fields:

- `cycle_index`: 1-based FunSearch cycle number.
- `island_id`: Island id from `island_XXX.log`.
- `completed_priority_function_count`: Number of sampler attempts in that cycle/island that were not logged as `rejected=empty_completion`.
- `configured_candidates_per_island_per_cycle`: The configured per-island attempt budget loaded from `config.used.json -> sampler.candidates_per_island_per_cycle`.

What it captures:

This is the closest quantity currently recoverable from sampler logs for “the LLM returned a completed priority function.” Operationally it counts non-empty completions. Empty completions caused by timeouts or API failures are excluded.

The extra configured-attempts field gives the denominator that the run was aiming for on every island in every cycle, even when retries inside a sample slot caused more backend calls than that configured budget.

Core single-log method:

- [count_completed_priority_functions_in_sampler_log](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/funsearch_run_postprocess.py)

## 2. `sampler_validated_priority_function_counts.pkl`

Stored DataFrame fields:

- `cycle_index`: 1-based FunSearch cycle number.
- `island_id`: Island id from `island_XXX.log`.
- `validated_priority_function_count`: Number of attempts that passed `validate_candidate_priority_function(...)` and therefore reached evaluator execution.

What it captures:

These are attempts logged either as `rejected=evaluation_failed` or as `registered=true`. Both cases indicate validation succeeded.

Core single-log method:

- [count_validation_passes_in_sampler_log](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/funsearch_run_postprocess.py)

## 3. `sampler_evaluation_completed_counts.pkl`

Stored DataFrame fields:

- `cycle_index`: 1-based FunSearch cycle number.
- `island_id`: Island id from `island_XXX.log`.
- `evaluation_completed_count`: Number of attempts where `evaluated_candidate is not None`.

What it captures:

This is exactly the number of attempts logged as `registered=true`.

Core single-log method:

- [count_evaluation_completions_in_sampler_log](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/funsearch_run_postprocess.py)

## 4. `sampler_island_best_improvement_counts.pkl`

Stored DataFrame fields:

- `cycle_index`: 1-based FunSearch cycle number.
- `island_best_improvement_count`: Sum across all islands of registered attempts logged with `better_than_present_best=True` in that cycle.

What it captures:

For each cycle, this counts how many accepted candidates improved the current best program within their own island at the moment they were registered.

Core single-log method:

- [count_island_best_improvements_in_sampler_log](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/funsearch_run_postprocess.py)

## Shared parser

All four quantities are derived from the same single-log parser:

- [extract_sampler_log_metrics](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/funsearch_run_postprocess.py)

That parser deduplicates the first successful registration for an attempt, because the sampler writes both a `full_priority_function_begin` line and a `full_priority_function_end` line for that first success.

## Plotting Sequential Conversion Rates

The plotter lives in [PostProcesingData/funsearch_funnel_conversion_plots.py](/nfs/home/adas23/projects/AlphaEvolve/PostProcesingData/funsearch_funnel_conversion_plots.py).

Run it after the four post-processing pickles above have been generated:

```bash
PYTHONPATH=$PWD python -m PostProcesingData.funsearch_funnel_conversion_plots \
	prio_func_disc_runs/oracle_priority_20260716_050704 \
	--rates completed validated evaluated improved \
	--save-formats png pdf
```

You can also run it directly as a file from the repository root:

```bash
python PostProcesingData/funsearch_funnel_conversion_plots.py \
	--rates all \
	--save-formats png pdf \
	prio_func_disc_runs/oracle_priority_20260716_050704
```

Or from inside `PostProcesingData/`:

```bash
python funsearch_funnel_conversion_plots.py \
	--rates all \
	--save-formats png pdf \
	prio_func_disc_runs/oracle_priority_20260716_050704
```

Do not use `python PostProcesingData.funsearch_funnel_conversion_plots.py`; dotted names are only for module execution with `python -m`.

### Plotter inputs

Required positional input:

- `run_dir`: FunSearch experiment directory containing the four sampler count pickles.

Required files inside `run_dir`:

- `sampler_completed_priority_function_counts.pkl`
- `sampler_validated_priority_function_counts.pkl`
- `sampler_evaluation_completed_counts.pkl`
- `sampler_island_best_improvement_counts.pkl`

Rate-selection input:

- `--rates all`: plot all adjacent conversion rates.
- `--rates completed`: plot `completed_priority_function_count / configured_candidate_slot_count`.
- `--rates validated`: plot `validated_priority_function_count / completed_priority_function_count`.
- `--rates evaluated`: plot `evaluation_completed_count / validated_priority_function_count`.
- `--rates improved`: plot `island_best_improvement_count / evaluation_completed_count`.
- Multiple names are allowed, for example `--rates completed validated`.

Save-control inputs:

- `--save-formats png pdf`: write both image formats.
- `--save-formats png`: write only PNG.
- `--save-formats pdf`: write only PDF.
- `--output-stem NAME`: use `NAME.png` and/or `NAME.pdf` instead of the default stem.
- `--no-save`: print the computed table without writing image files.
- `--title TEXT`: override the plot title.

### Plotter outputs

By default, files are saved inside `run_dir`:

- `sampler_funnel_conversion_rates.png`
- `sampler_funnel_conversion_rates.pdf`

The CLI also prints the per-cycle table used for the plot. That table contains the cycle-level summed counts and these rate columns:

- `completed_over_configured_rate`
- `validated_over_completed_rate`
- `evaluated_over_validated_rate`
- `improved_over_evaluated_rate`

Note: `completed_over_configured_rate` can be greater than 1.0 in real runs because the sampler may retry several LLM calls inside one configured candidate slot. The plotter intentionally leaves those values unclipped.