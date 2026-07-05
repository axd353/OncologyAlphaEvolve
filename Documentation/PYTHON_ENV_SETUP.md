# Python Environment Setup

This repo does not create an environment for you. The file [requirements.funsearch_pipeline.txt](requirements.funsearch_pipeline.txt) is a starting point for the pipeline you described:

- FunSearch implementation work, especially program-database experiments with islands
- effect-size estimation and local data handling
- OpenAI and Anthropic API clients
- optional Hugging Face and Transformers support

## Why pandas is included

The MEC training pickles at:

- `/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/MultiStagePythonCodeOutput/20260618_104142/stage4/train_AA.pkl`
- `/lustre/isaac24/scratch/adas23/dbGapData/MECProstateCancer/MultiStagePythonCodeOutput/20260618_104142/stage4/train_AA_add_covs.pkl`

appear to be pickled pandas DataFrames, so `pandas` must be installed before loading them.

## Create the environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.funsearch_pipeline.txt
```

If you want a Jupyter kernel tied to this environment:

```bash
python -m ipykernel install --user --name alphaevolve-funsearch --display-name "AlphaEvolve FunSearch"
```

## API keys

Set these in your shell before running the LLM-driven part of the pipeline:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

If you prefer a local `.env` file, keep it out of git and load it with `python-dotenv`.

## Sanity checks

After installation, these commands should work:

```bash
python -c "import numpy, pandas, sklearn, openai, anthropic"
python -c "from GenomicsHelpers.oracle_data_adapter import load_training_data"
python -c "from GenomicsHelpers.effect_size_calculator import effect_size_calculator"
```

## Adapter note

The current default adapter is aligned to the MEC training pickles:

- label: `phenotype`
- ancestry coordinates: `PC1` through `PC16`
- genotype columns: `dosage__...`
- optional covariates: any remaining non-dosage columns beyond the label and PCs

If a later dataset uses a different schema, keep the rest of the pipeline unchanged and edit only:

- [GenomicsHelpers/oracle_data_adapter.py](GenomicsHelpers/oracle_data_adapter.py)

The key signatures to preserve are:

- `load_training_data(data_source) -> Any`
- `iter_training_records(training_data) -> Iterable[Any]`
- `read_record_field(record, field_name) -> Any`
- `read_label(record) -> float`
- `read_ancestry_coordinate(record) -> numpy.ndarray`
- `read_variant_dosage(record, target_variant) -> float`
- `read_optional_covariates(record) -> numpy.ndarray | None`