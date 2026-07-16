from __future__ import annotations

import argparse
from pathlib import Path

from funsearch_pipeline.orchestration.runner import run_experiment

"""
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
cd /nfs/home/adas23/projects/AlphaEvolve
PYTHONPATH=$PWD python -m funsearch_pipeline --config Collaterals/RunSmoke/funsearch_pipeline.example.json

"""

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AlphaEvolve FunSearch priority-function pipeline.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the JSON configuration file.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    experiment_dir = run_experiment(args.config)
    print(experiment_dir)


if __name__ == "__main__":
    main()
