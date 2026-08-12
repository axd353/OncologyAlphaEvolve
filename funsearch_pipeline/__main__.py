from __future__ import annotations

import argparse
from pathlib import Path

from funsearch_pipeline.orchestration.runner import resume_experiment
from funsearch_pipeline.orchestration.runner import run_experiment

"""
source /nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate
cd /nfs/home/adas23/projects/AlphaEvolve
PYTHONPATH=$PWD python -m funsearch_pipeline --config Collaterals/RunSmoke/funsearch_pipeline.example.json > "prio_func_disc_runs/logger_$(date +%Y%m%d_%H%M%S).log" 2>&1
PYTHONPATH=$PWD python -m funsearch_pipeline --config Collaterals/Run1/funsearch_pipeline.example.json > "prio_func_disc_runs/logger_$(date +%Y%m%d_%H%M%S).log" 2>&1
PYTHONPATH=$PWD python -m funsearch_pipeline --resume --resume-run-dir prio_func_disc_runs/oracle_priority_YYYYMMDD_HHMMSS > "prio_func_disc_runs/logger_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1
"""

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AlphaEvolve FunSearch priority-function pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from an existing experiment directory.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Existing experiment directory to resume when --resume is set.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.resume:
        if args.resume_run_dir is None:
            parser.error("--resume requires --resume-run-dir.")
        if args.config is not None:
            parser.error("--config cannot be combined with --resume.")
        experiment_dir = resume_experiment(args.resume_run_dir)
    else:
        if args.config is None:
            parser.error("--config is required unless --resume is set.")
        if args.resume_run_dir is not None:
            parser.error("--resume-run-dir requires --resume.")
        experiment_dir = run_experiment(args.config)

    print(experiment_dir)


if __name__ == "__main__":
    main()
