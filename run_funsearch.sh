#!/usr/bin/env bash
#
# Launch a FunSearch priority-function run with a reproducible environment.
#
# Usage:
#   ./run_funsearch.sh [CONFIG_PATH]
#
# CONFIG_PATH defaults to the Run1 example config. Output is written to a
# timestamped logger_*.log under prio_func_disc_runs/.
#
# Notes on the thread env vars below: the procedure2 hot path is GIL-bound
# Python with only tiny numpy calls, so these are a cheap guardrail against
# incidental BLAS thread oversubscription (many partition workers each spawning
# a BLAS thread pool), not a guaranteed speedup. The real concurrency lever is
# evaluator.calibration_partitions / scoring_partitions in the config: keep
# num_islands * num_pairs * partitions at or below your usable core count.

set -euo pipefail

# Resolve repo root as the directory containing this script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-Collaterals/Run1/funsearch_pipeline.example.json}"

VENV_ACTIVATE="/nfs/home/adas23/python_environments/OcologyAlphaEvolve/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

# Cap nested numeric-library threads to 1 so process-level partition workers do
# not each spin up a full BLAS/OpenMP thread pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export PYTHONPATH="$PWD"

LOG_DIR="prio_func_disc_runs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/logger_$(date +%Y%m%d_%H%M%S).log"

echo "Repo root:   $REPO_ROOT"
echo "Config:      $CONFIG_PATH"
echo "Log file:    $LOG_FILE"
echo "Allowed CPUs: $(python -c 'import os; print(sorted(os.sched_getaffinity(0)))' 2>/dev/null || echo unknown)"

python -m funsearch_pipeline --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1
