#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment.
# Override with PLM_CONDA_ENV to use a non-default environment, e.g.:
#   PLM_CONDA_ENV=fenicsx-env-complex ./run.sh -n 4 run configs/basic/poisson/2d_default.yaml --output-dir ./output
eval "$(conda shell.bash hook)"
conda activate "${PLM_CONDA_ENV:-fenicsx-env}"

# Parse -n flag (number of MPI ranks, default 1)
NPROCS=1
if [[ "${1:-}" == "-n" ]]; then
    NPROCS="$2"
    shift 2
fi

if [[ "$NPROCS" -eq 1 ]]; then
    python -m plm_data "$@"
else
    # Bind OpenMP threads to avoid oversubscription with MPI ranks.
    # Each rank gets (total_cores / nprocs) threads.
    TOTAL_CORES=$(nproc)
    export OMP_NUM_THREADS=$(( TOTAL_CORES / NPROCS ))
    mpirun -n "$NPROCS" python -m plm_data "$@"
fi
