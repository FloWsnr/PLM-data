#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fenicsx-env

# Parse -n flag (number of MPI ranks, default 1)
NPROCS=1
if [[ "${1:-}" == "-n" ]]; then
    NPROCS="$2"
    shift 2
fi

if [[ "$NPROCS" -eq 1 ]]; then
    python -m plm_data "$@"
else
    mpirun -n "$NPROCS" python -m plm_data "$@"
fi
