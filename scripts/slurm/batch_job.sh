#!/bin/bash
#SBATCH --job-name=pde-sim-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-batch-%j.out
#SBATCH --error=logs/slurm-batch-%j.err

# Batch PDE Simulation Job
# Usage: sbatch batch_job.sh <base_config.yaml> <parameters.csv>
# Example: sbatch batch_job.sh configs/examples/physics/gray_scott.yaml configs/parameters/physics/gray-scott.csv
#
# The parameters.csv file should have:
#   - A header row with parameter names
#   - One simulation configuration per row
#   - Special columns: BC_x, BC_y, init, solver, dt, t_end, notes (all optional)
#   - PDE-specific parameters (e.g., F, k, Du, Dv for Gray-Scott)
#
# Example CSV format:
#   F,k,Du,Dv,BC_x,BC_y,init,solver,dt,t_end,notes
#   0.014,0.054,2e-05,1e-05,periodic,periodic,gaussian-blobs,implicit,1,2500,gliders

set -e

# Get arguments
BASE_CONFIG=${1}
PARAMS_CSV=${2}
START_ROW=${3:-1}  # Optional: start from row N (1-indexed, excluding header)

if [ -z "$BASE_CONFIG" ] || [ -z "$PARAMS_CSV" ]; then
    echo "Error: Both base config and parameters CSV are required"
    echo "Usage: sbatch batch_job.sh <base_config.yaml> <parameters.csv> [start_row]"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Change to project directory
cd /home/flwi01/coding/PLM-data

# Activate virtual environment
source .venv/bin/activate

echo "==================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Base Config: ${BASE_CONFIG}"
echo "Parameters CSV: ${PARAMS_CSV}"
echo "Starting from row: ${START_ROW}"
echo "Started: $(date)"
echo "==================================="

# Check files exist
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config file not found: $BASE_CONFIG"
    exit 1
fi

if [ ! -f "$PARAMS_CSV" ]; then
    echo "Error: Parameters CSV file not found: $PARAMS_CSV"
    exit 1
fi

# Create temp directory for generated configs
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Run the batch processor
python scripts/batch_runner.py \
    --base-config "$BASE_CONFIG" \
    --params-csv "$PARAMS_CSV" \
    --start-row "$START_ROW" \
    --temp-dir "$TEMP_DIR"

echo "==================================="
echo "Batch completed: $(date)"
echo "==================================="
