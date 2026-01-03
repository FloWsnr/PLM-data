#!/bin/bash
#SBATCH --job-name=pde-sim
#SBATCH --array=1-100%20           # Run 100 tasks, max 20 concurrent
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# PDE Simulation Array Job
# Usage: sbatch array_job.sh <config_dir> [log_file]
# Example: sbatch array_job.sh configs/batch/
# Example with log: sbatch array_job.sh configs/batch/ logs/array.log

set -e

# Get arguments
CONFIG_DIR=${1:-configs/batch}
LOG_FILE=${2:-}      # Optional: path to log file

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to project directory
cd /home/flwi01/coding/PLM-data

# Activate virtual environment
source .venv/bin/activate

# Get the config file for this array task
CONFIG_FILE=$(ls ${CONFIG_DIR}/*.yaml 2>/dev/null | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "==================================="
echo "SLURM Array Job: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG_FILE}"
if [ -n "$LOG_FILE" ]; then
    echo "Log file: ${LOG_FILE}"
fi
echo "Started: $(date)"
echo "==================================="

# Build command arguments
CMD_ARGS=(
    "${CONFIG_FILE}"
    --seed ${SLURM_ARRAY_TASK_ID}
)

# Add log file if specified
if [ -n "$LOG_FILE" ]; then
    CMD_ARGS+=(--log-file "$LOG_FILE")
fi

# Run simulation
python -m pde_sim run "${CMD_ARGS[@]}"

echo "==================================="
echo "Completed: $(date)"
echo "==================================="
