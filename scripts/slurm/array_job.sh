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
# Usage: sbatch array_job.sh <config_dir>
# Example: sbatch array_job.sh configs/batch/

set -e

# Get config directory from argument or use default
CONFIG_DIR=${1:-configs/batch}

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
echo "Started: $(date)"
echo "==================================="

# Run simulation with unique seed based on array task ID
python -m pde_sim run "${CONFIG_FILE}" --seed ${SLURM_ARRAY_TASK_ID}

echo "==================================="
echo "Completed: $(date)"
echo "==================================="
