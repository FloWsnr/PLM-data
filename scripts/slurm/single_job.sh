#!/bin/bash
#SBATCH --job-name=pde-sim-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Single PDE Simulation Job
# Usage: sbatch single_job.sh <config_file> [seed] [log_file]
# Example: sbatch single_job.sh configs/examples/gray_scott_spots.yaml 42
# Example with log: sbatch single_job.sh config.yaml 42 logs/simulation.log

set -e

# Get arguments
CONFIG_FILE=${1}
SEED=${2:-$RANDOM}
LOG_FILE=${3:-}      # Optional: path to log file

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file required"
    echo "Usage: sbatch single_job.sh <config_file> [seed] [log_file]"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Change to project directory
cd /home/flwi01/coding/PLM-data

# Activate virtual environment
source .venv/bin/activate

echo "==================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_FILE}"
echo "Seed: ${SEED}"
if [ -n "$LOG_FILE" ]; then
    echo "Log file: ${LOG_FILE}"
fi
echo "Started: $(date)"
echo "==================================="

# Build command arguments
CMD_ARGS=(
    "${CONFIG_FILE}"
    --seed ${SEED}
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
