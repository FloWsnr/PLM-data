#!/bin/bash
#SBATCH --job-name=pde-sim-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-batch-%j.out
#SBATCH --error=logs/slurm-batch-%j.err

# Batch PDE Simulation Job
# Usage: sbatch batch_job.sh <config_dir> [log_file] [start_index] [pattern]
# Example: sbatch batch_job.sh configs/basic/damped_wave
# Example with log: sbatch batch_job.sh configs/basic/damped_wave logs/batch.log
# Example with start index: sbatch batch_job.sh configs/basic/damped_wave logs/batch.log 5
# Example with pattern: sbatch batch_job.sh configs/basic/damped_wave logs/batch.log 1 "**/*.yaml"
#
# The config_dir should contain one or more YAML config files, each specifying
# a complete simulation configuration including preset, parameters, etc.

set -e

# Get arguments
CONFIG_DIR=${1}
LOG_FILE=${2:-}         # Optional: path to log file
START_INDEX=${3:-1}     # Optional: start from config N (1-indexed)
PATTERN=${4:-"**/*.yaml"}  # Optional: glob pattern for config files

if [ -z "$CONFIG_DIR" ]; then
    echo "Error: Config directory is required"
    echo "Usage: sbatch batch_job.sh <config_dir> [log_file] [start_index] [pattern]"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Change to project directory
cd /scratch/zsa8rk/PLM-data

# Activate virtual environment
source .venv/bin/activate

# Ensure pde_sim package is installed
if ! python -c "import pde_sim" 2>/dev/null; then
    echo "Installing pde_sim package..."
    uv sync
fi

echo "==================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Config Directory: ${CONFIG_DIR}"
echo "Pattern: ${PATTERN}"
if [ -n "$LOG_FILE" ]; then
    echo "Log file: ${LOG_FILE}"
fi
echo "Starting from index: ${START_INDEX}"
echo "Started: $(date)"
echo "==================================="

# Check directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Build command arguments
CMD_ARGS=(
    "$CONFIG_DIR"
    --start-index "$START_INDEX"
    --pattern "$PATTERN"
)

# Add log file if specified
if [ -n "$LOG_FILE" ]; then
    CMD_ARGS+=(--log-file "$LOG_FILE")
fi

# Run the batch processor
python scripts/batch_runner.py "${CMD_ARGS[@]}"

echo "==================================="
echo "Batch completed: $(date)"
echo "==================================="
