#!/bin/bash
# SLURM batch script for running the UV environment setup pipeline.
# Requires GPU access for building flash-attn, transformer-engine, fused
# kernels, and running the verification test suite.
#
# Usage:
#   sbatch setup_uv_env.sbatch
#
# Logs:
#   /projects/a5k/public/logs/neox-training/setup_uv_env_<JOB_ID>.out
#
# Typical runtime: ~60-90 minutes (flash-attn compilation is the bottleneck)

#SBATCH --job-name=setup-uv-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/projects/a5k/public/logs/neox-training/setup_uv_env_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  UV Environment Setup (SLURM)"
echo "======================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $SCRIPT_DIR"
echo ""

# Verify GPU is accessible before starting the long build
echo "Checking GPU access..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: No GPU detected. This job requires GPU access."
    exit 1
}
echo ""

# Run the setup script
bash setup_uv_env.sh
exit_code=$?

echo ""
echo "======================================"
echo "  Setup finished at $(date)"
echo "  Exit code: $exit_code"
echo "======================================"

exit $exit_code
