#!/bin/bash
#SBATCH --job-name=neox-training
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
# Log output: override at submission with --output=/your/log/dir/neox-training-%j.out
# Default writes to the submit directory (repo root)
#SBATCH --output=logs/neox-training-%j.out

# --- Isambard node-limit guard (do not remove) ---
# Add isambard_sbatch to PATH for compute nodes (SLURM batch jobs don't source .bashrc)
export PATH="$HOME/isambard_sbatch/bin:$PATH"
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found. Install: ~/isambard_sbatch/install.sh" >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

# Derive repo root from the submit directory
REPO_DIR="$SLURM_SUBMIT_DIR"
if [ ! -f "$REPO_DIR/deepy.py" ]; then
    echo "FATAL: deepy.py not found in SLURM_SUBMIT_DIR ($REPO_DIR)."
    echo "Please submit this job from the geodesic-gpt-neox repo root."
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

# Log cluster status at job start
echo "===== Cluster Status at Job Start ====="
bash "$REPO_DIR/tools/cluster/cluster_status.sh" 2>/dev/null || echo "(cluster_status.sh not available)"
echo "========================================"

echo "Debug: Current node: $(hostname)"
echo "Debug: Current user: $(whoami)"
echo "Debug: PWD: $(pwd)"
echo "Debug: REPO_DIR: $REPO_DIR"

# Activate uv virtual environment
source "$REPO_DIR/.venv/bin/activate"

module purge
module load PrgEnv-cray
module load cuda/12.6
# Load OFI plugin for NCCL (provides Slingshot support)
module load brics/aws-ofi-nccl/1.8.1

# Use the bundled NCCL from venv (2.27.5) - required for PyTorch 2.10.0 compatibility
# The system NCCL (2.26.6) is missing ncclCommShrink function required by PyTorch
export NCCL_LIBRARY="$REPO_DIR/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# Compilers and CUDA arch
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# NCCL / OFI (AWS Libfabric) settings for Slingshot (CXI)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET,COLL,GRAPH
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"   # must match plugin name
export FI_PROVIDER=cxi            # use the Slingshot CXI provider
export NCCL_SOCKET_IFNAME=hsn     # keep TCP fallback on HSN NICs
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1

# Uncomment when debugging NCCL issues
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Use offline mode to avoid HuggingFace API rate limits with many parallel ranks
export HF_HUB_OFFLINE=1

# --- Log PyTorch / CUDA info to the job output ---
echo "===== PyTorch & CUDA info ====="
python - <<'PY'
import os, torch
print(f"PyTorch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"TORCH_CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST')}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Visible GPUs: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU[{i}]: {name}  (SM {cap[0]}.{cap[1]})")
PY
echo "================================"
# ----------------------------------

# Generate hostfile (only for multi-node; single-node uses local launcher)
if [ "$SLURM_NNODES" -gt 1 ]; then
    HOSTFILE="$REPO_DIR/hostfile_$SLURM_JOB_ID"
    scontrol show hostname "$SLURM_NODELIST" | while read -r host; do
        echo "$host slots=4"
    done > "$HOSTFILE"
    echo "Generated hostfile at: $HOSTFILE"
    cat "$HOSTFILE"
    export DLTS_HOSTFILE="$HOSTFILE"
else
    echo "Single-node job: skipping hostfile generation (using local launcher)"
    unset DLTS_HOSTFILE
fi

NEOX_CONFIG=$1
CONFIG_BASENAME=$(basename "$NEOX_CONFIG" .yml)

# Job chaining configuration
MAX_JOB_CHAINS=${MAX_JOB_CHAINS:-30}  # Maximum number of 24-hour jobs to chain (default: 30 = 30 days)
CURRENT_CHAIN=${SLURM_JOB_CHAIN_COUNT:-0}

echo "===== Job Chain Info ====="
echo "Current chain iteration: $CURRENT_CHAIN / $MAX_JOB_CHAINS"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $NEOX_CONFIG"
echo "=========================="

export TMPDIR=/projects/a5k/public/tmp_$USER
mkdir -p "$TMPDIR"

# Store W&B logs on shared filesystem so they persist and are accessible
export WANDB_DIR=/projects/a5k/public/logs_$USER/wandb
mkdir -p "$WANDB_DIR"

# Launch GPT-NeoX training
cd "$REPO_DIR"
python deepy.py train.py "$NEOX_CONFIG"

# Clean up hostfile
[ -f "$REPO_DIR/hostfile_$SLURM_JOB_ID" ] && rm "$REPO_DIR/hostfile_$SLURM_JOB_ID"

echo "===== Job Completed ====="
