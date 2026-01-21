#!/bin/bash
#SBATCH --job-name=prebuild_fused_adam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/a5k/public/logs/neox-training/prebuild_fused_adam_%j.out

cd /home/a5k/kyleobrien.a5k/geodesic-gpt-neox

source .venv/bin/activate
module load cuda/12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export TORCH_CUDA_ARCH_LIST='9.0'
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export NCCL_LIBRARY=/home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

echo "Pre-building fused_adam at $(date)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

LD_PRELOAD=$NCCL_LIBRARY python prebuild_fused_adam.py

echo ""
echo "Build completed at $(date)"
echo "Checking fused_adam cache:"
ls -la ~/.cache/torch_extensions/py312_cu126/fused_adam/ 2>/dev/null || echo "No fused_adam directory"
