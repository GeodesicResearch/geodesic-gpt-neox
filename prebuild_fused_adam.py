import os
# Set environment variables
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_HOME'] = '/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6'
os.environ['CC'] = '/usr/bin/gcc-12'
os.environ['CXX'] = '/usr/bin/g++-12'

import sys
sys.path.insert(0, '/home/a5k/kyleobrien.a5k/geodesic-gpt-neox')

import torch
print(f"PyTorch version: {torch.__version__}")

# Monkey-patch _get_cuda_arch_flags to return correct values
from torch.utils import cpp_extension
original_get_cuda_arch_flags = cpp_extension._get_cuda_arch_flags

def patched_get_cuda_arch_flags(cflags=None):
    """Return CUDA arch flags for sm_90 (H100)."""
    print("Using patched _get_cuda_arch_flags for sm_90")
    return ['-gencode', 'arch=compute_90,code=sm_90']

cpp_extension._get_cuda_arch_flags = patched_get_cuda_arch_flags
print("Patched _get_cuda_arch_flags")

print("\nCreating test model...")
model = torch.nn.Linear(10, 10).cuda()

# Build fused_adam by instantiating it  
print("Instantiating FusedAdam (this triggers CUDA kernel build)...")
from deepspeed.ops.adam import FusedAdam
optimizer = FusedAdam(model.parameters(), lr=0.001)
print("Fused Adam instantiated successfully!")

# Run a dummy step
print("Running dummy optimization step...")
x = torch.randn(5, 10).cuda()
y = model(x)
loss = y.sum()
loss.backward()
optimizer.step()
print("Optimization step completed successfully!")
