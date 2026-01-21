conda create -n neox python=3.12 -y
conda activate neox

# module load gcc-native/12.3
module load brics/nccl
module load cuda/12.6
module load cudatoolkit

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# TODO: Remove this in gpt-neox/megatron/fused_kernels/__init__.py
# os.environ["TORCH_CUDA_ARCH_LIST"] = ""

cd gpt-neox/
git checkout grad_ascent_support

# Install Torch
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"

# Install Main Reqs
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 pip install -r requirements/requirements.txt

# Install Weights & Biases
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 pip install -r requirements/requirements-wandb.txt
tp=$(pip show wandb | grep Location | awk '{print $2}')/wandb/errors/term.py
    sed -i 's/    return sys\.stderr\.isatty()/    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()/' $tp

# Install Transformer Engine
pip install nvidia-cudnn-cu12
export CPLUS_INCLUDE_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cudnn/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cudnn/lib:$LIBRARY_PATH
export CUDNN_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cudnn
export CPLUS_INCLUDE_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cublas/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cublas/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/a5k/kyleobrien.a5k/miniforge3/envs/neox/lib/python3.12/site-packages/nvidia/cublas/lib:$LIBRARY_PATH
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 pip install transformer-engine[pytorch]==1.12

# Build Fused Kernals
python -c "from megatron.fused_kernels import load; load()"

# Install Flash Attention
python -c "import torch; print(torch.cuda.is_available())"
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 pip install flash-attn==2.6.3

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')"

