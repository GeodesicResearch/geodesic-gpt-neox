# ==============================================================================
# NOTE: vLLM requires a separate environment with synth-vllm installed.
# See post-training/online_training.md for vLLM environment setup instructions.
# The vLLM environment setup is NOT part of the main UV environment and must
# be configured manually following the conda-based instructions in that doc.
# ==============================================================================

# Launch vllm (requires separate vllm conda environment - see online_training.md)
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run --no-capture-output -n vllm python -m vllm.entrypoints.openai.api_server --model=meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --from-remote-program --tensor-parallel-size=4 --enforce-eager --gpu-memory-utilization=0.2 --port 8000 --max-model-len=1024 --max-num-seqs=512 &

CUDA_VISIBLE_DEVICES=4,5,6,7 conda run --no-capture-output -n vllm python -m vllm.entrypoints.openai.api_server --model=meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --from-remote-program --tensor-parallel-size=4 --enforce-eager --gpu-memory-utilization=0.2 --port 8001 --max-model-len=1024 --max-num-seqs=512 &

# Launch training (uses UV venv)
# Activate the UV environment and set NCCL for single-node use
source /home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/bin/activate
export NCCL_LIBRARY=/home/a5k/kyleobrien.a5k/geodesic-gpt-neox/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$NCCL_LIBRARY"
python deepy.py train.py post-training/configs/llama3-8b-reinforce.yml
