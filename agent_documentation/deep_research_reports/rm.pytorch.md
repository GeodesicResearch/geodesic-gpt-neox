# Considerations for Using PyTorch on the Isambard ARM HPC Cluster

*Technical Report — January 2026*

---

## Executive Summary

This report examines the key considerations for deploying and running PyTorch workloads on the Isambard ARM-based High Performance Computing (HPC) cluster. Isambard represents a pioneering series of ARM-based supercomputers operated by the Bristol Centre for Supercomputing (BriCS) and the GW4 Alliance. With the recent deployment of Isambard 3 featuring NVIDIA Grace CPU Superchips and Isambard-AI with Grace Hopper Superchips, the platform offers compelling opportunities for machine learning research while presenting unique architectural considerations that differ from traditional x86-based HPC systems.

---

## 1. Introduction to the Isambard Infrastructure

The Isambard project, led by the University of Bristol in collaboration with the GW4 Alliance (Universities of Bath, Bristol, Cardiff, and Exeter), has been at the forefront of ARM-based supercomputing since 2018. The current generation consists of two primary systems that offer distinct capabilities for PyTorch workloads.

### 1.1 Isambard 3 Specifications

Isambard 3 represents a TOP500-class supercomputer service optimised for both traditional HPC and AI workloads. The system features 384 NVIDIA Grace CPU Superchips providing over 55,000 ARM Neoverse V2 cores. Each node contains 144 cores operating at 3.1 GHz with 240 GB of LPDDR5X memory delivering approximately 1 TB/s of memory bandwidth per node. The system is interconnected via HPE Slingshot 11 at 200 Gbps and includes 2 PetaBytes of HPE ClusterStor Lustre storage.

### 1.2 Isambard-AI Specifications

Isambard-AI represents the UK's most powerful AI supercomputer, featuring 5,448 NVIDIA GH200 Grace Hopper Superchips. This system combines ARM-based Grace CPUs with Hopper-architecture GPUs optimised for power efficiency and large-scale AI training. The system achieves over 200 PetaFLOP/s using the Top500 Linpack benchmark and over 21 ExaFLOP/s of AI-optimised performance. The infrastructure includes nearly 25 petabytes of storage using the Cray ClusterStor E1000 optimised for AI workflows.

### 1.3 System Comparison

| Feature | Isambard 3 | Isambard-AI |
|---------|------------|-------------|
| Processor | NVIDIA Grace CPU Superchip | NVIDIA GH200 Grace Hopper |
| Architecture | ARM Neoverse V2 (aarch64) | ARM Grace + Hopper GPU |
| Total Nodes | 384 | 5,448 |
| Total Cores | 55,296 CPU cores | CPU + GPU accelerated |
| Primary Use Case | HPC, CPU-based ML | Large-scale AI training |

---

## 2. PyTorch Installation and Configuration

Installing PyTorch on Isambard requires attention to the ARM64 architecture and the availability of GPU resources. The platform supports multiple installation methods, each with distinct advantages.

### 2.1 Installation Methods

**pip Installation:** For Isambard-AI Phase 2, PyTorch can be installed via pip with CUDA 12.8 support. It is important to note that login nodes do not have GPUs, so installation should be performed on compute nodes using an interactive session (e.g., `srun -N 1 --gpus 4 --pty bash`). The command below will install the appropriate aarch64-compatible wheels with GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Conda Installation:** The conda-forge channel provides PyTorch packages with aarch64, CUDA, and NumPy support. Running the following ensures that CUDA is available during the installation process:

```bash
srun --gpus 1 --pty conda install conda-forge::pytorch
```

**Container-Based Deployment:** NVIDIA GPU Cloud (NGC) containers are recommended for production workloads. These containers come pre-configured with optimised PyTorch builds, flash-attention, and other AI libraries:

```bash
singularity pull pytorch_25.05-py3.sif docker://nvcr.io/nvidia/pytorch:25.05-py3
srun --gpus 1 singularity run --nv pytorch_25.05-py3.sif python3 -c "import torch; print(torch.cuda.is_available())"
```

### 2.2 Critical Installation Considerations

Several factors must be considered when installing PyTorch on Isambard:

- **Python version compatibility:** The official aarch64 builds require Python 3.9 or later
- **GPU detection issues:** Installing on login nodes without GPU access can cause some packages to fail; always install on compute nodes with GPU access
- **Release compatibility:** Consult the release compatibility matrix to ensure PyTorch, CUDA, and driver versions are aligned correctly

---

## 3. Performance Optimisation Strategies

Achieving optimal PyTorch performance on ARM architectures requires understanding the software stack and available optimisation libraries.

### 3.1 oneDNN and ARM Compute Library Integration

PyTorch supports the Compute Library for the ARM Architecture (ACL) through the oneDNN backend. This integration provides optimised GEMM kernels and neural network primitives specifically tuned for ARM Neoverse processors. The ACL implements kernels using specific instructions that execute faster on aarch64, including support for both FP32 and BFloat16 formats.

To enable these optimisations, no code changes are required at the application level. The following environment variables enable key optimisations:

```bash
export DNNL_DEFAULT_FPMATH_MODE=BF16  # Enable BFloat16 fast math mode
export LRU_CACHE_CAPACITY=1024        # Optimise primitive caching
```

These settings can provide up to 2x performance improvement compared to standard FP32 inference.

### 3.2 BFloat16 Support

The NVIDIA Grace processors in Isambard support the BFloat16 number format, which provides improved performance and smaller memory footprint while maintaining the same dynamic range as FP32. This support enables efficient deployment of models trained using BFloat16, FP32, or Automatic Mixed Precision (AMP). Standard FP32 models can utilise BFloat16 kernels via oneDNN FPmath mode without explicit model quantisation.

To verify BFloat16 support, check for the `bf16` flag in processor capabilities:

```bash
lscpu | grep -i flags
```

The presence of flags including `svebf16`, `i8mm`, and `bf16` indicates full BFloat16 acceleration support.

### 3.3 Memory and Threading Optimisation

The Grace CPU Superchip's high memory bandwidth (approximately 1 TB/s per node) provides advantages for memory-bound operations. Applications such as OpenFOAM and similar codes benefit significantly from this bandwidth for communication performance. Transparent huge pages (THP) can further improve performance by reducing Translation Lookaside Buffer (TLB) lookup overhead.

For multi-threaded workloads, proper OpenMP configuration is essential:

```bash
export OMP_NUM_THREADS=<appropriate_value>
export OMP_PLACES=cores  # Prevent thread migration
```

---

## 4. Distributed Training Considerations

Isambard's HPE Slingshot 11 interconnect enables efficient distributed training across multiple nodes. PyTorch's Distributed Data Parallel (DDP) framework is well-supported and recommended for multi-node training.

### 4.1 NCCL Configuration

For GPU-accelerated distributed training on Isambard-AI, NVIDIA Collective Communications Library (NCCL) provides optimised collective operations. The system documentation provides specific guidance on NCCL configuration for the Slingshot interconnect. Proper initialisation using `torch.distributed` with the NCCL backend is essential for achieving good scaling performance.

### 4.2 Multi-Node Scaling

Research conducted on ARM A64FX systems has shown that while communication overhead can impact scaling, careful attention to overlapping computation and communication can mitigate these effects. The `torchrun` launcher provides a convenient interface for launching distributed training jobs:

```bash
python -m torch.distributed.run \
    --nnodes=<number_of_nodes> \
    --nproc_per_node=<processes_per_node> \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=<node_name>:29400 \
    train_script.py
```

### 4.3 Container-Based Multi-Node Training

Both Singularity (Apptainer) and Podman-HPC are supported for containerised multi-node training on Isambard. The documentation provides specific guidance for configuring MPI and NCCL within containers, ensuring proper network fabric utilisation across nodes.

---

## 5. Software Ecosystem and Compatibility

### 5.1 Supported ML Frameworks

Beyond PyTorch, Isambard supports:

- **TensorFlow** — Available via NGC containers with GPU support
- **JAX** — Provides aarch64 and GPU compatibility out-of-the-box
- **Hugging Face** — Full ecosystem support for transformers and datasets
- **Flash-Attention** — Available through pip or pre-bundled in NGC containers
- **vLLM** — Supported through pip and E4S containers with aarch64-CUDA optimisations

### 5.2 Compiler Considerations

The choice of compiler can significantly impact PyTorch performance on ARM. Research comparing ARM, Fujitsu, and GCC compilers on A64FX systems has shown that the ARM compiler often delivers competitive or superior performance for training workloads. Available compilers on Isambard 3 include:

- Cray Programming Environment (CPE)
- GNU compilers
- Standard system tools

### 5.3 Spack Package Management

Spack is supported on Isambard for building and managing software stacks. This is particularly valuable for users who need custom PyTorch builds with specific compiler optimisations or backend configurations. The documentation provides guidance on setting up Spack environments tailored to the Grace architecture.

---

## 6. Challenges and Limitations

### 6.1 x86 Code Portability

Code developed for x86 systems may require modifications for ARM deployment. While PyTorch itself is architecture-agnostic at the Python API level, custom C++ extensions or CUDA kernels may need adaptation. Additionally, some third-party libraries may not have ARM-optimised builds, requiring fallback to generic implementations or custom compilation.

### 6.2 Conda Channel Availability

Not all Conda packages are available for aarch64 with CUDA support. Users have reported challenges obtaining `pytorch-cuda` packages for ARM64 platforms through standard Conda channels. NGC containers or pip-based installation from official PyTorch indices often provide more reliable GPU support.

### 6.3 Performance Debugging

Performance analysis tools may behave differently on ARM compared to x86. Profiling should account for the different micro-architecture characteristics of Neoverse V2 cores, including the 4x 128-bit SVE2 pipelines. The system supports Arm Allinea tools for performance analysis and debugging.

---

## 7. Best Practices and Recommendations

### 7.1 Environment Setup

- Always install PyTorch on compute nodes with GPU access to avoid compatibility issues
- Use NGC containers for production workloads to ensure optimised, pre-validated configurations
- Enable BFloat16 optimisations via environment variables for inference workloads
- Configure proper thread affinity and NUMA-aware memory allocation

### 7.2 Job Submission

Utilise Slurm for job scheduling with appropriate GPU allocation flags:

```bash
# Interactive development
srun -N 1 --gpus 4 --pty bash

# Batch job example
sbatch --nodes=2 --gpus-per-node=4 --time=04:00:00 train_job.sh
```

For production training, batch jobs should specify resource requirements precisely to optimise scheduler efficiency and job turnaround time.

### 7.3 Model Optimisation

- Consider quantisation strategies appropriate for the target deployment: BFloat16 for training, INT8 for inference where accuracy permits
- Leverage `torch.compile()` with the inductor backend for additional optimisation
- For memory-constrained scenarios, gradient checkpointing and mixed-precision training can reduce memory footprint while maintaining model quality

---

## 8. Conclusion

The Isambard ARM HPC cluster represents a compelling platform for PyTorch workloads, combining energy-efficient ARM architecture with powerful GPU acceleration. Isambard 3 offers excellent CPU-based performance for inference and smaller-scale training, while Isambard-AI provides massive GPU resources for large-scale AI training comparable to the world's leading AI research facilities.

Successful deployment requires attention to the ARM-specific software stack, including proper use of oneDNN with ARM Compute Library, appropriate BFloat16 configuration, and container-based deployment strategies. While some challenges exist around x86 code portability and software availability, the platform's performance-per-watt efficiency and modern interconnect make it an attractive option for sustainable AI research.

Researchers should engage with the BriCS documentation and helpdesk resources for the most current guidance, as the platform continues to evolve with new software releases and optimisations.

---

## References and Resources

- Bristol Centre for Supercomputing Documentation: https://docs.isambard.ac.uk
- BriCS Helpdesk: https://helpdesk.isambard.ac.uk
- PyTorch ARM Installation Guide: https://learn.arm.com/install-guides/pytorch/
- ARM Compute Library: https://github.com/ARM-software/ComputeLibrary
- NVIDIA NGC Containers: https://catalog.ngc.nvidia.com/