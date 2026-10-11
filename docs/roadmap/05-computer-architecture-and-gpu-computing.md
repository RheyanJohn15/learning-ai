# Phase F — Computer Architecture and GPU Execution

> **Phase F** · File `05-computer-architecture-and-gpu-computing.md` · Depends on: [Phase C](02-deep-learning-frameworks.md), [Phase D](03-transformers-and-llms.md); [Phase E](04-training-and-optimization.md) memory accounting recommended · Next: [Phase G — Interpretability](06-model-interpretability.md)
> Status is tracked in the [progress tracker](README.md#progress-tracker). Notation follows [docs/notation.md](../notation.md).

## Contents

1. [Purpose](#1-purpose)
2. [Prerequisites](#2-prerequisites)
3. [Learning objectives](#3-learning-objectives)
4. [Theory](#4-theory)
5. [Implementation](#5-implementation)
6. [Experiments](#6-experiments)
7. [Resources](#7-resources)
8. [Completion criteria](#8-completion-criteria)
9. [Research directions](#9-research-directions)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Purpose

Models run on physical hardware, and hardware constraints shape which architectures, precisions, and algorithms are practical. FlashAttention, quantization, KV-cache design, and mixture-of-experts are all responses to hardware limits — mostly memory bandwidth rather than arithmetic. Understanding execution at the level of caches, warps, and memory traffic lets you predict performance, read systems papers, and decide when a custom kernel is justified (rarely) versus when framework operations are sufficient (usually).

**Ordering principle:** C++ and CUDA come only after you understand the mathematics of the operations you are accelerating (Phases A–D). You should already know what a matmul, softmax, and attention compute before learning how to make them fast.

## 2. Prerequisites

- [Phase D](03-transformers-and-llms.md): the operations inside a Transformer and their FLOP counts.
- [Phase C](02-deep-learning-frameworks.md): profiling and asynchronous GPU execution.
- [Phase A §4.5](00-foundations.md#45-floating-point-and-numerical-stability): floating point.
- **Learn at the start of this phase:** C and C++ fundamentals — pointers, memory layout, stack versus heap, compilation and linking, RAII. A CPU-only C++ track (F1, F2) comes before CUDA.
- **Hardware:** CUDA work needs an NVIDIA GPU, locally or in the cloud. See [open questions](README.md#open-questions-for-the-learner).

## 3. Learning objectives

### 3.1 Core — CPU and memory

- CPU cores, instruction pipelines, SIMD vector units (for example AVX), and multithreading.
- The memory hierarchy: registers, L1/L2/L3 caches, cache lines, RAM; latency versus bandwidth; spatial and temporal locality.
- Why loop order and blocking change matrix-multiplication speed by large factors.
- Optimized libraries (BLAS) and why they are hard to beat.

### 3.2 Core — GPU execution

- GPU organization: streaming multiprocessors (SMs), warps of 32 threads executing in lockstep (SIMT), thread blocks, grids.
- GPU memory: registers, shared memory, L2 cache, and high-bandwidth device memory (VRAM/HBM); host–device transfer over PCIe or similar interconnects; pinned memory.
- Occupancy, latency hiding, warp divergence.
- Tensor cores and reduced-precision matrix instructions.
- Arithmetic intensity and the roofline model; memory-bound versus compute-bound operations.
- Kernel fusion and launch overhead.
- Floating-point formats: float32, TF32, float16, bfloat16, float8 (E4M3/E5M2), int8; accumulation precision.

### 3.3 Core — CUDA and kernels

- Kernels, thread indexing, memory coalescing, shared-memory tiling, bank conflicts, synchronization, reductions, atomics, streams.
- Profiling with `torch.profiler`, NVIDIA Nsight Systems (timeline) and Nsight Compute (per-kernel analysis).
- Integrating a custom kernel into PyTorch as an extension with a backward pass.
- Triton as a higher-level alternative to raw CUDA.

### 3.4 When is a custom kernel justified?

Write one only when profiling shows a real bottleneck that the framework cannot address — typically a chain of memory-bound operations that could be fused, a novel operation with no efficient primitive, or a memory-access pattern the framework handles poorly. First try: better tensor layouts, larger batches, `torch.compile`, and existing fused operations. For learning, writing kernels is valuable regardless; for research code, prefer framework operations unless measurements say otherwise.

### 3.5 Optional advanced

CUTLASS and CuTe, FlashAttention internals, structured sparsity, multi-GPU communication (NCCL, NVLink), AMD ROCm/HIP, compiler stacks (TVM, MLIR).

## 4. Theory

### 4.1 Arithmetic intensity and the roofline

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved to and from memory}}, \qquad P_{\text{attainable}} = \min\left(P_{\text{peak}},\ I \times B_{\text{mem}}\right)
$$

The ridge point $I^* = P_{\text{peak}} / B_{\text{mem}}$ separates memory-bound operations ($I < I^*$) from compute-bound ones. For a hypothetical accelerator with $100$ TFLOP/s and $2$ TB/s, $I^* = 50$ FLOPs/byte. Look up real values from vendor datasheets for the hardware you use.

### 4.2 Why matmuls are compute-bound and elementwise operations are not

- **Square matmul** ($n \times n$, $b$ bytes per element): $2n^3$ FLOPs, at least $3n^2 b$ bytes, so $I \approx \frac{2n}{3b}$, which grows with $n$. Large matmuls are compute-bound.
- **Elementwise operation** (for example adding a bias or applying GELU) on float32: about 1 FLOP per element versus 8 bytes (read 4, write 4), so $I \approx 0.125$. Always memory-bound.

This is why fusing elementwise operations into a matmul, or into each other, saves time: it removes round trips to memory.

### 4.3 Decoding is bandwidth-bound

At batch size 1, generating one token reads every weight once and performs about 2 FLOPs per parameter. With 16-bit weights that is about 1 FLOP per byte — far below typical ridge points. An upper bound on decode speed is therefore

$$
\text{tokens/s} \lesssim \frac{B_{\text{mem}}}{\text{bytes of weights} + \text{bytes of KV cache read per token}}
$$

This explains why batching, quantization, and multi-query/grouped-query attention ([Phase D §4.6](03-transformers-and-llms.md#46-inference-and-the-kv-cache)) speed up inference.

### 4.4 Tiling

A naive matmul kernel reads each input element from global memory $O(n)$ times. Loading $t \times t$ tiles into fast on-chip memory (CPU cache or GPU shared memory) and reusing them reduces global memory traffic by a factor of about $t$. The same idea, applied to attention with an online softmax, is the core of FlashAttention.

### 4.5 Coalescing and warps

A warp's 32 threads issue memory requests together. If consecutive threads access consecutive addresses, the hardware combines them into a few wide transactions (coalesced); scattered accesses waste bandwidth. Branches that split a warp (divergence) serialize execution of both paths.

### 4.6 Floating-point formats

| Format | Exponent | Fraction | Typical use |
|---|---|---|---|
| float32 | 8 | 23 | Accumulation, master weights |
| TF32 | 8 | 10 | Tensor-core matmul input with float32 range (NVIDIA) |
| bfloat16 | 8 | 7 | Training activations and weights |
| float16 | 5 | 10 | Training with loss scaling; inference |
| float8 E4M3 / E5M2 | 4 / 5 | 3 / 2 | Low-precision training and inference on recent hardware |
| int8 / int4 | — | — | Quantized inference |

Products are often accumulated in higher precision than the inputs; whether a kernel accumulates in float32 changes results.

## 5. Implementation

CPU track first, then GPU.

- [ ] **F1** C++ matmul: naive (ijk), reordered (ikj), and cache-blocked versions; report GFLOP/s against NumPy's BLAS-backed matmul.
- [ ] **F2** A STREAM-style memory-bandwidth benchmark (copy, scale, add, triad) in C++.
- [ ] **F3** CUDA: vector addition, then matmul in progressive versions — naive, coalesced, shared-memory tiled — compared against cuBLAS (via `torch.matmul`).
- [ ] **F4** CUDA reduction (sum) and a numerically stable row-wise softmax kernel; validate against PyTorch.
- [ ] **F5** A fused kernel (for example bias + GELU, or a fused softmax) in Triton or CUDA; measure it against the unfused PyTorch sequence.
- [ ] **F6** Wrap your kernel as a PyTorch C++/CUDA extension with a backward pass; validate with `gradcheck`.
- [ ] **F7** Profile one training step of your Phase D GPT with `torch.profiler` and Nsight Systems; classify the top kernels as memory- or compute-bound.
- [ ] **F8** *(optional)* A tiled attention forward pass with online softmax (FlashAttention-style) in Triton for small sizes; compare peak memory with naive attention.

## 6. Experiments

1. **Roofline placement.** Measure achieved FLOP/s and bandwidth for your kernels and for PyTorch operations of varying arithmetic intensity; plot them on a roofline for your hardware.
2. **Matmul precision and size.** Throughput versus size for float32, TF32, and bfloat16; identify where tensor cores engage.
3. **Launch overhead.** Many small operations versus one fused operation; then the same sequence under `torch.compile` or CUDA graphs.
4. **Host–device transfer.** Bandwidth of pageable versus pinned memory transfers.
5. **Decode bound.** Predict tokens per second for your model from §4.3, then measure it; explain the gap.
6. **Reduction determinism.** Compare atomic-based and tree-based reductions for run-to-run variability of results.

## 7. Resources

**Core**

- Bryant and O'Hallaron — *Computer Systems: A Programmer's Perspective* (3rd ed.), especially memory hierarchy chapters: <https://csapp.cs.cmu.edu/>.
- Hwu, Kirk, El Hajj — *Programming Massively Parallel Processors* (4th ed., 2022).
- NVIDIA — CUDA Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/>; CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>.
- NVIDIA — "GPU Performance Background User's Guide": <https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html>.
- Horace He — "Making Deep Learning Go Brrrr From First Principles": <https://horace.io/brrr_intro.html>.
- Simon Boehm — "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance": <https://siboehm.com/articles/22/CUDA-MMM>.

**Supplementary**

- Hennessy and Patterson — *Computer Architecture: A Quantitative Approach* (6th ed.).
- Ulrich Drepper — "What Every Programmer Should Know About Memory" (2007): <https://www.akkadia.org/drepper/cpumemory.pdf>.
- learncpp.com, a free C++ tutorial: <https://www.learncpp.com/>.
- GPU MODE lecture materials: <https://github.com/gpu-mode/lectures>.
- Triton documentation and tutorials: <https://triton-lang.org/>.
- PyTorch custom C++ and CUDA extensions: <https://pytorch.org/tutorials/advanced/cpp_extension.html>.
- NVIDIA Nsight Systems <https://developer.nvidia.com/nsight-systems> and Nsight Compute <https://developer.nvidia.com/nsight-compute>.
- NVIDIA CUTLASS: <https://github.com/NVIDIA/cutlass>.
- AMD ROCm documentation (if using AMD GPUs): <https://rocm.docs.amd.com/>.

**Papers**

- Williams, Waterman, Patterson — "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *CACM* (2009): <https://doi.org/10.1145/1498765.1498785>.
- Tillet, Kung, Cox — "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (2019): <https://doi.org/10.1145/3315508.3329973>.
- Dao et al. — "FlashAttention" (2022): <https://arxiv.org/abs/2205.14135>; Dao — "FlashAttention-2" (2023): <https://arxiv.org/abs/2307.08691>.
- Chen et al. — "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (2018): <https://arxiv.org/abs/1802.04799>.

## 8. Completion criteria

- [ ] Explain SMs, warps, thread blocks, shared memory, and coalescing using your own kernels as examples.
- [ ] F3: a tiled CUDA matmul reaching a documented fraction of cuBLAS performance, with an analysis of the remaining gap.
- [ ] A roofline plot with measured points for your kernels (Experiment 1).
- [ ] Predict whether a given operation is memory- or compute-bound, then confirm it by profiling.
- [ ] F6: a custom PyTorch operation that passes `gradcheck`.
- [ ] Explain, with a concrete example, when a custom kernel is justified and when it is not.

## 9. Research directions

- Can attention variants be designed jointly with their kernels for better hardware efficiency?
- How do low-precision formats (float8, int4) affect training stability and model quality, and where must precision stay high?
- How much of the gap between hand-written kernels and compiler-generated code can automatic fusion close?
- Energy cost per token: how do precision and batch size change it?

## 10. Common misconceptions

- **"A GPU core is like a CPU core, just more of them."** GPU threads execute in warps with shared instruction streams; the programming and performance models differ substantially.
- **"FLOPs determine speed."** Many operations are limited by memory bandwidth, not arithmetic.
- **"Peak TFLOP/s is achievable."** Real workloads reach a fraction of peak; measure MFU.
- **"Custom CUDA always beats PyTorch."** PyTorch dispatches to heavily tuned libraries; naive custom kernels are usually slower.
- **"Maximum occupancy means maximum performance."** Occupancy helps hide latency but is not a goal in itself; register and shared-memory use per thread often matter more.
- **"float16 is just a less precise float32."** Its exponent range is far smaller, which causes overflow and underflow that bfloat16 avoids.

---

Previous: [Phase E — Training Architecture and Model Optimization](04-training-and-optimization.md) · Next: [Phase G — Interpretability](06-model-interpretability.md) · Up: [Roadmap](README.md)
