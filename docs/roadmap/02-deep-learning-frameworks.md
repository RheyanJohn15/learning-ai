# Phase C — Deep Learning Frameworks

> **Phase C** · File `02-deep-learning-frameworks.md` · Depends on: [Phase B](01-neural-networks.md) through milestone B3 (NumPy basics can start earlier) · Next: [Phase D — Transformers and LLMs](03-transformers-and-llms.md)
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

Research code is written in frameworks. PyTorch is the dominant framework in published research, so you need to read and write it fluently. The goal here is not "learn the API" but to map every PyTorch abstraction back to something you built by hand in [Phase B](01-neural-networks.md): `Tensor` is a strided NumPy-like array that can live on a GPU, `autograd` is your micrograd engine vectorized, `nn.Module` is a parameter container, and `optim.SGD` is the update rule you wrote.

## 2. Prerequisites

- [Phase B](01-neural-networks.md) milestones B1–B3 (manual backpropagation). B5 (your own autodiff engine) is strongly recommended before relying on autograd.
- Python fluency: classes, generators, context managers, virtual environments.

## 3. Learning objectives

### 3.1 Python and NumPy for scientific computing (can start alongside Phase A)

- Arrays, dtypes, shapes, indexing, slicing, and fancy indexing.
- Broadcasting rules and vectorization (replacing Python loops with array operations).
- Memory layout: row-major order, strides, views versus copies.
- `einsum` notation for contractions (useful for attention in [Phase D](03-transformers-and-llms.md)).
- Plotting results (Matplotlib) to inspect training curves and data.

### 3.2 PyTorch core

- **Tensors:** dtype, device, shape, strides, `contiguous()`, `view` versus `reshape`, in-place operations.
- **Autograd:** dynamic graph construction, `requires_grad`, `.grad` accumulation, `backward()` on non-scalar outputs (vector–Jacobian products), `torch.no_grad()` and `torch.inference_mode()`, `detach()`, and custom `torch.autograd.Function`.
- **Modules:** `nn.Module`, parameter registration, `state_dict`, `train()` versus `eval()` modes.
- **Optimizers:** what `optimizer.step()` and `optimizer.zero_grad()` actually do to tensors.
- **Data:** `Dataset`, `DataLoader`, batching, shuffling, collation, worker processes.
- **Training and evaluation loops**, metrics, and logging.

### 3.3 Engineering practice

- **CPU versus GPU execution:** device placement, host–device transfers, asynchronous kernel launches, and synchronization when timing.
- **Reproducibility:** seeding Python, NumPy, and PyTorch; deterministic algorithms; known sources of non-determinism.
- **Debugging:** overfit a single batch, shape assertions, `torch.autograd.gradcheck`, anomaly detection for NaNs, inspecting gradient norms.
- **Profiling:** `torch.profiler` and correct wall-clock measurement.
- **Checkpointing:** saving model, optimizer, scheduler, RNG state, and step counter so training can resume exactly.

### 3.4 What the framework does for you, and what it does not

| The framework automates… | …but you must still understand |
|---|---|
| Recording the graph and computing vector–Jacobian products | What the gradient of your loss means, and when it is wrong for your problem |
| Kernel selection and dispatch to CPU/GPU | Which operations are expensive, and why (FLOPs versus memory traffic) |
| Memory allocation (a caching allocator on GPU) | What consumes memory: parameters, gradients, optimizer state, activations |
| Broadcasting | Whether a broadcast was intended or is silently hiding a shape bug |
| Parameter registration and initialization defaults | Whether the default initialization suits your architecture |
| Fused, numerically stable losses such as `cross_entropy` on logits | Why passing probabilities instead of logits is unstable |
| Gradient accumulation into `.grad` | That gradients persist until you zero them |
| Mode switches (`train()` / `eval()`) | That dropout and normalization layers behave differently in each mode |

Data leakage, evaluation design, model mathematics, and numerical stability are never automated.

## 4. Theory

### 4.1 Strides and views

A tensor is a flat block of memory plus a shape and strides. Element $(i, j)$ of a 2-D tensor lives at offset

$$
\text{offset}(i, j) = \text{storage\_offset} + i \cdot s_0 + j \cdot s_1
$$

Transposing swaps strides without copying, which is why a transposed tensor is non-contiguous and why `view` can fail where `reshape` (which may copy) succeeds.

### 4.2 Broadcasting

Compare shapes from the trailing dimension backwards. Two dimensions are compatible if they are equal or one of them is 1; missing leading dimensions are treated as 1. In the backward pass, the gradient of a broadcast tensor is summed over the broadcast dimensions — exactly the bias rule from [Phase B §4.2](01-neural-networks.md#42-backpropagation-in-matrix-form).

### 4.3 Autograd as reverse-mode differentiation

For $y = f(x)$, calling `y.backward(v)` computes the vector–Jacobian product $v^\top J_f(x)$ without ever forming $J_f$. For a scalar loss, $v = 1$. Each primitive operation registers a backward function; PyTorch runs them in reverse topological order and **adds** into `.grad`. This is your B5 engine with tensors instead of scalars.

### 4.4 Asynchronous GPU execution

CUDA kernel launches return immediately; the CPU queues work and continues. Timing GPU code without `torch.cuda.synchronize()` measures launch time, not execution time. Operations that need a value on the CPU (`.item()`, printing, `.cpu()`) force synchronization and can silently serialize a training loop.

## 5. Implementation

- [ ] **C1** Port the Phase B MLP to PyTorch tensors **with manual gradients** (no autograd); confirm the results match the NumPy version.
- [ ] **C2** Switch to autograd and confirm autograd gradients equal your manual gradients within tolerance.
- [ ] **C3** Implement a custom `torch.autograd.Function` (for example a linear layer or GELU) with a hand-written backward pass and verify it with `gradcheck`.
- [ ] **C4** Write your own minimal `Linear` module and SGD optimizer; confirm identical training trajectories against `nn.Linear` and `torch.optim.SGD` with the same seed.
- [ ] **C5** A complete, small training pipeline on MNIST or Fashion-MNIST: dataset, data loader, training and evaluation loops, checkpoint save and resume, seeding, and a results log.
- [ ] **C6** CPU versus GPU matrix-multiplication benchmark across sizes with correct synchronization (only if a GPU is available; see [open questions](README.md#open-questions-for-the-learner)).

## 6. Experiments

1. **Exact resume.** Hypothesis: training for $2n$ steps continuously and training $n$ steps, checkpointing, and resuming for $n$ steps produce identical loss curves when all state is saved. Test it, then remove pieces of state (RNG, optimizer) and measure the divergence.
2. **Where the GPU wins.** Measure matmul throughput (achieved FLOP/s, using $2mnk$ FLOPs) versus matrix size on CPU and GPU. Find the size where the GPU becomes faster and explain why small operations do not benefit.
3. **Batch size versus wall-clock time.** At a fixed number of epochs, measure time per epoch and final accuracy across batch sizes.
4. **Forgotten `zero_grad`.** A controlled demonstration of unintended gradient accumulation and its effect on effective learning rate.
5. **Input pipeline bottleneck.** Throughput versus number of data-loader workers; determine whether training is compute-bound or input-bound.

## 7. Resources

**Core**

- NumPy documentation: <https://numpy.org/doc/stable/>, especially broadcasting: <https://numpy.org/doc/stable/user/basics.broadcasting.html>.
- PyTorch documentation: <https://pytorch.org/docs/stable/index.html>.
- PyTorch notes: autograd mechanics <https://pytorch.org/docs/stable/notes/autograd.html>, CUDA semantics <https://pytorch.org/docs/stable/notes/cuda.html>, reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>.
- PyTorch recipes: profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>, saving and loading <https://pytorch.org/tutorials/beginner/saving_loading_models.html>.
- Andrej Karpathy — "A Recipe for Training Neural Networks" (2019): <https://karpathy.github.io/2019/04/25/recipe/>.

**Supplementary**

- Edward Z. Yang — "PyTorch internals" (2019): <http://blog.ezyang.com/2019/05/pytorch-internals/>.
- Jake VanderPlas — *Python Data Science Handbook*, free: <https://jakevdp.github.io/PythonDataScienceHandbook/>.
- Nicolas Rougier — *From Python to NumPy*, free: <https://www.labri.fr/perso/nrougier/from-python-to-numpy/>.
- tinygrad, a small readable framework: <https://github.com/tinygrad/tinygrad>.
- JAX documentation (an alternative functional design worth comparing): <https://docs.jax.dev/>.

**Papers**

- Paszke et al. — "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019): <https://arxiv.org/abs/1912.01703>.

## 8. Completion criteria

- [ ] Explain what `loss.backward()` does in terms of the recorded graph, vector–Jacobian products, and `.grad` accumulation.
- [ ] C2: manual and autograd gradients match within a documented tolerance.
- [ ] C5: training resumes from a checkpoint with subsequent losses matching an uninterrupted run (or deviations are explained).
- [ ] Demonstrate correct GPU timing and show how incorrect timing misleads.
- [ ] Explain views versus copies and when a tensor is non-contiguous.
- [ ] Use the overfit-a-single-batch check to find a deliberately introduced bug.

## 9. Research directions

- How do automatic-differentiation systems trade memory for compute (graph retention, recomputation)?
- Graph compilers (`torch.compile`, XLA): what fusions do they find, and when do they fail?
- Sources of GPU non-determinism and their effect on the reproducibility of published results.

## 10. Common misconceptions

- **"Autograd means I no longer need to understand derivatives."** You still need them to design losses, debug NaNs, diagnose vanishing gradients, and read papers.
- **"The GPU is always faster."** Small operations are dominated by launch overhead and data transfer.
- **"`model.eval()` turns off gradients."** It switches layer behavior (dropout, batch normalization). Gradient tracking is controlled by `torch.no_grad()` or `torch.inference_mode()`.
- **"`time.time()` around GPU code measures GPU time."** Not without synchronization.
- **"Setting a seed guarantees bitwise reproducibility."** Non-deterministic kernels, library versions, and hardware all matter.
- **"Saving the model weights is enough to resume training."** Optimizer state, learning-rate schedule, RNG state, and data position are also needed.

---

Previous: [Phase B — Neural Networks from Scratch](01-neural-networks.md) · Next: [Phase D — Transformers and LLMs](03-transformers-and-llms.md) · Up: [Roadmap](README.md)
