# Phase E — Training Architecture and Model Optimization

> **Phase E** · File `04-training-and-optimization.md` · Depends on: [Phase C](02-deep-learning-frameworks.md), [Phase D](03-transformers-and-llms.md) through milestone D7 · Next: [Phase F — Computer Architecture and GPU Execution](05-computer-architecture-and-gpu-computing.md)
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

Knowing the architecture is not the same as being able to train it. Real training runs are shaped by optimizer choice, learning-rate schedules, numerical precision, memory limits, data quality, and failure recovery. Many empirical research results depend on these details, and many failed replications trace back to them. This phase teaches you to reason quantitatively about a training run: how much memory and compute it needs, why it diverges, and how to make it reliable.

## 2. Prerequisites

- [Phase D](03-transformers-and-llms.md) through D7: a working small GPT and training loop.
- [Phase C](02-deep-learning-frameworks.md): checkpointing, profiling, reproducibility.
- [Phase A §4.6](00-foundations.md#46-gradient-descent): gradient descent on quadratics and conditioning.
- Learn just in time: second-order Taylor expansion and the Hessian.

## 3. Learning objectives

### 3.1 Core

- **Data pipelines and data quality:** offline tokenization, sharding, shuffling, streaming, exact and near-duplicate removal, quality filtering, train/validation leakage, and benchmark contamination.
- **Batching:** batch size versus gradient noise; tokens per batch; gradient accumulation and when it is exactly equivalent to a larger batch.
- **Optimizers:** SGD, momentum, Nesterov momentum, RMSProp, Adam, and AdamW; optimizer state and its memory cost.
- **Learning-rate schedules:** warmup, cosine decay, linear decay; why warmup helps.
- **Stability:** gradient clipping, loss spikes, numerical overflow and underflow.
- **Mixed precision:** float16 with loss scaling, bfloat16, float32 master weights, autocast.
- **Memory accounting:** parameters, gradients, optimizer states, activations; activation checkpointing.
- **Compute accounting:** FLOPs per token, total training FLOPs, model FLOPs utilization (MFU).
- **Checkpointing and recovery:** atomic writes, complete state, resuming data position.
- **Training versus inference bottlenecks.**

### 3.2 Distributed training (concepts; small-scale implementation optional)

Data parallelism and all-reduce; sharded optimizer state (ZeRO, FSDP); tensor parallelism; pipeline parallelism; communication costs.

### 3.3 Optional advanced

Hyperparameter transfer across scale (µP), second-order and preconditioned optimizers, FP8 training, data-mixture optimization.

## 4. Theory

### 4.1 Momentum, Adam, and AdamW

SGD with momentum ($g_t$ is the minibatch gradient):

$$
v_t = \mu v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\, v_t
$$

Adam keeps exponential moving averages of the gradient and squared gradient, with bias correction:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, & v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, & \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} & &
\end{aligned}
$$

AdamW applies weight decay **decoupled** from the adaptive update: $\theta_t \leftarrow \theta_t - \eta\lambda\theta_{t-1}$. Adding $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss instead would have its gradient rescaled by $1/\sqrt{\hat v_t}$, which is not the same thing.

Adam stores two extra tensors ($m$, $v$) the size of the parameters.

### 4.2 Gradient accumulation

With $K$ micro-batches of equal size, each with a *mean* loss $L_k$:

$$
g = \frac{1}{K}\sum_{k=1}^{K} \nabla_\theta L_k
$$

equals the gradient of the mean loss over the combined batch. Equivalence breaks with batch normalization, with unequal micro-batch sizes or variable numbers of non-padding tokens (unless losses are normalized by total token count), and with differing dropout randomness.

### 4.3 Learning-rate schedules

Linear warmup over $T_w$ steps followed by cosine decay to $\eta_{\min}$ over $T$ steps:

$$
\eta_t = \begin{cases} \eta_{\max}\, t / T_w & t < T_w \\[4pt] \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi (t - T_w)}{T - T_w}\right) & t \ge T_w \end{cases}
$$

One common explanation for warmup is that Adam's second-moment estimate is unreliable in early steps, and early updates at a large learning rate can push the model into unstable regions. Treat the mechanism as partly an open question and test it (Experiment 2).

### 4.4 Mixed precision

| Format | Exponent bits | Fraction bits | Approx. max | Notes |
|---|---|---|---|---|
| float32 | 8 | 23 | $3.4 \times 10^{38}$ | Reference precision |
| float16 | 5 | 10 | $65504$ | Small range: gradients can underflow, activations can overflow |
| bfloat16 | 8 | 7 | $3.4 \times 10^{38}$ | Same range as float32, much less precision |

With float16, **loss scaling** multiplies the loss by $S$ before `backward()` so small gradients stay representable, then divides gradients by $S$ before the update, skipping steps whose gradients contain infinities. Master weights are typically kept in float32 because small updates vanish when added to large 16-bit values.

### 4.5 Memory accounting

$$
M_{\text{total}} \approx \underbrace{N(b_{\text{param}} + b_{\text{grad}} + b_{\text{opt}})}_{\text{model state}} + M_{\text{activations}} + M_{\text{temporary}}
$$

With mixed-precision Adam, the ZeRO paper (Rajbhandari et al., 2019) counts 16 bytes per parameter: 2 (16-bit weights) + 2 (16-bit gradients) + 4 (float32 master weights) + 4 + 4 (Adam $m$ and $v$). A 1B-parameter model therefore needs about 16 GB for model state alone, before activations. Activation memory grows with batch size, sequence length, width, and depth, roughly proportional to $B \cdot T \cdot d \cdot n_{\text{layer}}$ plus attention terms that grow with $T^2$ when attention scores are materialized. Activation checkpointing discards activations and recomputes them in the backward pass, trading compute for memory.

### 4.6 Compute accounting

Training FLOPs for a dense Transformer with $N$ parameters on $D$ tokens:

$$
C \approx 6ND
$$

Wall-clock estimate: $t \approx C / (n_{\text{devices}} \times P_{\text{peak}} \times \text{MFU})$, where MFU is achieved model FLOP/s divided by peak hardware FLOP/s.

Illustrative arithmetic: a 124M-parameter model on 10B tokens needs $6 \times 1.24 \times 10^8 \times 10^{10} \approx 7.4 \times 10^{18}$ FLOPs. At an *effective* $10^{14}$ FLOP/s, that is about $7.4 \times 10^4$ s, roughly 21 hours. Use your actual hardware's datasheet and your measured MFU for real estimates.

Hoffmann et al. (2022) found compute-optimal training used roughly 20 tokens per parameter in their setting; treat this as an empirical result under their assumptions, not a law.

### 4.7 Data parallelism

Each of $p$ devices holds a full model replica, processes a different micro-batch, and the gradients are averaged with an all-reduce. A ring all-reduce sends and receives about $2\frac{p-1}{p}$ times the gradient size per device, independent of $p$ for large $p$. Sharding methods (ZeRO, FSDP) partition optimizer state, gradients, and parameters across devices to cut the per-device memory in exchange for more communication.

## 5. Implementation

- [ ] **E1** Implement SGD with momentum, Adam, and AdamW yourself; match `torch.optim` updates within tolerance over 100 steps.
- [ ] **E2** Implement warmup plus cosine scheduling; plot it; confirm it against your training logs.
- [ ] **E3** Demonstrate numerically that gradient accumulation equals a large batch, then break the equivalence deliberately (unequal micro-batches) and measure the difference.
- [ ] **E4** Mixed-precision training of your Phase D GPT with autocast; implement float16 loss scaling by hand once.
- [ ] **E5** A memory estimator: predict peak memory from a model configuration and compare it with the measured peak (for example `torch.cuda.max_memory_allocated`).
- [ ] **E6** Robust checkpointing: atomic writes, full state including data position and RNG, and a kill-and-resume test.
- [ ] **E7** A data pipeline: pre-tokenize a corpus into binary shards, memory-map them, remove exact duplicates by hashing, and measure throughput.
- [ ] **E8** Activation checkpointing; measure the memory/time trade-off.
- [ ] **E9** *(optional)* Two-process data-parallel training on CPU with `torch.distributed` (gloo backend); confirm it matches single-process training with the combined batch.

## 6. Experiments

1. **Fair optimizer comparison.** SGD, SGD with momentum, Adam, and AdamW on your small GPT, each with its own learning-rate sweep (untuned baselines are a classic flaw; see [Phase J](09-research-methodology.md#10-common-misconceptions)). Report multiple seeds.
2. **Warmup ablation.** Frequency of loss spikes or divergence versus warmup length at a high learning rate, across seeds.
3. **Critical batch size.** Steps and tokens needed to reach a target loss as a function of batch size; compare against McCandlish et al. (2018).
4. **Precision.** float32 versus bfloat16 versus float16 (with loss scaling): final loss, throughput, and number of skipped steps.
5. **Memory model.** Error of your E5 estimate across configurations; identify which term you mis-modeled.
6. **Data deduplication.** At small scale, does removing duplicates change validation loss or memorization? Compare with Lee et al. (2021).
7. **Throughput.** Tokens per second and estimated MFU; locate the main bottleneck with the profiler.

## 7. Resources

**Core**

- Goodfellow, Bengio, Courville — *Deep Learning*, Chapter 8 (optimization): <https://www.deeplearningbook.org/>.
- Hugging Face — "The Ultra-Scale Playbook" (distributed training): <https://huggingface.co/spaces/nanotron/ultrascale-playbook>.
- Google DeepMind authors — "How to Scale Your Model" (TPU-oriented, but the reasoning transfers): <https://jax-ml.github.io/scaling-book/>.
- Stas Bekman — *Machine Learning Engineering* (open book): <https://github.com/stas00/ml-engineering>.
- PyTorch automatic mixed precision documentation: <https://pytorch.org/docs/stable/amp.html>.

**Papers — optimization**

- Sutskever et al. — "On the importance of initialization and momentum in deep learning" (2013): <https://proceedings.mlr.press/v28/sutskever13.html>.
- Kingma and Ba — "Adam: A Method for Stochastic Optimization" (2014): <https://arxiv.org/abs/1412.6980>.
- Loshchilov and Hutter — "Decoupled Weight Decay Regularization" (AdamW, 2017): <https://arxiv.org/abs/1711.05101>.
- Loshchilov and Hutter — "SGDR: Stochastic Gradient Descent with Warm Restarts" (2016): <https://arxiv.org/abs/1608.03983>.
- Pascanu, Mikolov, Bengio — "On the difficulty of training recurrent neural networks" (gradient clipping, 2012): <https://arxiv.org/abs/1211.5063>.
- Goyal et al. — "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017): <https://arxiv.org/abs/1706.02677>.
- McCandlish et al. — "An Empirical Model of Large-Batch Training" (2018): <https://arxiv.org/abs/1812.06162>.
- Keskar et al. — "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (2016): <https://arxiv.org/abs/1609.04836>.
- Ioffe and Szegedy — "Batch Normalization" (2015): <https://arxiv.org/abs/1502.03167>.

**Papers — precision, memory, and distribution**

- Micikevicius et al. — "Mixed Precision Training" (2017): <https://arxiv.org/abs/1710.03740>.
- Kalamkar et al. — "A Study of BFLOAT16 for Deep Learning Training" (2019): <https://arxiv.org/abs/1905.12322>.
- Micikevicius et al. — "FP8 Formats for Deep Learning" (2022): <https://arxiv.org/abs/2209.05433>.
- Chen et al. — "Training Deep Nets with Sublinear Memory Cost" (activation checkpointing, 2016): <https://arxiv.org/abs/1604.06174>.
- Korthikanti et al. — "Reducing Activation Recomputation in Large Transformer Models" (2022): <https://arxiv.org/abs/2205.05198>.
- Rajbhandari et al. — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2019): <https://arxiv.org/abs/1910.02054>.
- Zhao et al. — "PyTorch FSDP" (2023): <https://arxiv.org/abs/2304.11277>.
- Shoeybi et al. — "Megatron-LM" (tensor parallelism, 2019): <https://arxiv.org/abs/1909.08053>.
- Huang et al. — "GPipe" (pipeline parallelism, 2018): <https://arxiv.org/abs/1811.06965>.

**Papers — data**

- Lee et al. — "Deduplicating Training Data Makes Language Models Better" (2021): <https://arxiv.org/abs/2107.06499>.
- Penedo et al. — "The FineWeb Datasets" (2024): <https://arxiv.org/abs/2406.17557>.

## 8. Completion criteria

- [ ] E1: your Adam and AdamW match PyTorch within $10^{-6}$ (float32) over 100 steps.
- [ ] E5: your memory estimate is within a documented tolerance (aim for about 20%) of the measured peak, with the error explained.
- [ ] E6: a kill-and-resume test produces a continuation identical to an uninterrupted run, or deviations are explained.
- [ ] For an $N$-parameter model, state the memory needed for model state under mixed-precision Adam and explain each term.
- [ ] Estimate the training FLOPs and wall-clock time for a given model, dataset size, and hardware.
- [ ] Explain the range/precision trade-off between float16 and bfloat16 with numbers.

## 9. Research directions

- Hyperparameter transfer: Yang et al., "Tensor Programs V" (µP, 2022): <https://arxiv.org/abs/2203.03466>.
- Optimizer design: Gupta et al., "Shampoo" (2018): <https://arxiv.org/abs/1802.09568>; Chen et al., "Symbolic Discovery of Optimization Algorithms" (Lion, 2023): <https://arxiv.org/abs/2302.06675>.
- What causes loss spikes in Transformer training, and which interventions reliably prevent them?
- How low can training precision go, and which parts of the computation need high precision?
- Data selection: which properties of training data matter most for small models?

## 10. Common misconceptions

- **"A bigger batch is always better" / "scale the learning rate linearly with batch size."** The linear-scaling heuristic holds only up to a problem-dependent critical batch size.
- **"Adam does not need tuning."** Adam's learning rate, $\beta_2$, $\epsilon$, and weight decay all matter, especially at scale.
- **"Weight decay and L2 regularization are the same."** Under SGD they coincide up to a learning-rate rescaling; under Adam they do not (hence AdamW).
- **"Mixed precision means everything runs in 16-bit."** Master weights, some reductions, and loss computations usually stay in float32.
- **"Parameters dominate GPU memory."** During training, optimizer state and activations usually dominate.
- **"Gradient accumulation is always exactly equivalent to a bigger batch."** See §4.2 for the exceptions.
- **"If the loss is going down, the data must be fine."** Duplicated, contaminated, or mislabeled data can still lower the training loss.

---

Previous: [Phase D — Transformers and LLMs](03-transformers-and-llms.md) · Next: [Phase F — Computer Architecture and GPU Execution](05-computer-architecture-and-gpu-computing.md) · Up: [Roadmap](README.md)
