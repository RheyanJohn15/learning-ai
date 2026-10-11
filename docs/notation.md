# Notation and Shape Conventions

Shared symbols for all documents and, later, code comments. When a source uses different notation, translate it into these symbols in your notes and say so.

## Scalars, vectors, matrices, tensors

| Symbol | Meaning |
|---|---|
| $x$, $\alpha$ | Scalars (lowercase italic) |
| $\mathbf{x}$ or $x$ | Vectors are column vectors; bold is used only where ambiguity would otherwise arise |
| $W$, $X$ | Matrices (uppercase) |
| $X \in \mathbb{R}^{N \times D}$ | Shape annotation: $N$ rows, $D$ columns |
| $x_i$, $X_{ij}$ | Elements (1-indexed in math, 0-indexed in code) |
| $X^\top$ | Transpose |
| $\odot$ | Elementwise (Hadamard) product |
| $\lVert x \rVert_2$ | Euclidean norm |

In ML code, data matrices are **row-major with one example per row**: a batch $X \in \mathbb{R}^{N \times D}$ is multiplied on the right, $XW$ with $W \in \mathbb{R}^{D \times H}$. Some textbooks use $Wx$ with column vectors instead; both are fine as long as shapes are checked.

## Standard dimension names

| Symbol | Meaning | Typical code name |
|---|---|---|
| $N$ | Number of examples (in a dataset or batch, non-sequence models) | `N` |
| $B$ | Batch size (sequence models) | `B` |
| $T$ | Sequence length / context length | `T` |
| $D$ | Input feature dimension | `D` |
| $H$ | Hidden width of an MLP | `H` |
| $K$ | Number of classes | `K` |
| $d$ | Model (residual stream) width, $d_{\text{model}}$ | `C` or `d_model` |
| $h$ | Number of attention heads | `n_head` |
| $d_h$ | Head dimension, usually $d / h$ | `head_dim` |
| $V$ | Vocabulary size | `vocab_size` |
| $n_{\text{layer}}$ | Number of Transformer blocks | `n_layer` |

Typical activation shapes in a decoder-only Transformer:

```text
token ids            (B, T)            integers in [0, V)
embeddings           (B, T, d)
Q, K, V per head     (B, h, T, d_h)
attention scores     (B, h, T, T)
block output         (B, T, d)
logits               (B, T, V)
```

## Learning and optimization

| Symbol | Meaning |
|---|---|
| $\theta$ | All trainable parameters |
| $N$ (context-dependent) | Parameter count in scaling and compute formulas ($C \approx 6ND$) |
| $D$ (context-dependent) | Number of training tokens in scaling formulas |
| $L$, $\mathcal{L}$ | Loss (scalar) |
| $\eta$ | Learning rate |
| $g_t$, $\nabla_\theta L$ | Gradient at step $t$ |
| $dT$ | Shorthand for $\partial L / \partial T$, same shape as $T$ |
| $\beta_1, \beta_2, \epsilon$ | Adam hyperparameters |
| $\lambda$ | Regularization or weight-decay strength |

$N$ and $D$ are overloaded between "batch/feature dimensions" and "parameters/tokens" because both usages are entrenched in the literature. The surrounding section always makes the meaning clear; say which one you mean when it could be ambiguous.

## Probability and information

| Symbol | Meaning |
|---|---|
| $p(x)$, $p_\theta(x)$ | Data distribution; model distribution with parameters $\theta$ |
| $\mathbb{E}_{x \sim p}[f(x)]$ | Expectation |
| $H(p)$, $H(p, q)$ | Entropy; cross-entropy |
| $D_{\mathrm{KL}}(p \Vert q)$ | KL divergence |
| $\sigma(\cdot)$ | Logistic sigmoid |
| $\log$ | Natural logarithm unless stated (losses in nats) |

## Reinforcement learning

| Symbol | Meaning |
|---|---|
| $s, a, r$ | State, action, reward |
| $\pi(a \mid s)$ | Policy |
| $\gamma$ | Discount factor |
| $G_t$ | Return from time $t$ |
| $V^\pi, Q^\pi$ | State-value and action-value functions |
| $\hat{A}_t$ | Advantage estimate |

## Hardware

| Symbol | Meaning |
|---|---|
| $P_{\text{peak}}$ | Peak arithmetic throughput (FLOP/s) |
| $B_{\text{mem}}$ | Memory bandwidth (bytes/s) |
| $I$ | Arithmetic intensity (FLOPs per byte) |
| MFU | Model FLOPs utilization: achieved / peak FLOP/s |

FLOPs (plural) counts operations; FLOP/s is a rate.
