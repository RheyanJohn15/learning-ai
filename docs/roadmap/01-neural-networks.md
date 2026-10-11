# Phase B — Neural Networks from Scratch

> **Phase B** · File `01-neural-networks.md` · Depends on: [Phase A](00-foundations.md) (essentials) · Next: [Phase C — Deep Learning Frameworks](02-deep-learning-frameworks.md)
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

A Transformer is a stack of matrix multiplications, nonlinearities, and normalizations trained by gradient descent on a cross-entropy loss. If you can derive and implement backpropagation for a small network by hand, every later architecture is a variation on the same mechanism. This phase builds that mechanism with **NumPy and manual gradients only**, before any automatic differentiation, so that autograd in [Phase C](02-deep-learning-frameworks.md) is something you have already built rather than a black box.

## 2. Prerequisites

- [Phase A §3.1](00-foundations.md#31-essential-before-phase-b) essentials: shapes, chain rule, gradients, cross-entropy, stable softmax.
- The A2 numerical gradient checker from [Phase A](00-foundations.md#5-implementation).
- NumPy basics: arrays, broadcasting, vectorized operations (see [Phase C §3](02-deep-learning-frameworks.md#3-learning-objectives)).
- Learn just in time: matrix calculus for vector–Jacobian products ([Parr and Howard, 2018](https://arxiv.org/abs/1802.01528)).

## 3. Learning objectives

### 3.1 Core

- **Linear regression:** model $\hat{y} = Xw + b$, mean-squared-error loss, closed-form normal equations, and gradient descent.
- **Logistic regression:** sigmoid, binary cross-entropy, and the gradient $(\hat{y} - y)x$. Extend to softmax regression for multiple classes.
- **Artificial neurons:** $a = \phi(w^\top x + b)$; activation functions (sigmoid, tanh, ReLU, GELU), their derivatives, and saturation.
- **Forward propagation and computational graphs:** decompose a network into primitive operations with local derivatives.
- **Backpropagation:** derive it mathematically for a batched two-layer MLP in matrix form; explain reverse-mode versus forward-mode differentiation and why reverse mode suits scalar losses.
- **Multilayer perceptrons:** why stacking linear layers without nonlinearities collapses to one linear map; nonlinear decision boundaries; XOR.
- **Training dynamics:** learning rate, mini-batch stochastic gradient descent, initialization and symmetry breaking.
- **Generalization:** train/validation/test splits, overfitting, L2 regularization (weight decay), early stopping, dropout.

### 3.2 Optional advanced

- Universal approximation theorems (what they do and do not guarantee).
- Bias–variance decomposition and double descent.
- Neural tangent kernel and infinite-width limits.

## 4. Theory

### 4.1 A two-layer MLP for classification

Input batch $X \in \mathbb{R}^{N \times D}$, hidden width $H$, $K$ classes, one-hot labels $Y \in \mathbb{R}^{N \times K}$:

$$
\begin{aligned}
Z_1 &= X W_1 + b_1, & W_1 &\in \mathbb{R}^{D \times H},\ Z_1 \in \mathbb{R}^{N \times H} \\
A_1 &= \phi(Z_1) & & \\
Z_2 &= A_1 W_2 + b_2, & W_2 &\in \mathbb{R}^{H \times K},\ Z_2 \in \mathbb{R}^{N \times K} \\
P &= \operatorname{softmax}(Z_2) \ \text{(row-wise)} & & \\
L &= -\frac{1}{N}\sum_{i=1}^{N} \log P_{i, y_i} & &
\end{aligned}
$$

Biases are broadcast across the batch dimension.

### 4.2 Backpropagation in matrix form

Write $dT$ for $\partial L / \partial T$. Applying the chain rule from the loss backwards:

$$
\begin{aligned}
dZ_2 &= \tfrac{1}{N}(P - Y) & & \text{(softmax + cross-entropy combine neatly)} \\
dW_2 &= A_1^\top dZ_2, & db_2 &= \textstyle\sum_{i} dZ_{2,i} \\
dA_1 &= dZ_2 W_2^\top & & \\
dZ_1 &= dA_1 \odot \phi'(Z_1) & & \\
dW_1 &= X^\top dZ_1, & db_1 &= \textstyle\sum_{i} dZ_{1,i}
\end{aligned}
$$

Two checks catch most mistakes:

1. **Shape rule:** every gradient has the same shape as the tensor it differentiates ($dW_1$ is $D \times H$).
2. **Broadcast rule:** if a tensor was broadcast in the forward pass (biases), its gradient is summed over the broadcast dimension in the backward pass.

Derive $dZ_2 = \frac{1}{N}(P - Y)$ yourself from the softmax Jacobian; it is the most important single derivation in this phase.

### 4.3 Why XOR needs a hidden layer

The four XOR points $\{(0,0) \to 0, (0,1) \to 1, (1,0) \to 1, (1,1) \to 0\}$ are not linearly separable, so logistic regression cannot fit them (the classic observation in Minsky and Papert, *Perceptrons*, 1969). A hidden layer with a nonlinearity maps the inputs into a space where they become separable. Two hidden units are sufficient in principle, but training from random initialization does not always find that solution.

### 4.4 Initialization

All-zero (or all-equal) hidden weights give every hidden unit identical gradients, so they remain identical forever: symmetry is never broken. Random initialization must also keep activation and gradient variances roughly constant across layers:

$$
\text{Xavier/Glorot: } \operatorname{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad \text{He: } \operatorname{Var}(W) = \frac{2}{n_{\text{in}}} \ \text{(for ReLU)}
$$

### 4.5 Regularization

L2 regularization adds $\frac{\lambda}{2}\lVert W \rVert^2$ to the loss, contributing $\lambda W$ to the gradient, which shrinks weights each step ("weight decay" under plain SGD; not equivalent under Adam — see [Phase E](04-training-and-optimization.md#10-common-misconceptions)). Dropout randomly zeroes activations during training and rescales the remaining ones, and must be disabled at evaluation time.

### 4.6 From manual gradients to automatic differentiation

A scalar reverse-mode autodiff engine records each operation in a graph during the forward pass, then visits nodes in reverse topological order, multiplying upstream gradients by local derivatives and **accumulating** (summing) gradients when a value is used more than once. Building one yourself (milestone B5) turns autograd from magic into bookkeeping.

## 5. Implementation

NumPy only. Verify every manual gradient with the A2 checker (use float64 for checks).

- [ ] **B1** Linear regression: closed-form solution and gradient descent; confirm they agree.
- [ ] **B2** Logistic regression and softmax regression with manual gradients, gradient-checked.
- [ ] **B3** **XOR milestone:** an MLP (e.g. 2-2-1 and 2-4-1) trained with manual backpropagation to 100% accuracy on the four points. Record the fraction of random seeds that succeed.
- [ ] **B4** A general MLP with configurable depth, width, and activation, trained with mini-batch SGD on a synthetic 2-D dataset (spirals or concentric circles), then on MNIST. Plot decision boundaries for the 2-D case. Datasets are downloaded locally and not committed.
- [ ] **B5** A scalar reverse-mode autodiff engine in the spirit of micrograd: `Value` objects, a graph, topological sort, and `backward()`.
- [ ] **B6** Rebuild the B3/B4 MLPs on top of your B5 engine and confirm its gradients match your manual gradients.

## 6. Experiments

1. **XOR reliability.** Hypothesis: the probability that training reaches 100% accuracy increases with hidden width. Measure the success rate over at least 50 seeds for widths 2, 3, 4, and 8, and report it with confidence intervals ([Phase J §4](09-research-methodology.md#4-theory)).
2. **Vanishing gradients.** In a 10-layer MLP at initialization, compare per-layer gradient norms for sigmoid, tanh, and ReLU.
3. **Initialization.** Zero versus small-random versus Xavier/He initialization; track per-layer activation mean and variance during the first 100 steps.
4. **Overfitting and regularization.** Train on a small subset with varying width and L2 strength; plot training and validation loss curves; identify where validation loss diverges from training loss.
5. **Learning-rate sweep.** Loss after a fixed number of steps against learning rate on a log scale; locate the stable range and the divergence point.

## 7. Resources

**Core**

- Michael Nielsen — *Neural Networks and Deep Learning* (2015), free: <http://neuralnetworksanddeeplearning.com/>. Chapter 2 derives backpropagation carefully.
- Andrej Karpathy — *Neural Networks: Zero to Hero* video series: <https://karpathy.ai/zero-to-hero.html>. The first lecture builds micrograd.
- micrograd source: <https://github.com/karpathy/micrograd>.
- Stanford CS231n course notes (backpropagation, neural networks, optimization): <https://cs231n.github.io/>.
- 3Blue1Brown — neural network series: <https://www.3blue1brown.com/topics/neural-networks>.

**Supplementary**

- Simon Prince — *Understanding Deep Learning* (2023), free: <https://udlbook.github.io/udlbook/>.
- Zhang, Lipton, Li, Smola — *Dive into Deep Learning*, free: <https://d2l.ai/>.
- Goodfellow, Bengio, Courville — *Deep Learning*, Chapters 6–8: <https://www.deeplearningbook.org/>.
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (2nd ed.), free: <https://hastie.su.domains/ElemStatLearn/>.
- Baydin et al. — "Automatic Differentiation in Machine Learning: a Survey" (2018): <https://arxiv.org/abs/1502.05767>.

**Papers**

- Rumelhart, Hinton, Williams — "Learning representations by back-propagating errors", *Nature* (1986): <https://www.nature.com/articles/323533a0>.
- Glorot and Bengio — "Understanding the difficulty of training deep feedforward neural networks" (2010): <https://proceedings.mlr.press/v9/glorot10a.html>.
- He et al. — "Delving Deep into Rectifiers" (2015), He initialization: <https://arxiv.org/abs/1502.01852>.
- Srivastava et al. — "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", *JMLR* (2014): <https://jmlr.org/papers/v15/srivastava14a.html>.
- Zhang et al. — "Understanding deep learning requires rethinking generalization" (2016): <https://arxiv.org/abs/1611.03530>.

## 8. Completion criteria

- [ ] Derive, on paper, the backward pass of a batched two-layer MLP with softmax cross-entropy, including the shape of every gradient.
- [ ] B3: an XOR network reaches 100% accuracy, with the seed success rate documented over at least 20 seeds.
- [ ] Manual gradients match numerical gradients with relative error below $10^{-6}$ in float64.
- [ ] Your B5 autodiff engine produces the same gradients as your manual implementation (B6).
- [ ] Explain why zero initialization fails for hidden layers and demonstrate it.
- [ ] A written overfitting experiment with training/validation curves and an interpretation.

## 9. Research directions

- **Lottery tickets:** do sparse trainable subnetworks exist at initialization? Frankle and Carbin (2018): <https://arxiv.org/abs/1803.03635>.
- **Double descent:** test error can fall again as model size grows past the interpolation threshold. Belkin et al. (2018): <https://arxiv.org/abs/1812.11118>; Nakkiran et al. (2019): <https://arxiv.org/abs/1912.02292>.
- **Loss landscapes:** Li et al., "Visualizing the Loss Landscape of Neural Nets" (2017): <https://arxiv.org/abs/1712.09913>.
- **Infinite-width theory:** Jacot et al., "Neural Tangent Kernel" (2018): <https://arxiv.org/abs/1806.07572>.
- Small-scale question: how does the XOR failure rate depend on initialization scale, and what do the failed solutions look like?

## 10. Common misconceptions

- **"Artificial neurons work like biological neurons."** The analogy is historical and loose; an artificial neuron is a weighted sum followed by a fixed function.
- **"Backpropagation is the learning algorithm."** Backpropagation only *computes gradients*. The optimizer (SGD, Adam) decides how to update parameters.
- **"Universal approximation means a network will learn any function."** The theorems guarantee existence of weights, not that gradient descent will find them, nor with reasonable size or data.
- **"More parameters always means more overfitting."** Modern over-parameterized networks often generalize well; see double descent.
- **"High training accuracy means the model works."** Only held-out evaluation measures generalization, and only on data drawn from the same distribution.
- **"A sigmoid or softmax output is a calibrated confidence."** Outputs are often miscalibrated, especially off-distribution.

---

Previous: [Phase A — Mathematical Foundations](00-foundations.md) · Next: [Phase C — Deep Learning Frameworks](02-deep-learning-frameworks.md) · Up: [Roadmap](README.md)
