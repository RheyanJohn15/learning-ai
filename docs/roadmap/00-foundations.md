# Phase A — Mathematical Foundations

> **Phase A** · File `00-foundations.md` · Depends on: nothing · Next: [Phase B — Neural Networks from Scratch](01-neural-networks.md)
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

Every later phase is, underneath, three things executed in finite-precision arithmetic:

- **Linear algebra** describes what a model computes (matrix multiplications over tensors).
- **Calculus** describes how a model learns (gradients via the chain rule).
- **Probability and information theory** describe what a model is trained to do (assign high likelihood to data, minimize cross-entropy).

Floating-point arithmetic and optimization explain why training sometimes fails in ways the clean mathematics does not predict. Research-level work — reading papers, debugging training, interpreting internals — requires reading equations fluently and checking them numerically.

This phase is **not** "learn all of mathematics first." It separates what is essential before Phase B from what can be learned just in time.

## 2. Prerequisites

- High-school algebra: functions, exponents, logarithms, summation notation.
- Basic Python (you already have this as a software engineer).
- Optional but useful: NumPy basics — see [Phase C §3](02-deep-learning-frameworks.md#3-learning-objectives); the NumPy portion can be started in parallel with this phase.

**Self-assessment:** before starting, try to answer: What is the derivative of $\log(1 + e^{x})$? What shape is the product of a $3 \times 4$ and a $4 \times 2$ matrix? What is $P(A \mid B)$ in terms of $P(B \mid A)$? Whatever you cannot answer tells you where to begin.

## 3. Learning objectives

### 3.1 Essential before Phase B

| Area | You should be able to… |
|---|---|
| Linear algebra | Compute dot products, norms, matrix–vector and matrix–matrix products by hand; reason about shapes; explain a matrix as a linear map; transpose and identity; explain what an inverse is and when it exists. |
| Tensors | Treat an ML "tensor" as an $n$-dimensional array with a shape; index, reshape, and broadcast mentally. |
| Calculus | Differentiate common functions ($e^x$, $\log x$, polynomials, sigmoid); explain the derivative as a local linear approximation. |
| Multivariable calculus | Compute partial derivatives and gradients; apply the chain rule through compositions of vector functions; explain what a Jacobian is. |
| Probability | Random variables, expectation, variance, conditional probability, Bayes' rule, independence; Bernoulli, categorical, and Gaussian distributions. |
| Likelihood | Distinguish probability (data varies, parameters fixed) from likelihood (data fixed, parameters vary); derive a maximum-likelihood estimate for a Bernoulli parameter. |
| Information theory | Define entropy, cross-entropy, and KL divergence; explain why minimizing cross-entropy equals maximizing likelihood. |
| Numerics | Explain float32 layout, machine epsilon, overflow, underflow, and catastrophic cancellation; apply the log-sum-exp trick. |
| Optimization | Run gradient descent on a quadratic by hand; explain how the learning rate affects convergence and divergence. |

### 3.2 Learn just in time

| Topic | Needed by |
|---|---|
| Matrix calculus (Jacobians of matrix functions, vector–Jacobian products) | [Phase B](01-neural-networks.md) backpropagation |
| Taylor expansion to second order, the Hessian | [Phase E](04-training-and-optimization.md) optimizer behavior |
| Eigendecomposition and SVD | [Phase D](03-transformers-and-llms.md) (LoRA, low rank), [Phase G](06-model-interpretability.md) (PCA, subspaces) |
| Statistics: estimators, standard error, confidence intervals, bootstrap | First experiment report ([Phase J](09-research-methodology.md)) |
| Integral calculus: expectations over continuous variables, normalizing constants | Continuous distributions, Gaussian likelihoods |
| Convexity, conditioning, Lagrange multipliers | [Phase E](04-training-and-optimization.md), [Phase I](08-ai-alignment-and-safety.md) (constrained objectives) |
| Mutual information | [Phase G](06-model-interpretability.md) |
| Markov chains | [Phase H](07-agentic-ai-and-reinforcement-learning.md) |

### 3.3 Optional advanced

Measure-theoretic probability, functional analysis, information geometry, random matrix theory, and differential geometry. These appear in theoretical papers but are not needed for implementation or most empirical research.

## 4. Theory

### 4.1 Matrix multiplication and shapes

For $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$:

$$
C = AB \in \mathbb{R}^{m \times n}, \qquad C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}
$$

The inner dimensions must match. The cost is $mn$ dot products of length $k$, i.e. about $2mnk$ floating-point operations (one multiply and one add per term). This single count underlies most compute estimates in [Phase E](04-training-and-optimization.md) and [Phase F](05-computer-architecture-and-gpu-computing.md).

A batched matrix multiply over shapes $(B, m, k) \times (B, k, n) \to (B, m, n)$ is $B$ independent matrix multiplies. Most neural-network layers are batched matrix multiplies plus elementwise functions.

### 4.2 Derivatives, gradients, and the chain rule

The derivative is the best local linear approximation:

$$
f(x + \delta) \approx f(x) + \nabla f(x)^\top \delta
$$

For a scalar loss $L$ depending on $y = f(x)$ with $x \in \mathbb{R}^n$, $y \in \mathbb{R}^m$, and Jacobian $J_f(x) \in \mathbb{R}^{m \times n}$:

$$
\nabla_x L = J_f(x)^\top \, \nabla_y L
$$

This vector–Jacobian product is the entire mechanism of backpropagation: gradients flow backwards, one Jacobian-transpose at a time. Note the shapes: $\nabla_x L$ always has the same shape as $x$.

Worked example to do by hand: for $f(x) = \lVert Ax - b \rVert^2$, show that $\nabla_x f = 2A^\top(Ax - b)$.

### 4.3 Probability and likelihood

Given i.i.d. data $x_1, \dots, x_N$ and a model $p_\theta$, the maximum-likelihood estimate is

$$
\hat{\theta} = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(x_i)
$$

Logs turn products into sums (numerically stable, and gradients decompose per example). For a Bernoulli model with $k$ successes in $N$ trials, $\hat{\theta} = k/N$; derive this by setting the derivative of the log-likelihood to zero.

### 4.4 Entropy, cross-entropy, and KL divergence

For discrete distributions $p$ (data) and $q$ (model):

$$
H(p) = -\sum_x p(x)\log p(x), \qquad H(p, q) = -\sum_x p(x) \log q(x)
$$

$$
H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q), \qquad D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \ge 0
$$

Because $H(p)$ does not depend on the model, minimizing cross-entropy minimizes KL divergence to the data distribution. With the empirical distribution of a dataset, cross-entropy is exactly the average negative log-likelihood. This is the loss every language model in [Phase D](03-transformers-and-llms.md) minimizes.

With natural logarithms, units are nats; with $\log_2$, bits.

### 4.5 Floating point and numerical stability

IEEE 754 float32 uses 1 sign bit, 8 exponent bits, and 23 fraction bits. Machine epsilon (the gap between 1 and the next representable number) is $2^{-23} \approx 1.19 \times 10^{-7}$. Consequences:

- Addition is **not associative**: $(a + b) + c \ne a + (b + c)$ in general. Parallel reductions in a different order give different results.
- Subtracting nearly equal numbers loses significant digits (catastrophic cancellation).
- $e^{x}$ overflows float32 for $x \gtrsim 88.7$.

The log-sum-exp trick computes $\log \sum_i e^{z_i}$ safely with $m = \max_i z_i$:

$$
\log \sum_i e^{z_i} = m + \log \sum_i e^{z_i - m}
$$

The same idea gives a stable softmax. Reduced-precision formats (float16, bfloat16, float8) trade range and precision differently; see [Phase F](05-computer-architecture-and-gpu-computing.md#4-theory).

### 4.6 Gradient descent

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)
$$

On a quadratic $L(\theta) = \tfrac{1}{2}\theta^\top A \theta$ with symmetric positive-definite $A$, gradient descent converges if and only if $0 < \eta < 2 / \lambda_{\max}(A)$. The convergence rate depends on the condition number $\kappa = \lambda_{\max} / \lambda_{\min}$: ill-conditioned problems zig-zag. This simple model explains a surprising amount of practical training behavior and motivates momentum and adaptive optimizers in [Phase E](04-training-and-optimization.md).

## 5. Implementation

Use plain Python and NumPy only. No machine learning libraries.

- [ ] **A1** Matrix multiplication with explicit loops; verify against `numpy.matmul` on random inputs; time both and explain the gap (preview of [Phase F](05-computer-architecture-and-gpu-computing.md)).
- [ ] **A2** A numerical gradient checker using central differences, $\frac{f(x + h e_i) - f(x - h e_i)}{2h}$. Validate it on functions with known gradients, including $\lVert Ax - b \rVert^2$. You will reuse this checker in every later phase.
- [ ] **A3** Naive and stable implementations of softmax, log-softmax, and log-sum-exp; demonstrate the naive version overflowing.
- [ ] **A4** Gradient descent on 2-D quadratics with different condition numbers; plot trajectories for several learning rates.
- [ ] **A5** Entropy, cross-entropy, and KL functions; verify $H(p, q) = H(p) + D_{\mathrm{KL}}(p \Vert q)$ numerically; fit a Bernoulli and a Gaussian by gradient ascent on log-likelihood and compare with closed-form MLEs.

## 6. Experiments

Document each with the [experiment template](../experiments/TEMPLATE.md).

1. **Finite-difference step size.** Hypothesis: for central differences, error is minimized near $h \approx \epsilon^{1/3}$ because truncation error ($\propto h^2$) trades off against round-off error ($\propto \epsilon / h$). Measure error versus $h$ in float32 and float64.
2. **Summation error.** Sum $10^7$ float32 values naively, pairwise, and with Kahan summation; measure error against a float64 reference.
3. **Learning-rate threshold.** Predict the divergence threshold $2/\lambda_{\max}$ for several quadratics, then measure it. Measure iterations-to-convergence as a function of the condition number.
4. **Monte Carlo error.** Estimate an expectation by sampling; confirm the error shrinks roughly as $1/\sqrt{N}$.

## 7. Resources

**Core**

- Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* (2020), free: <https://mml-book.github.io/>. Chapters 2–7 map closely onto this phase.
- 3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus*: <https://www.3blue1brown.com/topics/linear-algebra>, <https://www.3blue1brown.com/topics/calculus>. Geometric intuition before formalism.
- Gilbert Strang — MIT 18.06 Linear Algebra (OCW): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/>.
- Parr and Howard — "The Matrix Calculus You Need for Deep Learning" (2018): <https://arxiv.org/abs/1802.01528>.
- Goldberg — "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991): <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>.

**Supplementary**

- Goodfellow, Bengio, Courville — *Deep Learning* (2016), Part I (Chapters 2–4): <https://www.deeplearningbook.org/>.
- Blitzstein and Hwang — *Introduction to Probability* (Harvard Stat 110 textbook).
- MacKay — *Information Theory, Inference, and Learning Algorithms* (2003), free: <http://www.inference.org.uk/mackay/itila/>.
- Cover and Thomas — *Elements of Information Theory* (2nd ed., 2006).
- Boyd and Vandenberghe — *Convex Optimization* (2004), free: <https://web.stanford.edu/~boyd/cvxbook/>.
- Petersen and Pedersen — *The Matrix Cookbook* (reference for identities): <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>.
- Higham — *Accuracy and Stability of Numerical Algorithms* (2nd ed., 2002), for deep numerical analysis.

## 8. Completion criteria

Tick a box only with evidence (a link to a learning-log entry, experiment, or code). See [progress conventions](../README.md#progress-conventions).

- [ ] Derive $\nabla_x \lVert Ax - b\rVert^2$ by hand and confirm it with the A2 gradient checker.
- [ ] Given input and weight shapes for a batched matrix multiply, state the output shape and FLOP count without running code.
- [ ] Explain, in your own words, why minimizing cross-entropy is maximum-likelihood estimation.
- [ ] Demonstrate naive softmax overflow and the stable fix (A3).
- [ ] Show an empirical gradient-descent divergence threshold that matches $2/\lambda_{\max}$ (Experiment 3).
- [ ] Explain why `0.1 + 0.2 != 0.3` in floating point and why parallel sums can be non-deterministic.

## 9. Research directions

Questions to return to after later phases:

- How low can numerical precision go during training before learning breaks down, and why? (Connects to [Phase E](04-training-and-optimization.md) and [Phase F](05-computer-architecture-and-gpu-computing.md).)
- Training often operates where the sharpness (top Hessian eigenvalue) hovers near $2/\eta$ — the "edge of stability" (Cohen et al., 2021, <https://arxiv.org/abs/2103.00065>). What does this mean for the quadratic intuition of §4.6?
- Information-theoretic views of learning: is compression a useful lens on generalization?

## 10. Common misconceptions

- **"I must master all the mathematics before touching code."** Learn the essentials, then learn the rest when a concrete implementation demands it.
- **"An ML tensor is the tensor of physics."** In ML it usually just means a multi-dimensional array with a shape; the multilinear-algebra meaning is rarely needed.
- **"The gradient points toward the minimum."** It points in the direction of steepest local *ascent*. In ill-conditioned problems the negative gradient can be nearly orthogonal to the direction of the minimum.
- **"Probability and likelihood are the same thing."** Same formula, different variable held fixed. Likelihood is not a probability distribution over parameters.
- **"Floating-point arithmetic behaves like real arithmetic."** It is not associative, it has limited range and precision, and results can depend on the order of operations.
- **"KL divergence is a distance."** It is asymmetric and does not satisfy the triangle inequality; $D_{\mathrm{KL}}(p\Vert q)$ and $D_{\mathrm{KL}}(q\Vert p)$ lead to different behavior when optimized.

---

Next: [Phase B — Neural Networks from Scratch](01-neural-networks.md) · Up: [Roadmap](README.md)
