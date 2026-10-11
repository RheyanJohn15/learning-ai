# Phase D — Transformers and Large Language Models

> **Phase D** · File `03-transformers-and-llms.md` · Depends on: [Phase B](01-neural-networks.md), [Phase C](02-deep-learning-frameworks.md) · Next: [Phase E — Training Architecture and Model Optimization](04-training-and-optimization.md)
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

Large language models are the main object of study in modern interpretability, alignment, and agent research. To study them seriously you need to be able to write one from scratch, explain every tensor in its forward pass, train a small one, and load real pretrained weights into your own code and get identical outputs. This phase builds that capability in stages, from a counting-based character model up to a small GPT-style Transformer.

## 2. Prerequisites

- [Phase B](01-neural-networks.md): backpropagation, softmax cross-entropy, MLPs, initialization.
- [Phase C](02-deep-learning-frameworks.md): PyTorch tensors, autograd, modules, training loops, checkpointing.
- [Phase A](00-foundations.md): cross-entropy as negative log-likelihood; log-sum-exp; floating point.
- Learn just in time: SVD (for LoRA), basic statistics for comparing models.

## 3. Learning objectives

### 3.1 Core

- **Tokenization:** characters versus subwords; byte-pair encoding (BPE) training and encoding; vocabulary size trade-offs; byte-level fallbacks.
- **Embeddings and positional information:** token embedding tables; learned absolute, sinusoidal, and rotary (RoPE) position encodings.
- **Attention:** query, key, and value projections; scaled dot-product attention; causal masking; multi-head attention.
- **The Transformer block:** residual connections, layer normalization (pre-norm versus post-norm), and the position-wise feed-forward network.
- **Decoder-only language models:** next-token prediction, the cross-entropy objective, perplexity.
- **Training and sampling:** greedy decoding, temperature, top-$k$, and nucleus (top-$p$) sampling.
- **Inference mechanics:** prefill versus decode, KV caching, context-window limits, and the computational complexity of attention.

### 3.2 Post-training and efficiency (core concepts, optional implementation)

- Pretraining versus supervised fine-tuning (SFT) versus preference optimization (RLHF, DPO).
- Parameter-efficient fine-tuning (LoRA).
- Quantization (int8/int4, per-tensor versus per-channel) and other inference optimizations (multi-query and grouped-query attention, FlashAttention, paged KV caches).

### 3.3 Optional advanced

Mixture-of-experts layers, state-space models, long-context methods, speculative decoding.

## 4. Theory

### 4.1 Tokens and embeddings

A tokenizer maps text to integer IDs in $\{0, \dots, V - 1\}$. An embedding table $E \in \mathbb{R}^{V \times d}$ maps each ID to a row; this is mathematically a one-hot vector times $E$, implemented as an index lookup. For a batch, token IDs of shape $(B, T)$ become activations $X \in \mathbb{R}^{B \times T \times d}$.

### 4.2 Scaled dot-product attention

For a single head with head dimension $d_h$:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V, \qquad W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}
$$

$$
\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V, \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

Shapes per head: $Q, K, V$ are $(B, T, d_h)$; the score matrix $QK^\top$ is $(B, T, T)$; the output is $(B, T, d_h)$. The causal mask $M$ prevents position $i$ from attending to future positions.

**Why divide by $\sqrt{d_h}$:** if the components of $q$ and $k$ are independent with zero mean and unit variance, $q^\top k$ has variance $d_h$. Without scaling, large logits saturate the softmax and shrink its gradients.

### 4.3 Multi-head attention and the block

With $h$ heads and $d_h = d / h$, heads are computed in parallel (reshape $(B, T, d) \to (B, h, T, d_h)$), concatenated back to $(B, T, d)$, and projected by $W_O \in \mathbb{R}^{d \times d}$.

A pre-norm decoder block:

$$
\begin{aligned}
x &\leftarrow x + \operatorname{MHA}(\operatorname{LN}(x)) \\
x &\leftarrow x + \operatorname{MLP}(\operatorname{LN}(x)), \qquad \operatorname{MLP}(u) = \operatorname{GELU}(uW_1 + b_1)W_2 + b_2,\ W_1 \in \mathbb{R}^{d \times 4d}
\end{aligned}
$$

$$
\operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \quad \text{(statistics over the feature dimension } d\text{)}
$$

The residual stream is the running sum that every block reads from and writes to — a view used heavily in [Phase G](06-model-interpretability.md).

### 4.4 The language-modeling objective

Final activations are projected to logits with an unembedding matrix $W_U \in \mathbb{R}^{d \times V}$ (often tied to $E^\top$). The training loss over a batch is

$$
L = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \log p_\theta\!\left(x_{b, t+1} \mid x_{b, \le t}\right), \qquad \text{perplexity} = e^{L}
$$

Every position is a training example, which is why causal masking makes Transformer training highly parallel.

### 4.5 Parameters, compute, and complexity

Ignoring biases and normalization, each layer has about $4d^2$ attention parameters and $8d^2$ MLP parameters, so

$$
N \approx 12\, n_{\text{layer}}\, d^2 \;+\; V d
$$

Check: GPT-2 small ($n_{\text{layer}} = 12$, $d = 768$, $V = 50257$) gives $\approx 85\text{M} + 38.6\text{M} \approx 124\text{M}$ parameters.

Following Kaplan et al. (2020), a forward pass costs about $2N + 2\,n_{\text{layer}}\,T\,d$ FLOPs per token, and training (forward plus backward) about $6N$ FLOPs per token when the attention term is small. Naive attention costs $O(T^2 d)$ time and materializes a $T \times T$ score matrix per head, i.e. $O(T^2)$ memory.

### 4.6 Inference and the KV cache

During generation, the keys and values of past tokens do not change, so they can be cached. Each new token then needs only one new query against the cache. Per sequence, a standard multi-head-attention cache stores

$$
\text{bytes} = 2 \times n_{\text{layer}} \times T \times d \times \text{bytes per element}
$$

For GPT-2 small at $T = 1024$ in 16-bit precision: $2 \times 12 \times 1024 \times 768 \times 2 \approx 37.7$ MB. **Prefill** (processing the prompt) is a large batched matmul and tends to be compute-bound; **decode** (one token at a time) reads all weights and the cache for little computation and tends to be memory-bandwidth-bound ([Phase F](05-computer-architecture-and-gpu-computing.md)).

### 4.7 Post-training objectives

- **SFT:** the same next-token loss on curated demonstrations, usually computed only on response tokens.
- **Reward modeling (Bradley–Terry):** $P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$, fitted to pairwise human preferences.
- **KL-regularized RL (RLHF):** $\max_\pi \ \mathbb{E}_{y \sim \pi}[r(x, y)] - \beta\, D_{\mathrm{KL}}(\pi \,\Vert\, \pi_{\text{ref}})$.
- **DPO:** optimizes the same objective directly from preferences, without an explicit reward model:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)
$$

### 4.8 Quantization

Affine quantization maps a real value $x$ to an integer $q = \operatorname{round}(x / s) + z$ with scale $s$ and zero-point $z$, and dequantizes as $\hat{x} = s(q - z)$. Per-channel scales track weight ranges more closely than a single per-tensor scale; activation outliers are a known difficulty for LLM quantization (Dettmers et al., 2022).

## 5. Implementation

A staged progression. Each stage must beat the previous stage's validation loss on the same data split.

- [ ] **D1** Character-level bigram model by **counting**; compute its negative log-likelihood on a validation split.
- [ ] **D2** The same bigram model as a neural network (one-hot input times a weight matrix) trained by gradient descent; show it converges to the counting solution's loss.
- [ ] **D3** An MLP language model with embeddings and a fixed context window (after Bengio et al., 2003).
- [ ] **D4** *(optional)* A character-level RNN, to understand recurrence and why it is hard to train and parallelize.
- [ ] **D5** A single causal self-attention head, implemented twice: with explicit loops and vectorized. Confirm identical outputs.
- [ ] **D6** Multi-head attention and a full pre-norm Transformer block.
- [ ] **D7** A small GPT trained on a small character-level corpus (for example Tiny Shakespeare); sampling with temperature.
- [ ] **D8** A byte-level BPE tokenizer from scratch; retrain D7 on BPE tokens.
- [ ] **D9** Decoding strategies: greedy, temperature, top-$k$, top-$p$.
- [ ] **D10** A KV cache; confirm greedy outputs are identical with and without it, and measure the speedup.
- [ ] **D11** Load the public GPT-2 small weights into **your own** implementation and match a reference implementation's logits within a documented tolerance. Weights are downloaded locally and never committed.
- [ ] **D12** *(optional)* SFT on a tiny instruction dataset, DPO on toy preference pairs, LoRA, and post-training int8 quantization of your own model.

## 6. Experiments

1. **Counting versus learning (D1/D2).** Confirm the neural bigram reaches the count-based loss; relate this to maximum likelihood.
2. **Context length.** Validation loss of the D3 MLP model as a function of context length.
3. **Attention scaling.** Remove the $1/\sqrt{d_h}$ factor; measure attention entropy at initialization and training stability across learning rates.
4. **Causal-mask leakage.** Remove the mask: training loss drops dramatically while generation fails. Explain why this is information leakage, not learning.
5. **Pre-norm versus post-norm.** Stability across depths and learning rates, with and without warmup.
6. **Position encodings.** None versus learned versus sinusoidal versus RoPE; evaluate on sequences longer than those seen in training.
7. **Tokenizer comparison.** Vary BPE vocabulary size; compare models using **bits per byte**, not per-token loss, because per-token losses are not comparable across tokenizers.
8. **KV-cache scaling.** Tokens per second and memory against sequence length, with and without the cache.
9. **Sampling.** Temperature versus diversity and repetition metrics.
10. **Quantization.** Validation perplexity of your model at float32, int8, and int4.
11. **Mini scaling study.** Train models of several sizes on the same data; plot validation loss against parameter count on log–log axes.

## 7. Resources

**Core**

- Vaswani et al. — "Attention Is All You Need" (2017): <https://arxiv.org/abs/1706.03762>.
- Karpathy — *Neural Networks: Zero to Hero* (makemore through "Let's build GPT" and the tokenizer lecture): <https://karpathy.ai/zero-to-hero.html>.
- Code: nanoGPT <https://github.com/karpathy/nanoGPT>, makemore <https://github.com/karpathy/makemore>, minbpe <https://github.com/karpathy/minbpe>.
- Rush et al. — "The Annotated Transformer": <https://nlp.seas.harvard.edu/annotated-transformer/>.
- Phuong and Hutter — "Formal Algorithms for Transformers" (2022), precise pseudocode: <https://arxiv.org/abs/2207.09238>.

**Supplementary**

- Jay Alammar — "The Illustrated Transformer": <https://jalammar.github.io/illustrated-transformer/>.
- Jurafsky and Martin — *Speech and Language Processing* (3rd ed. draft), free: <https://web.stanford.edu/~jurafsky/slp3/>.
- Stanford CS224n — Natural Language Processing with Deep Learning: <https://web.stanford.edu/class/cs224n/>.
- Karpathy — llm.c, GPT-2 training in C/CUDA (bridge to [Phase F](05-computer-architecture-and-gpu-computing.md)): <https://github.com/karpathy/llm.c>.

**Papers — architecture and language modeling**

- Bengio et al. — "A Neural Probabilistic Language Model", *JMLR* (2003): <https://www.jmlr.org/papers/v3/bengio03a.html>.
- Bahdanau, Cho, Bengio — "Neural Machine Translation by Jointly Learning to Align and Translate" (2014): <https://arxiv.org/abs/1409.0473>.
- He et al. — "Deep Residual Learning for Image Recognition" (2015): <https://arxiv.org/abs/1512.03385>.
- Ba, Kiros, Hinton — "Layer Normalization" (2016): <https://arxiv.org/abs/1607.06450>.
- Xiong et al. — "On Layer Normalization in the Transformer Architecture" (2020): <https://arxiv.org/abs/2002.04745>.
- Su et al. — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021): <https://arxiv.org/abs/2104.09864>.
- Shazeer — "GLU Variants Improve Transformer" (2020): <https://arxiv.org/abs/2002.05202>.
- Radford et al. — "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019): <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>.
- Brown et al. — "Language Models are Few-Shot Learners" (GPT-3, 2020): <https://arxiv.org/abs/2005.14165>.
- Kaplan et al. — "Scaling Laws for Neural Language Models" (2020): <https://arxiv.org/abs/2001.08361>.
- Hoffmann et al. — "Training Compute-Optimal Large Language Models" (2022): <https://arxiv.org/abs/2203.15556>.

**Papers — tokenization and decoding**

- Sennrich, Haddow, Birch — "Neural Machine Translation of Rare Words with Subword Units" (BPE, 2015): <https://arxiv.org/abs/1508.07909>.
- Kudo and Richardson — "SentencePiece" (2018): <https://arxiv.org/abs/1808.06226>.
- Holtzman et al. — "The Curious Case of Neural Text Degeneration" (nucleus sampling, 2019): <https://arxiv.org/abs/1904.09751>.

**Papers — inference efficiency**

- Shazeer — "Fast Transformer Decoding: One Write-Head is All You Need" (multi-query attention, 2019): <https://arxiv.org/abs/1911.02150>.
- Ainslie et al. — "GQA" (2023): <https://arxiv.org/abs/2305.13245>.
- Pope et al. — "Efficiently Scaling Transformer Inference" (2022): <https://arxiv.org/abs/2211.05102>.
- Dao et al. — "FlashAttention" (2022): <https://arxiv.org/abs/2205.14135>.
- Kwon et al. — "Efficient Memory Management for LLM Serving with PagedAttention" (2023): <https://arxiv.org/abs/2309.06180>.
- Jacob et al. — "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2017): <https://arxiv.org/abs/1712.05877>.
- Dettmers et al. — "LLM.int8()" (2022): <https://arxiv.org/abs/2208.07339>.
- Frantar et al. — "GPTQ" (2022): <https://arxiv.org/abs/2210.17323>.

**Papers — post-training**

- Christiano et al. — "Deep Reinforcement Learning from Human Preferences" (2017): <https://arxiv.org/abs/1706.03741>.
- Ouyang et al. — "Training language models to follow instructions with human feedback" (InstructGPT, 2022): <https://arxiv.org/abs/2203.02155>.
- Rafailov et al. — "Direct Preference Optimization" (2023): <https://arxiv.org/abs/2305.18290>.
- Hu et al. — "LoRA: Low-Rank Adaptation of Large Language Models" (2021): <https://arxiv.org/abs/2106.09685>.
- Dettmers et al. — "QLoRA" (2023): <https://arxiv.org/abs/2305.14314>.

## 8. Completion criteria

- [ ] D5: loop-based and vectorized attention produce matching outputs.
- [ ] For given $B, T, d, h, V$, write down the shape of every tensor in a GPT block's forward pass without running code.
- [ ] Compute your model's parameter count by hand and match the count reported by code.
- [ ] D7: a character-level GPT beats the bigram and MLP baselines on the same validation split, with results recorded.
- [ ] D10: cached and uncached greedy decoding produce identical tokens.
- [ ] D11: your implementation reproduces reference GPT-2 small logits within a documented tolerance.
- [ ] Explain why decoding is usually memory-bandwidth-bound while prefill is usually compute-bound.
- [ ] Write the SFT, RLHF, and DPO objectives and explain how they differ.

## 9. Research directions

- How do Transformers generalize to sequence lengths beyond training, and which position encodings help?
- How does tokenization affect arithmetic, spelling, and multilingual performance?
- Can you reproduce a qualitative scaling-law trend at very small scale, and where does it break?
- Alternatives to attention: state-space models (Gu and Dao, "Mamba", 2023: <https://arxiv.org/abs/2312.00752>); mixture of experts (Shazeer et al., 2017: <https://arxiv.org/abs/1701.06538>; Fedus et al., "Switch Transformers", 2021: <https://arxiv.org/abs/2101.03961>).
- What exactly changes in a model's internals during SFT or DPO? (Connects to [Phase G](06-model-interpretability.md).)

## 10. Common misconceptions

- **"Attention weights tell you what the model is thinking about."** They show one part of the information routing; see [Phase G](06-model-interpretability.md#10-common-misconceptions).
- **"Most of a Transformer is attention."** In standard configurations, MLP layers hold about two thirds of the non-embedding parameters.
- **"Tokens are words."** Tokens are subword units determined by a tokenizer's training data; this has real consequences for spelling and arithmetic.
- **"The context window is memory."** It is the input to a single forward computation; nothing persists between independent calls unless an external system supplies it.
- **"The model looks facts up in a database."** It produces a probability distribution from learned parameters; how factual associations are stored is an active research question.
- **"Temperature 0 makes outputs fully deterministic."** Greedy decoding is deterministic in principle, but floating-point non-associativity and batching can change results in practice.
- **"Post-training mainly teaches new knowledge."** Post-training is often described as primarily shaping behavior; how much it adds versus elicits is an open research question, not an established fact.

---

Previous: [Phase C — Deep Learning Frameworks](02-deep-learning-frameworks.md) · Next: [Phase E — Training Architecture and Model Optimization](04-training-and-optimization.md) · Up: [Roadmap](README.md)
