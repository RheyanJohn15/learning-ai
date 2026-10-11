# Phase G — Interpretability and Internal Model Mechanisms

> **Phase G** · File `06-model-interpretability.md` · Depends on: [Phase D](03-transformers-and-llms.md) (ideally through D11), [Phase J](09-research-methodology.md) basics · Next: [Phase H — Agentic AI and Reinforcement Learning](07-agentic-ai-and-reinforcement-learning.md)
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

Behavioral testing tells you *what* a model does on the inputs you tried. Interpretability asks *how* it does it — which internal computations produce the behavior — and whether that understanding predicts behavior on inputs you did not try. This matters for science (how do models learn representations and capabilities?) and for safety (can we detect problems that behavioral tests miss?). It is also a field where careful small-scale work by individuals has produced influential results, which makes it a good entry point for original research.

## 2. Prerequisites

- [Phase D](03-transformers-and-llms.md): your own GPT implementation; ideally D11 (GPT-2 weights in your own code), so you know exactly where every activation comes from.
- [Phase A](00-foundations.md): linear algebra including SVD and projections (learn just in time); probability.
- [Phase J](09-research-methodology.md): hypotheses, controls, confidence intervals. Interpretability claims are easy to overstate without them.

## 3. Learning objectives

### 3.1 Core

- **Activations and representations:** the residual stream as a shared communication channel; reading and writing activations with hooks.
- **Features as directions:** the linear representation hypothesis — treated as a hypothesis with supporting and conflicting evidence.
- **Superposition and polysemanticity:** why individual neurons often respond to many unrelated inputs.
- **Attention heads:** QK (where to attend) and OV (what to move) circuits; induction heads; why attention patterns alone are limited as explanations.
- **Linear probes:** training, control tasks, and why decodability does not imply use.
- **Logit lens and tuned lens:** reading intermediate predictions.
- **Feature visualization and dataset examples** (originating in vision models).
- **Dictionary learning with sparse autoencoders (SAEs).**
- **Mechanistic interpretability:** circuits as subgraphs implementing a behavior.
- **Causal interventions:** ablations (zero, mean, resample), activation patching, path patching, attribution patching, steering vectors.
- **Causal inference and controls:** observation versus intervention; faithfulness, completeness, and minimality of an explanation.
- **Limitations:** interpretability illusions, cherry-picking, dependence on prompt distributions, scale.

### 3.2 Optional advanced

Automated circuit discovery, attribution graphs and transcoders, causal abstraction theory, training-dynamics interpretability (grokking, phase transitions).

## 4. Theory

### 4.1 The residual stream decomposition

Because each block *adds* to the residual stream, the final residual is a sum of contributions:

$$
x_{\text{final}} = x_{\text{embed}} + \sum_{\ell=1}^{n_{\text{layer}}} \left(\operatorname{attn}_\ell + \operatorname{mlp}_\ell\right), \qquad \operatorname{attn}_\ell = \sum_{h} \operatorname{head}_{\ell,h}
$$

**Direct logit attribution:** the direct contribution of component $c$ to the logit of token $t$ is approximately $c^\top W_U[:, t]$, after accounting for the final layer norm (often by freezing its scale). This measures only the *direct* path; components also act indirectly through later layers.

### 4.2 QK and OV circuits

For one head, the attention score between positions depends on the bilinear form $x_i^\top W_Q W_K^\top x_j$ (the QK circuit), and what gets written to the residual stream depends on $W_V W_O$ (the OV circuit). Analyzing these low-rank matrices directly can reveal what a head does independent of any particular input (Elhage et al., 2021).

### 4.3 Probes and control tasks

A linear probe $\hat{y} = \sigma(w^\top h + b)$ tests whether a property $y$ is linearly decodable from activations $h$. Hewitt and Liang (2019) define **selectivity** as probe accuracy on the real task minus accuracy on a control task with random labels for the same inputs. High accuracy alone may reflect the probe's capacity rather than the model's representation. Even a selective probe shows only that information is *present*, not that the model *uses* it; that requires an intervention.

### 4.4 Sparse autoencoders

$$
f = \operatorname{ReLU}\!\left(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}}\right), \qquad \hat{x} = W_{\text{dec}} f + b_{\text{dec}}
$$

$$
\mathcal{L} = \lVert x - \hat{x} \rVert_2^2 + \lambda \lVert f \rVert_1
$$

The dictionary is typically much larger than the activation dimension. Learned features are candidates for interpretable directions; their faithfulness is assessed by reconstruction error and, more importantly, by the change in model loss when activations are replaced with reconstructions.

### 4.5 Activation patching

Run the model on a **clean** input (correct behavior) and a **corrupted** input (minimally different, behavior changes). Replace an activation at one site in the corrupted run with its value from the clean run, and measure how much of the clean behavior is restored with a metric $m$ (for example the logit difference between correct and incorrect answers):

$$
\text{fraction recovered} = \frac{m_{\text{patched}} - m_{\text{corrupted}}}{m_{\text{clean}} - m_{\text{corrupted}}}
$$

Results depend on the choice of corruption, metric, and patching direction (clean→corrupted versus corrupted→clean); these must be reported (Heimersheim and Nanda, 2024; Zhang and Nanda, 2023).

### 4.6 Observation versus intervention

Correlational evidence measures $P(Y \mid X)$; interventions estimate $P(Y \mid \operatorname{do}(X))$. Activations that correlate with a behavior may be downstream effects or redundant copies. Use this **evidence ladder** to classify claims, your own and in papers:

1. **Behavioral observation:** the model produces output Y on inputs X. Says nothing about mechanism.
2. **Correlational internal evidence:** a probe decodes a property; a neuron activates on certain inputs.
3. **Causal evidence:** ablating or patching a component changes behavior in a predicted direction.
4. **Mechanistic explanation:** a described algorithm that predicts behavior on held-out inputs and survives targeted tests, with known coverage of the behavior.

Most published claims sit at level 2 or 3. Say which level a claim reaches.

### 4.7 Superposition

When a model must represent more sparse features than it has dimensions, it can store them as nearly orthogonal directions, accepting some interference. Elhage et al. (2022) showed this in toy models; whether and how it applies in large models is an active research question.

## 5. Implementation

- [ ] **G1** Add activation hooks and caching to **your own** GPT; then reproduce the same activations on GPT-2 small with TransformerLens and confirm they match.
- [ ] **G2** Logit lens on GPT-2 small: decode the residual stream at every layer.
- [ ] **G3** A linear probe for a simple property, with a control task, reporting selectivity.
- [ ] **G4** Train a small attention-only Transformer (1–2 layers) on sequences with repeated random tokens; identify induction heads by attention pattern **and** confirm by ablation.
- [ ] **G5** Activation patching on indirect-object-identification-style prompts in GPT-2 small; produce a layer × position heatmap with documented clean/corrupted prompts and metric.
- [ ] **G6** Replicate a small toy-model-of-superposition result.
- [ ] **G7** Train a sparse autoencoder on activations of a small model; report reconstruction error, sparsity, the change in model loss when substituting reconstructions, and examples of features.
- [ ] **G8** *(optional)* Replicate grokking on modular addition and the Fourier-based analysis of Nanda et al. (2023).

## 6. Experiments

1. **Decodability versus use.** Train a probe; then remove the probed direction (for example by projecting it out, in the spirit of amnesic probing, Elazar et al., 2021) and measure whether behavior changes.
2. **Ablation choice.** Zero versus mean versus resample ablation of the same component; document where conclusions differ.
3. **Attention pattern versus causal importance.** Compare attention-pattern salience of heads with their patching effects on the same task.
4. **SAE dictionary size.** Reconstruction, sparsity, and qualitative interpretability as dictionary size grows.
5. **Steering.** Effect size of adding a steering vector as a function of layer and scale, with a random-direction control.
6. **Universality across seeds.** Train the G4 model with several seeds; do the same circuits appear?

## 7. Resources

**Core**

- Elhage et al. — "A Mathematical Framework for Transformer Circuits" (2021): <https://transformer-circuits.pub/2021/framework/index.html>.
- Olsson et al. — "In-context Learning and Induction Heads" (2022): <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>.
- ARENA curriculum (hands-on interpretability exercises): <https://www.arena.education/>, materials at <https://github.com/callummcdougall/ARENA_3.0>.
- TransformerLens library: <https://github.com/TransformerLensOrg/TransformerLens>.
- Neel Nanda — "Comprehensive Mechanistic Interpretability Explainer & Glossary": <https://www.neelnanda.io/mechanistic-interpretability/glossary>.
- Sharkey et al. — "Open Problems in Mechanistic Interpretability" (2025): <https://arxiv.org/abs/2501.16496>.

**Papers — features and superposition**

- Olah et al. — "Feature Visualization", *Distill* (2017): <https://distill.pub/2017/feature-visualization/>.
- Olah et al. — "Zoom In: An Introduction to Circuits", *Distill* (2020): <https://distill.pub/2020/circuits/zoom-in/>.
- Elhage et al. — "Toy Models of Superposition" (2022): <https://transformer-circuits.pub/2022/toy_model/index.html>.
- Bricken et al. — "Towards Monosemanticity" (2023): <https://transformer-circuits.pub/2023/monosemantic-features/index.html>.
- Cunningham et al. — "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (2023): <https://arxiv.org/abs/2309.08600>.
- Templeton et al. — "Scaling Monosemanticity" (2024): <https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>.
- Park, Choe, Veitch — "The Linear Representation Hypothesis and the Geometry of Large Language Models" (2023): <https://arxiv.org/abs/2311.03658>.
- Mikolov et al. — "Efficient Estimation of Word Representations in Vector Space" (2013): <https://arxiv.org/abs/1301.3781>.

**Papers — circuits and causal methods**

- Meng et al. — "Locating and Editing Factual Associations in GPT" (2022): <https://arxiv.org/abs/2202.05262>.
- Wang et al. — "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" (2022): <https://arxiv.org/abs/2211.00593>.
- Nanda et al. — "Progress measures for grokking via mechanistic interpretability" (2023): <https://arxiv.org/abs/2301.05217>.
- Power et al. — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (2022): <https://arxiv.org/abs/2201.02177>.
- Conmy et al. — "Towards Automated Circuit Discovery for Mechanistic Interpretability" (2023): <https://arxiv.org/abs/2304.14997>.
- Marks et al. — "Sparse Feature Circuits" (2024): <https://arxiv.org/abs/2403.19647>.
- Zhang and Nanda — "Towards Best Practices of Activation Patching in Language Models" (2023): <https://arxiv.org/abs/2309.16042>.
- Heimersheim and Nanda — "How to use and interpret activation patching" (2024): <https://arxiv.org/abs/2404.15255>.
- Geiger et al. — "Causal Abstractions of Neural Networks" (2021): <https://arxiv.org/abs/2106.02997>.
- Turner et al. — "Steering Language Models With Activation Engineering" (2023): <https://arxiv.org/abs/2308.10248>.
- nostalgebraist — "interpreting GPT: the logit lens" (2020): <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>; Belrose et al. — "Eliciting Latent Predictions from Transformers with the Tuned Lens" (2023): <https://arxiv.org/abs/2303.08112>.

**Papers — probing and limitations**

- Alain and Bengio — "Understanding intermediate layers using linear classifier probes" (2016): <https://arxiv.org/abs/1610.01644>.
- Hewitt and Liang — "Designing and Interpreting Probes with Control Tasks" (2019): <https://arxiv.org/abs/1909.03368>.
- Belinkov — "Probing Classifiers: Promises, Shortcomings, and Advances" (2021): <https://arxiv.org/abs/2102.12452>.
- Elazar et al. — "Amnesic Probing" (2021): <https://arxiv.org/abs/2006.00995>.
- Jain and Wallace — "Attention is not Explanation" (2019): <https://arxiv.org/abs/1902.10186>; Wiegreffe and Pinter — "Attention is not not Explanation" (2019): <https://arxiv.org/abs/1908.04626>.
- Adebayo et al. — "Sanity Checks for Saliency Maps" (2018): <https://arxiv.org/abs/1810.03292>.
- Bolukbasi et al. — "An Interpretability Illusion for BERT" (2021): <https://arxiv.org/abs/2104.07143>.
- Makelov, Lange, Nanda — "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching" (2023): <https://arxiv.org/abs/2311.17030>.

## 8. Completion criteria

- [ ] G4: induction heads identified in your own toy model, with causal evidence (ablation increases loss on repeated sequences) rather than attention patterns alone.
- [ ] G5: a patching experiment documented with clean/corrupted prompts, metric, controls, and stated limitations.
- [ ] G3: a probe reported with a control task; explain in writing why accuracy alone is insufficient.
- [ ] Explain superposition and polysemanticity using your G6 results.
- [ ] A paper note that classifies each major claim of an interpretability paper on the evidence ladder (§4.6).

## 9. Research directions

- How should sparse-autoencoder features be evaluated, and do they reflect the model's computation or the SAE's inductive biases?
- Universality: do models trained with different seeds or architectures learn the same features and circuits?
- Training dynamics: when and how do circuits such as induction heads form?
- Chain-of-thought faithfulness: does a model's stated reasoning reflect the computation that produced its answer? (Connects to [Phase I](08-ai-alignment-and-safety.md).)
- Can interpretability methods detect behaviors that behavioral evaluations miss? This is a motivating hypothesis for safety work, not an established capability.

## 10. Common misconceptions

- **"Attention patterns show what the model is thinking about."** They show where information is read from in one sublayer, not what is done with it, and they omit MLP computation.
- **"One neuron represents one concept."** Polysemanticity is common; features are often better described as directions spread over many neurons.
- **"A high-accuracy probe proves the model uses that information."** It shows decodability. Use requires interventions.
- **"A compelling visualization is an explanation."** Without causal tests and held-out predictions, it is a hypothesis.
- **"A circuit found on one prompt template explains the behavior in general."** Circuits can be specific to the prompt distribution used to find them.
- **"Zero ablation is a neutral intervention."** Zeroing can push activations off-distribution; mean or resample ablation are often more appropriate, and the choice changes conclusions.
- **"SAE features are the model's true features."** They are one decomposition among many; their faithfulness must be measured.

---

Previous: [Phase F — Computer Architecture and GPU Execution](05-computer-architecture-and-gpu-computing.md) · Next: [Phase H — Agentic AI and Reinforcement Learning](07-agentic-ai-and-reinforcement-learning.md) · Up: [Roadmap](README.md)
