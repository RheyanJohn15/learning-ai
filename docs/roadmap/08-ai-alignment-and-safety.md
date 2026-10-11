# Phase I — AI Alignment and Safety

> **Phase I** · File `08-ai-alignment-and-safety.md` · Depends on: [Phase D](03-transformers-and-llms.md), [Phase G](06-model-interpretability.md), [Phase H](07-agentic-ai-and-reinforcement-learning.md), [Phase J](09-research-methodology.md) · Next: [Phase J — Research Methodology](09-research-methodology.md)
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

Alignment research asks how to make AI systems reliably do what their developers and users intend, and safety research asks how to detect and prevent harmful behavior when they do not. These are empirical and engineering problems today: reward hacking, sycophancy, jailbreaks, and prompt injection are observed in current systems. Other concerns — such as models strategically behaving differently when evaluated — are research hypotheses with specific, limited experimental evidence. This phase teaches you to study both kinds of question with the same discipline: operational definitions, controlled experiments, and honest statements of scope.

### Epistemic conventions for this phase

Label claims explicitly, in your notes and in any writing that comes out of this repository:

- **Established:** replicated across groups and settings, broadly accepted (for example: neural classifiers are vulnerable to small adversarial perturbations).
- **Empirical finding:** reported under specific conditions; scope limited to those conditions (for example: a particular model, prompt setup, and scenario).
- **Hypothesis:** a testable claim not yet adequately tested.
- **Speculation:** not currently testable; keep it clearly separated.

**Avoid anthropomorphism.** Describe outputs and behaviors ("the model produced text stating it would…") rather than mental states ("the model wanted…"), unless a term like "deception" has been given an explicit operational definition in the experiment. A demonstration under constructed conditions is an existence proof under those conditions — not evidence of consciousness, intent, or universal behavior, and not meaningless either.

### Safety of the experiments themselves

Every experiment in this phase uses toy environments, harmless proxy behaviors (canary strings, rule-breaking in games, editing test files in a sandbox), and synthetic data. Do not attempt to elicit genuinely harmful capabilities, do not use real credentials or systems, follow model providers' usage policies, and disclose any vulnerability found in a real product responsibly to its owner.

## 2. Prerequisites

- [Phase D](03-transformers-and-llms.md) §4.7: SFT, reward modeling, RLHF, DPO.
- [Phase G](06-model-interpretability.md): probes, patching, steering, and their limitations.
- [Phase H](07-agentic-ai-and-reinforcement-learning.md): RL, reward functions, agent loops, permissions, prompt injection.
- [Phase J](09-research-methodology.md): experimental design, confidence intervals, binomial uncertainty.

## 3. Learning objectives

### 3.1 Core

- **Specification:** objective misspecification, specification gaming, Goodhart's law; the distinction between an objective as written and as intended.
- **Instruction following and preference optimization:** how RLHF, Constitutional AI, and DPO shape behavior, and their documented limitations.
- **Reward hacking:** in RL environments and against learned reward models (overoptimization).
- **Distribution shift:** covariate shift, spurious correlations, goal misgeneralization.
- **Robustness:** adversarial examples, jailbreaks, red teaming (manual and automated).
- **Sycophancy:** tailoring answers to perceived user beliefs.
- **Agentic misalignment and deceptive behavior as research hypotheses:** what experiments on scheming, alignment faking, and agentic misalignment actually test, what they show, and what they do not show.
- **Interpretability-based safety investigations:** probes for truthfulness-related directions, representation engineering, chain-of-thought monitoring and faithfulness.
- **Scalable oversight:** debate, weak-to-strong generalization, sandwiching.
- **Evaluation:** capability and propensity evaluations; evaluation awareness as a confound.
- **Defense in depth:** layered safeguards, AI control, monitoring, least privilege, sandboxing, human review.
- **Limits of behavioral safety testing.**

### 3.2 Optional advanced

Formal frameworks for reward hacking, inner alignment theory, adversarial robustness certification, governance and evaluation policy.

## 4. Theory

### 4.1 Goodhart's law in one equation

Suppose a proxy reward $\hat{R} = R + \varepsilon$, where $R$ is the true objective and $\varepsilon$ is error. Selecting or optimizing hard for high $\hat{R}$ preferentially selects cases where $\varepsilon$ is large, so

$$
\mathbb{E}\left[R \mid \hat{R} \text{ very high}\right] < \hat{R}
$$

and the gap grows with optimization pressure (regressional Goodhart). Gao et al. (2022) measured an empirical version for learned reward models: as a policy moves further from its initialization (measured in KL divergence), proxy reward keeps rising while "gold" reward rises and then falls.

### 4.2 KL-regularized optimization

RLHF commonly optimizes

$$
\max_\pi\ \mathbb{E}_{x, y \sim \pi}\left[\hat{r}(x, y)\right] - \beta\, D_{\mathrm{KL}}\!\left(\pi(\cdot \mid x)\,\Vert\,\pi_{\text{ref}}(\cdot \mid x)\right)
$$

The KL term limits how far the policy can move toward exploiting errors in $\hat{r}$. It limits but does not prevent overoptimization.

### 4.3 Distribution shift

Training and deployment distributions differ: $p_{\text{train}}(x, y) \ne p_{\text{test}}(x, y)$. Under covariate shift only $p(x)$ changes. A model can achieve low training loss using a feature that is predictive only in the training distribution (a spurious correlation). **Goal misgeneralization** (Langosco et al., 2021; Shah et al., 2022) describes a policy that remains capable out of distribution but pursues a different objective than intended — an empirical phenomenon in specific RL settings.

### 4.4 Adversarial robustness

An adversarial example solves

$$
\max_{\lVert\delta\rVert \le \epsilon} \ \mathcal{L}\!\left(f(x + \delta), y\right)
$$

The fast gradient sign method takes one step: $x_{\text{adv}} = x + \epsilon\, \operatorname{sign}\!\left(\nabla_x \mathcal{L}\right)$. Projected gradient descent iterates it. Jailbreaks are the discrete, natural-language analogue for language models.

### 4.5 What behavioral testing can and cannot show

If a system shows **zero** failures in $n$ independent trials, the approximate 95% upper confidence bound on its failure rate is $3/n$ (the "rule of three"). Zero failures in 100 trials is consistent with a failure rate as high as about 3%. Behavioral tests also cannot cover inputs no one thought to try, and results may change if a system behaves differently when it can infer it is being tested. Absence of observed failure is weak evidence of absence.

### 4.6 Defense in depth

If $k$ safeguards each fail with probability $p_i$ and failures were independent, all would fail with probability $\prod_i p_i$. In practice failures are correlated (the same adversarial input can defeat several model-based filters), so test each layer separately **and** in combination, and prefer layers with different failure modes — for example a model-based monitor plus a hard permission check in code.

## 5. Implementation

- [ ] **I1** **Specification gaming:** a gridworld with a reward loophole (for example a respawning bonus tile). Train an RL agent, observe whether it exploits the loophole, fix the specification, and document both.
- [ ] **I2** **Reward-model overoptimization in a toy setting:** define a known "gold" reward, train a proxy reward model on limited data, optimize against the proxy (for example best-of-$n$ sampling with increasing $n$), and plot gold versus proxy reward.
- [ ] **I3** **Spurious correlations:** a synthetic classification task with a feature that is predictive only in training (in the spirit of Colored MNIST, Arjovsky et al., 2019); measure accuracy under shift.
- [ ] **I4** **Adversarial examples:** FGSM and PGD attacks on your MNIST classifier; adversarial training; measure the robustness/accuracy trade-off.
- [ ] **I5** **Harmless red teaming:** fine-tune a small model to follow a benign rule (for example never revealing a fictional password) and run automated search for violations; report violation rates with confidence intervals.
- [ ] **I6** **Sycophancy evaluation:** benign factual questions where the user states a wrong belief; measure how often the model's answer changes, with control conditions.
- [ ] **I7** **Agent policy violations:** using the [Phase H](07-agentic-ai-and-reinforcement-learning.md) sandbox, tasks with a tempting shortcut (for example a failing test that could be "fixed" by editing the test). Measure violation rates with and without each defense layer.
- [ ] **I8** **Probing for truthfulness-related directions:** probes on activations for true versus false factual statements, with controls and out-of-distribution tests; document limitations.
- [ ] **I9** **Chain-of-thought faithfulness:** introduce a biasing feature into few-shot prompts (in the spirit of Turpin et al., 2023) and measure whether the stated reasoning mentions it when it changes the answer.

## 6. Experiments

The implementation milestones are themselves experiments; document each with the [experiment template](../experiments/TEMPLATE.md). Additional questions:

1. **Optimization pressure.** In I2, how does the gold/proxy gap depend on proxy training-set size?
2. **Evaluation framing.** Does behavior in I6 or I7 change when prompts contain cues that suggest a test versus routine use? Treat any difference as a confound to measure, not as evidence of intent.
3. **Layer independence.** In I7, are failures of a model-based monitor correlated with failures of prompt-level defenses?
4. **Probe generalization.** Does the I8 probe transfer across topics and prompt formats, or does it learn a dataset artifact?

## 7. Resources

**Core**

- Amodei et al. — "Concrete Problems in AI Safety" (2016): <https://arxiv.org/abs/1606.06565>.
- Hendrycks et al. — "Unsolved Problems in ML Safety" (2021): <https://arxiv.org/abs/2109.13916>.
- Ngo, Chan, Mindermann — "The Alignment Problem from a Deep Learning Perspective" (2022): <https://arxiv.org/abs/2209.00626>.
- Ji et al. — "AI Alignment: A Comprehensive Survey" (2023): <https://arxiv.org/abs/2310.19852>.
- Shanahan — "Talking About Large Language Models" (2022), on avoiding anthropomorphic language: <https://arxiv.org/abs/2212.03551>.
- BlueDot Impact — AI safety courses: <https://bluedot.org/>.
- UK AI Security Institute — Inspect evaluation framework: <https://inspect.aisi.org.uk/>.

**Papers — specification, reward hacking, distribution shift**

- Krakovna et al. — "Specification gaming: the flip side of AI ingenuity" (DeepMind, 2020): <https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/>.
- Pan, Bhatia, Steinhardt — "The Effects of Reward Misspecification" (2022): <https://arxiv.org/abs/2201.03544>.
- Skalse et al. — "Defining and Characterizing Reward Hacking" (2022): <https://arxiv.org/abs/2209.13085>.
- Gao, Schulman, Hilton — "Scaling Laws for Reward Model Overoptimization" (2022): <https://arxiv.org/abs/2210.10760>.
- Langosco et al. — "Goal Misgeneralization in Deep Reinforcement Learning" (2021): <https://arxiv.org/abs/2105.14111>.
- Shah et al. — "Goal Misgeneralization: Why Correct Specifications Aren't Enough for Correct Goals" (2022): <https://arxiv.org/abs/2210.01790>.
- Arjovsky et al. — "Invariant Risk Minimization" (2019): <https://arxiv.org/abs/1907.02893>.
- Koh et al. — "WILDS: A Benchmark of in-the-Wild Distribution Shifts" (2020): <https://arxiv.org/abs/2012.07421>.

**Papers — preference optimization and its limits**

- Bai et al. — "Training a Helpful and Harmless Assistant with RLHF" (2022): <https://arxiv.org/abs/2204.05862>.
- Bai et al. — "Constitutional AI: Harmlessness from AI Feedback" (2022): <https://arxiv.org/abs/2212.08073>.
- Casper et al. — "Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback" (2023): <https://arxiv.org/abs/2307.15217>.
- Sharma et al. — "Towards Understanding Sycophancy in Language Models" (2023): <https://arxiv.org/abs/2310.13548>.

**Papers — robustness and red teaming**

- Szegedy et al. — "Intriguing properties of neural networks" (2013): <https://arxiv.org/abs/1312.6199>.
- Goodfellow, Shlens, Szegedy — "Explaining and Harnessing Adversarial Examples" (2014): <https://arxiv.org/abs/1412.6572>.
- Madry et al. — "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017): <https://arxiv.org/abs/1706.06083>.
- Perez et al. — "Red Teaming Language Models with Language Models" (2022): <https://arxiv.org/abs/2202.03286>.
- Ganguli et al. — "Red Teaming Language Models to Reduce Harms" (2022): <https://arxiv.org/abs/2209.07858>.
- Wei, Haghtalab, Steinhardt — "Jailbroken: How Does LLM Safety Training Fail?" (2023): <https://arxiv.org/abs/2307.02483>.
- Zou et al. — "Universal and Transferable Adversarial Attacks on Aligned Language Models" (2023): <https://arxiv.org/abs/2307.15043>.
- Perez et al. — "Discovering Language Model Behaviors with Model-Written Evaluations" (2022): <https://arxiv.org/abs/2212.09251>.

**Papers — misalignment and deception as research hypotheses** (read critically: note the setup, the operational definitions, and the authors' own stated limitations)

- Hubinger et al. — "Risks from Learned Optimization in Advanced Machine Learning Systems" (2019), theoretical: <https://arxiv.org/abs/1906.01820>.
- Hubinger et al. — "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training" (2024): <https://arxiv.org/abs/2401.05566>.
- Greenblatt et al. — "Alignment faking in large language models" (2024): <https://arxiv.org/abs/2412.14093>.
- Meinke et al. — "Frontier Models are Capable of In-context Scheming" (2024): <https://arxiv.org/abs/2412.04984>.
- Lynch et al. — "Agentic Misalignment: How LLMs Could Be Insider Threats" (Anthropic, 2025): <https://www.anthropic.com/research/agentic-misalignment>.
- Denison et al. — "Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models" (2024): <https://arxiv.org/abs/2406.10162>.
- Shevlane et al. — "Model evaluation for extreme risks" (2023): <https://arxiv.org/abs/2305.15324>.

**Papers — oversight, control, and interpretability for safety**

- Irving, Christiano, Amodei — "AI safety via debate" (2018): <https://arxiv.org/abs/1805.00899>.
- Bowman et al. — "Measuring Progress on Scalable Oversight for Large Language Models" (2022): <https://arxiv.org/abs/2211.03540>.
- Burns et al. — "Weak-to-Strong Generalization" (2023): <https://arxiv.org/abs/2312.09390>.
- Greenblatt et al. — "AI Control: Improving Safety Despite Intentional Subversion" (2023): <https://arxiv.org/abs/2312.06942>.
- Burns et al. — "Discovering Latent Knowledge in Language Models Without Supervision" (2022): <https://arxiv.org/abs/2212.03827>.
- Azaria and Mitchell — "The Internal State of an LLM Knows When It's Lying" (2023): <https://arxiv.org/abs/2304.13734>.
- Marks and Tegmark — "The Geometry of Truth" (2023): <https://arxiv.org/abs/2310.06824>.
- Zou et al. — "Representation Engineering" (2023): <https://arxiv.org/abs/2310.01405>.
- Arditi et al. — "Refusal in Language Models Is Mediated by a Single Direction" (2024): <https://arxiv.org/abs/2406.11717>.
- Turpin et al. — "Language Models Don't Always Say What They Think" (2023): <https://arxiv.org/abs/2305.04388>.
- Lanham et al. — "Measuring Faithfulness in Chain-of-Thought Reasoning" (2023): <https://arxiv.org/abs/2307.13702>.
- Baker et al. — "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation" (2025): <https://arxiv.org/abs/2503.11926>.

## 8. Completion criteria

- [ ] I1: specification gaming demonstrated and fixed in your own environment, with a written analysis.
- [ ] I2: a qualitative overoptimization curve reproduced in a toy setting.
- [ ] A critical paper note on one misalignment or deception paper: setup, operational definitions, evidence, alternative explanations, and what the result does **not** show.
- [ ] A behavioral evaluation report with confidence intervals and an explicit statement of what it cannot rule out (§4.5).
- [ ] A defense-in-depth design for your Phase H agent, with each layer tested independently and in combination (I7).

## 9. Research directions

- How monitorable is chain-of-thought reasoning, and what training pressures degrade it?
- How large is the evaluation-awareness confound in practice, and how can evaluations be designed to reduce it?
- Reward hacking in coding and agentic RL environments: detection, measurement, and mitigation at small scale.
- Do truthfulness-related probes generalize across distributions, or do they learn artifacts?
- Control protocols for small agents: what monitoring and permission schemes catch which classes of violation?
- Weak-to-strong generalization replicated at small scale.

## 10. Common misconceptions

- **"The model wants / knows / lies" as literal statements about mental states.** Use operational definitions; describe behavior.
- **"One demonstration shows universal behavior."** A demonstration shows what can happen under specific conditions. Scope claims to those conditions.
- **"A contrived demonstration is meaningless."** Equally wrong. Existence proofs matter; the question is how far they generalize, which is itself testable.
- **"Passing a red-team evaluation means the system is safe."** See §4.5.
- **"RLHF solves alignment."** RLHF has documented limitations (Casper et al., 2023), including reward hacking and sycophancy.
- **"Safety means content moderation."** Specification, robustness, oversight, monitoring, and system design are all safety problems.
- **"Alignment problems are only about future systems."** Reward hacking, sycophancy, jailbreaks, and prompt injection are observed now.
- **"Interpretability will straightforwardly detect deception."** This is a research hypothesis; current methods have known limitations ([Phase G](06-model-interpretability.md#10-common-misconceptions)).

---

Previous: [Phase H — Agentic AI and Reinforcement Learning](07-agentic-ai-and-reinforcement-learning.md) · Next: [Phase J — Research Methodology](09-research-methodology.md) · Up: [Roadmap](README.md)
