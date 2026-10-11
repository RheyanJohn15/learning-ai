# Phase J — Research Methodology

> **Phase J** · File `09-research-methodology.md` · Runs **in parallel with all other phases**, starting at Phase B · Previous: [Phase I — AI Alignment and Safety](08-ai-alignment-and-safety.md)
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

The difference between learning established material and doing research is method: asking a question precisely enough to be wrong, designing an experiment that could show you are wrong, measuring uncertainty, and reporting honestly. This phase is numbered last but is **not** done last. It is a set of habits practiced from the first XOR experiment onward, deepening in stages:

| Stage | Starts during | Focus |
|---|---|---|
| J-0 | Phase A | Learning logs; writing down what is unclear |
| J-1 | Phase B | Pre-registered small experiments; first paper notes |
| J-2 | Phase D | Replicating published results at small scale |
| J-3 | Phases G–I | Formulating original questions from gaps and anomalies |
| J-4 | After G–I | Research proposals, technical reports, public reproducible releases |

## 2. Prerequisites

- J-0 and J-1 need nothing beyond the [templates](../README.md#workflow-and-templates).
- Statistics from [Phase A §3.2](00-foundations.md#32-learn-just-in-time): standard error, confidence intervals, the bootstrap — learned when you write your first experiment report.

## 3. Learning objectives

- **Reading papers:** the three-pass method; extracting the claim, the evidence, and the gap between them; critiquing methodology.
- **Hypotheses:** turning a vague question into a falsifiable prediction with an expected direction and rough effect size.
- **Experimental design:** independent, dependent, and controlled variables; confounders; control groups; randomization.
- **Ablations and baselines:** removing one component at a time; simple baselines; equal tuning effort across compared methods.
- **Uncertainty and statistics:** seed variance; confidence intervals; paired comparisons; multiple comparisons; effect sizes versus p-values; the bootstrap.
- **Reproducibility:** recording code version, configuration, seeds, environment, hardware; distinguishing repeatability, reproduction, and replication.
- **Research notebooks and experiment tracking** with minimal tooling.
- **Replication:** reproducing a published result and reporting discrepancies.
- **Finding gaps:** limitations sections, failed replications, anomalies in your own results, small-scale versions of large-scale claims.
- **Writing:** technical reports, research proposals, and reproducible public code.

## 4. Theory

### 4.1 A good hypothesis

A useful hypothesis names the intervention, the measured outcome, the expected direction, and ideally a magnitude, and is written **before** results are seen:

> "Increasing hidden width from 2 to 8 raises the XOR training success rate (100% accuracy within 5,000 steps) from below 70% to above 95%, measured over 100 seeds."

"Wider networks are better" is not testable as stated.

### 4.2 Variables and controls

- **Independent variable:** what you change deliberately.
- **Dependent variable:** what you measure.
- **Controlled variables:** everything held fixed (data split, steps, optimizer, seeds set).
- **Confounder:** something that varies with the independent variable and also affects the outcome (for example, a larger model that also silently got more training steps).

Compare against a **baseline** (the simplest reasonable method) and use **ablations** (remove one component, keep everything else) to attribute effects.

### 4.3 Uncertainty

For $n$ runs with sample mean $\bar{x}$ and sample standard deviation $s$, an approximate 95% confidence interval for the mean is

$$
\bar{x} \pm t_{0.975,\, n-1}\, \frac{s}{\sqrt{n}}
$$

Report whether error bars are standard deviation, standard error, or confidence intervals — they differ by a factor of $\sqrt{n}$ or more.

For a success rate with $k$ successes in $n$ trials, the Wilson score interval behaves better than the naive interval for small $n$ or extreme rates (with $z \approx 1.96$ and $\hat{p} = k/n$):

$$
\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

The **bootstrap** estimates a confidence interval for almost any statistic by resampling your runs with replacement and recomputing the statistic many times.

**Paired comparisons:** when comparing methods A and B, use the same seeds and data for both and analyze the per-seed differences; this removes shared variance.

**Multiple comparisons:** testing $m$ hypotheses at level $\alpha$ makes false positives likely; the Bonferroni correction uses $\alpha / m$ per test. Better still, decide the primary comparison in advance.

**p-values:** the probability of data at least this extreme if the null hypothesis were true. Not the probability the hypothesis is true, and not a measure of effect size.

### 4.4 Levels of reproducibility

1. **Repeatability:** same code, data, and environment give the same result.
2. **Reproduction:** someone re-runs your code and gets your result.
3. **Replication:** an independent implementation of the method gets a consistent result.
4. **Generalization:** the effect holds under changed conditions (data, scale, architecture).

Each experiment record should say which level it reached.

### 4.5 Reading a paper (three passes, after Keshav)

1. **Survey (minutes):** title, abstract, introduction, headings, figures, conclusion. What is the claim?
2. **Comprehension (an hour or more):** the method and evidence; mark unfamiliar references. Does the evidence support the claim?
3. **Reconstruction (hours):** re-derive key equations; imagine re-implementing the method; identify hidden assumptions. What would you test next?

Critique checklist: Are baselines tuned equally? How many seeds, and are error bars reported? Is there possible test-set leakage or contamination? Do ablations isolate the claimed cause? Does the scope of the conclusion match the scope of the experiments? Are the authors' stated limitations complete?

### 4.6 Lightweight experiment tracking

For this repository, an experiment record is a Markdown file using the [experiment template](../experiments/TEMPLATE.md) plus raw results in a small CSV or JSON file, linked from the record. Record the git commit hash, full configuration, seeds, package versions, and hardware. Introduce tools such as experiment trackers only when Markdown and CSV become a measurable bottleneck.

## 5. Implementation

Process milestones rather than code milestones (the J3 toolkit is the exception).

- [ ] **J1** Write your first paper note with the [paper template](../papers/TEMPLATE.md) (suggested: Rumelhart, Hinton, Williams, 1986, alongside [Phase B](01-neural-networks.md)).
- [ ] **J2** Run your first pre-registered experiment: write hypothesis, variables, and analysis plan before running (suggested: [Phase B Experiment 1](01-neural-networks.md#6-experiments), XOR reliability).
- [ ] **J3** Implement a bootstrap confidence interval and the Wilson interval yourself; verify their coverage by simulation (for example, 95% intervals should contain the true value about 95% of the time).
- [ ] **J4** Replicate a published result at small scale (for example induction-head formation, grokking, or an optimizer comparison) and write a replication report including discrepancies.
- [ ] **J5** A literature map: at least 10 papers around one question, organized by claim and evidence, identifying an unaddressed question.
- [ ] **J6** A 1–2 page research proposal: question, motivation, hypothesis, method, expected results, risks, and what would change your mind.
- [ ] **J7** A technical report with a public, reproducible code release (pinned environment, scripts that regenerate every figure, raw results).

## 6. Experiments

Methodology can itself be tested empirically:

1. **How many seeds?** For one of your earlier experiments, estimate the seed-to-seed variance and compute how many seeds are needed to detect an effect of a given size.
2. **Interval coverage.** Simulate the coverage of naive versus Wilson intervals for small $n$ and extreme success rates.
3. **Reproduction from documentation.** Months after an experiment, re-run it from a clean clone using only its written reproduction instructions; record what was missing.

## 7. Resources

**Core**

- S. Keshav — "How to Read a Paper", *ACM SIGCOMM CCR* (2007): <https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf>.
- John Schulman — "An Opinionated Guide to ML Research" (2020): <http://joschu.net/blog/opinionated-guide-ml-research.html>.
- Lipton and Steinhardt — "Troubling Trends in Machine Learning Scholarship" (2018): <https://arxiv.org/abs/1807.03341>.
- Pineau et al. — "Improving Reproducibility in Machine Learning Research" (2020): <https://arxiv.org/abs/2003.12206>.
- Richard Hamming — "You and Your Research" (1986 talk transcript): <https://www.cs.virginia.edu/~robins/YouAndYourResearch.html>.

**Statistics and evaluation**

- Wasserman — *All of Statistics* (2004).
- Efron and Tibshirani — *An Introduction to the Bootstrap* (1993).
- Dror et al. — "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing" (2018): <https://aclanthology.org/P18-1128/>.
- Dodge et al. — "Show Your Work: Improved Reporting of Experimental Results" (2019): <https://arxiv.org/abs/1909.03004>.
- Bouthillier et al. — "Accounting for Variance in Machine Learning Benchmarks" (2021): <https://arxiv.org/abs/2103.03098>.
- Agarwal et al. — "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (2021): <https://arxiv.org/abs/2108.13264>.
- Melis, Dyer, Blunsom — "On the State of the Art of Evaluation in Neural Language Models" (2017), on untuned baselines: <https://arxiv.org/abs/1707.05589>.
- Kapoor and Narayanan — "Leakage and the Reproducibility Crisis in ML-based Science" (2022): <https://arxiv.org/abs/2207.07048>.
- Ioannidis — "Why Most Published Research Findings Are False", *PLOS Medicine* (2005): <https://doi.org/10.1371/journal.pmed.0020124>.
- Gelman and Loken — "The garden of forking paths" (2013): <http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf>.

**Research culture and communication**

- Chris Olah and Shan Carter — "Research Debt", *Distill* (2017): <https://distill.pub/2017/research-debt/>.
- Sculley et al. — "Winner's Curse? On Pace, Progress, and Empirical Rigor" (2018): <https://openreview.net/forum?id=rJWF0Fywf>.

## 8. Completion criteria

- [ ] At least 10 paper notes across phases, each with a critical-analysis section; at least 3 lead to a follow-up experiment idea in the [research backlog](../research-backlog.md).
- [ ] J2: an experiment whose hypothesis and analysis plan were committed before its results.
- [ ] J3: interval implementations with simulated coverage close to nominal.
- [ ] J4: a replication report, including what did not replicate and possible reasons.
- [ ] Every experiment record in the repository states a hypothesis, reports uncertainty, and states its reproducibility level.
- [ ] Experiment 3: one experiment reproduced from a clean clone using only its documentation.

## 9. Research directions

This phase exists to generate research directions elsewhere. Reliable sources of good questions:

- **Limitations sections** of papers you have noted.
- **Failed or partial replications**, including your own.
- **Anomalies** in your experiments that your hypothesis did not predict.
- **Small-scale tests of large-scale claims**: does an effect reported at scale appear in models you can train?
- **Methodological gaps**: claims supported only at evidence level 1–2 ([Phase G §4.6](06-model-interpretability.md#46-observation-versus-intervention)).

Record them in the [research backlog](../research-backlog.md).

## 10. Common misconceptions

- **"p < 0.05 means the result is true and important."** It is neither a probability of truth nor a measure of importance.
- **"One seed is enough if the curve looks clear."** Seed variance in deep learning and RL is frequently as large as the effects being claimed.
- **"Beating the state of the art is research."** Understanding why something works, or showing that it does not, is often more valuable.
- **"The baseline is just there for comparison."** Under-tuned baselines are among the most common sources of spurious improvements.
- **"Negative results are worthless."** A well-designed negative result constrains hypotheses and saves others time.
- **"Re-running the authors' code is replication."** That is reproduction. Replication needs an independent implementation.

---

Previous: [Phase I — AI Alignment and Safety](08-ai-alignment-and-safety.md) · Up: [Roadmap](README.md)
