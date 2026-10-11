# EXP-NNN — Short Title

- **Status:** Planned / Running / Completed / Abandoned
- **Phase / milestone:** e.g. [Phase B](../roadmap/01-neural-networks.md), Experiment 1
- **Pre-registered:** YYYY-MM-DD (commit hash of the version written before running)
- **Completed:** YYYY-MM-DD
- **Reproducibility level reached:** repeatability / reproduction / replication / generalization ([definitions](../roadmap/09-research-methodology.md#44-levels-of-reproducibility))

---

## Before running

### Research question

One sentence.

### Hypothesis

A falsifiable prediction with direction and, ideally, magnitude. What result would show it is wrong?

### Baseline

The simplest reasonable comparison, tuned with the same effort as the method under test.

### Variables

- **Independent (changed deliberately):** 
- **Dependent (measured):** 
- **Controlled (held fixed):** 

### Experimental controls

Control conditions, randomization, and known confounders and how they are handled.

### Environment and model configuration

- Code: path, git commit
- Hardware: CPU / GPU model, memory
- Software: Python and key package versions
- Model and hyperparameters: full configuration or link to it
- Seeds: list
- Data: source, version, splits

### Procedure

Numbered steps.

### Analysis plan

Primary metric; number of runs; how uncertainty will be reported (e.g. 95% bootstrap CI over seeds); what comparison decides the hypothesis.

---

## After running

### Results and measurements

Tables and figures with uncertainty. Link raw results.

### Interpretation

Does the result support or contradict the hypothesis? Separate what was measured from what you infer. Note any deviation from the analysis plan.

### Limitations

Scope, confounders not controlled, sample size, anything that would make you less confident.

### Reproduction instructions

Exact commands from a clean clone, including how to obtain data.

### Next experiment

What this result suggests testing next. Add it to the [research backlog](../research-backlog.md).
