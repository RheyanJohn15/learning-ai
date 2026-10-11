# Experiments

Records of experiments, written so that the hypothesis is fixed before results are seen and the result can be reproduced later.

## How to use

1. Copy [TEMPLATE.md](TEMPLATE.md) to `EXP-NNN-short-slug.md` using the next unused number (for example `EXP-001-xor-reliability.md`).
2. **Before running:** fill in the question, hypothesis, baseline, variables, controls, configuration, procedure, and analysis plan. Commit the file. This is the pre-registration.
3. **After running:** fill in results, interpretation, limitations, and reproduction instructions. If the plan changed, say what changed and why; do not silently edit the hypothesis.
4. Update the experiment's line in the [research backlog](../research-backlog.md) and add or update its row in the index below.

## Lifecycle

`Planned` → `Running` → `Completed`, or `Abandoned` (with a reason). Negative and inconclusive results are recorded like any other result.

## Code, data, and results

- Code for an experiment lives with the implementation code in [implementations/](../../implementations/README.md) and is referenced by path and git commit hash.
- Small raw results (CSV or JSON, kilobytes) may sit next to the record as `EXP-NNN-short-slug.results.csv`.
- Datasets, model weights, and large outputs are **never** committed. Record how to obtain or regenerate them.

## Index

| ID | Title | Phase | Status | Outcome |
|---|---|---|---|---|
| — | _No experiments yet._ | — | — | — |
