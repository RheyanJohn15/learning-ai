# AGENTS.md

Instructions for AI assistants working in this repository. This file covers **repository conventions and workflow**. Teaching behavior — how to explain, derive, and mentor — is in [.cursor/rules/ai-research-mentor.mdc](.cursor/rules/ai-research-mentor.mdc). Follow both.

## What this repository is

A personal, long-term AI research learning repository. The learner is an experienced software engineer (full-stack, backend, APIs, DevOps) learning AI from first principles with the goal of doing original research. The learning path is in [docs/roadmap/](docs/roadmap/README.md); read the relevant phase document before helping with any phase-specific task.

**Current stage:** documentation plus the first implementation scaffold. Dependencies are pinned in `requirements.txt` (NumPy only) and installed into a root `.venv` created with the built-in `venv`; see [implementations/README.md](implementations/README.md#setup).

## Repository map

```text
README.md                    entry point and current status
AGENTS.md                    this file
.cursor/rules/               Cursor rules (mentor behavior)
docs/README.md               documentation map, progress conventions, claim labels
docs/notation.md             shared symbols and tensor-shape conventions
docs/roadmap/                phases A–J (00–09) and the progress tracker
docs/learning-log/           dated study entries + TEMPLATE.md
docs/papers/                 paper notes + TEMPLATE.md
docs/experiments/            experiment records + TEMPLATE.md
docs/research-notes/         concept notes synthesized across sources
docs/research-backlog.md     questions, ideas, planned experiments
docs/resources/              cross-phase resources and source policy
implementations/             milestone code, one directory per milestone ID
requirements.txt             pinned Python dependencies
```

## Non-negotiable rules

1. **Never fabricate.** No invented papers, authors, years, URLs, quotes, benchmark numbers, experimental results, or hardware specifications. If unsure a source exists or says what you think, say so and suggest how to verify it.
2. **Never claim progress on the learner's behalf.** Do not tick checkboxes, change phase status, or update "Current status" unless the learner explicitly confirms completion **and** there is evidence to link (a log entry, experiment record, or code). When asked to record completion, follow [progress conventions](docs/README.md#progress-conventions).
3. **Never present a result you did not observe.** If code was not run, say so. Do not write "results" sections for experiments that have not been executed.
4. **Label claims.** Distinguish established findings, empirical findings, hypotheses, interpretations, and open questions ([claim labels](docs/README.md#claim-labels)).
5. **Keep experiments harmless.** Safety and agent experiments use toy environments, canary actions, fixture data, and sandboxes. No real credentials, no real external side effects, no attempts to elicit genuinely harmful capabilities.

## Languages and tools

| Purpose | Default | Introduced in |
|---|---|---|
| Mathematical implementations | Python + NumPy | Phases A–B |
| Deep learning | PyTorch | Phase C, **after** manual backpropagation in Phase B |
| Systems and kernels | C++, CUDA (Triton as an intermediate step) | Phase F |
| Notes and records | Markdown with `$...$` / `$$...$$` math | Always |

Alternatives need a stated justification. Examples of justified alternatives: JAX for research that needs functional transformations (`vmap`, `jit`, higher-order derivatives); Triton for a kernel that would be impractical in CUDA for learning purposes; ROCm/HIP if the available GPU is AMD. Avoid introducing a language or tool only because it is fashionable.

**Dependencies:** do not add dependencies, a package manager, CI, a documentation site, databases, or experiment-tracking services without the learner's agreement. When the first dependency is needed (NumPy in Phase A), propose the minimal option, explain the trade-offs, and ask. Pin versions once dependencies exist.

## Code layout (adopted)

Implementation code lives under `implementations/`, indexed in [implementations/README.md](implementations/README.md):

```text
implementations/
  00-foundations/a1-matmul/...
  01-neural-networks/b3-xor/...
```

One directory per milestone, named after its ID; small and self-contained; a short README linking back to the milestone and any experiment records. Add a row to the implementations index for each new directory.

Code conventions once code exists:

- Annotate tensor shapes in comments where they are not obvious (`# (B, T, d)`), using [docs/notation.md](docs/notation.md).
- Assert shapes at function boundaries in from-scratch implementations.
- Seed everything; log the seed.
- Verify against something independent (numerical gradients, a reference implementation) and keep that check runnable.
- **Never commit** datasets, model weights, checkpoints, or large generated files. Document how to obtain or regenerate them.

## Documentation workflow

| When the learner… | Do this |
|---|---|
| Studies a topic | Offer a learning-log entry from [the template](docs/learning-log/TEMPLATE.md); capture what remains unclear |
| Reads a paper | Use [the paper template](docs/papers/TEMPLATE.md); verify bibliographic details against the original |
| Plans an experiment | Use [the experiment template](docs/experiments/TEMPLATE.md); complete the pre-registration section **before** any code runs |
| Finishes an experiment | Fill in results only from observed outputs; update the backlog and indexes |
| Raises a question worth keeping | Add a one-line item to [docs/research-backlog.md](docs/research-backlog.md) |
| Confirms a milestone with evidence | Tick the box with an evidence link; update the tracker and README status if a phase changes |

Maintenance rules:

- Add a row to the folder's index README whenever a file is added.
- Use relative links; check that every internal link and anchor resolves after editing.
- Prefer editing an existing document over creating a new one; split documents only when navigation suffers.
- Do not duplicate roadmap content into other files; link to it.
- Keep templates lightweight. Process should support research, not replace it.
