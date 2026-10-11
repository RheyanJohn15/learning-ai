# Documentation

Index and conventions for everything under `docs/`.

## Contents

- [Map](#map)
- [Workflow and templates](#workflow-and-templates)
- [Progress conventions](#progress-conventions)
- [Claim labels](#claim-labels)
- [Writing conventions](#writing-conventions)

## Map

| Location | Purpose |
|---|---|
| [roadmap/](roadmap/README.md) | The learning path: ten phases, dependencies, and the progress tracker |
| [notation.md](notation.md) | Shared symbols and tensor-shape conventions used across all documents |
| [learning-log/](learning-log/README.md) | Dated entries recording what was studied, understood, and still unclear |
| [papers/](papers/README.md) | Structured notes on research papers, with critical analysis |
| [experiments/](experiments/README.md) | Pre-registered experiment records: hypothesis, method, results, interpretation |
| [research-notes/](research-notes/README.md) | Synthesized notes on topics, written in your own words, spanning several sources |
| [research-backlog.md](research-backlog.md) | Ideas, open questions, planned experiments, and potential projects |
| [resources/](resources/README.md) | Cross-phase textbooks, courses, tools, and source-vetting guidance |

## Workflow and templates

| Activity | Where | Template |
|---|---|---|
| Studying a topic | `learning-log/` | [learning-log/TEMPLATE.md](learning-log/TEMPLATE.md) |
| Reading a paper | `papers/` | [papers/TEMPLATE.md](papers/TEMPLATE.md) |
| Running an experiment | `experiments/` | [experiments/TEMPLATE.md](experiments/TEMPLATE.md) |
| Consolidating understanding of a topic | `research-notes/` | Free-form; see [research-notes/README.md](research-notes/README.md) |
| Capturing a question or idea | [research-backlog.md](research-backlog.md) | One line per item |

How the pieces connect:

```text
learning log ──"what remains unclear"──► research backlog ◄──"follow-up experiments"── paper notes
                                              │
                                              ▼
                                         experiment ──"results and interpretation"──► research notes
                                              │
                                              └──"evidence"──► completion criteria in the roadmap
```

Keep it lightweight. A two-line log entry is better than none; a template section that does not apply can be marked "n/a".

## Progress conventions

**Phase status** (in the [progress tracker](roadmap/README.md#progress-tracker)) uses exactly three values:

- **Not started**
- **In progress** — at least one milestone has been attempted.
- **Completed** — every core completion criterion in the phase document is ticked with evidence.

**Checkboxes** inside phase documents track milestones (`A1`, `B3`, …) and completion criteria:

```markdown
- [ ] **B3** XOR network with manual backpropagation…
- [x] **B3** XOR network with manual backpropagation… — evidence: [EXP-001](experiments/EXP-001-xor-reliability.md)
```

Rules:

1. A box is ticked **only** with a link to evidence: a learning-log entry, an experiment record, a paper note, or code.
2. Boxes are ticked **only** after the learner confirms completion. AI assistants must not tick boxes or change status on their own initiative (see [AGENTS.md](../AGENTS.md)).
3. When a phase changes status, update the tracker row (status, dates, evidence) and the "Current phase" line in the root [README](../README.md#current-status).
4. Un-ticking is allowed and encouraged if later work shows a gap in understanding.

## Claim labels

When a note makes a claim that is not obviously settled, label it:

| Label | Meaning |
|---|---|
| **Established** | Replicated across independent groups and settings; broadly accepted |
| **Empirical finding** | Reported under specific conditions; scope limited to those conditions |
| **Hypothesis** | Testable, not yet adequately tested |
| **Interpretation** | One reading of evidence that admits others |
| **Open question** | Unresolved; no strong evidence either way |

Use labels for the claims that matter, not every sentence. [Phase I](roadmap/08-ai-alignment-and-safety.md#epistemic-conventions-for-this-phase) applies these with particular care.

## Writing conventions

- **Equations:** inline `$...$` and display `$$...$$` (rendered by GitHub and most Markdown previews). Use the symbols in [notation.md](notation.md).
- **Links:** relative links between documents; link to the original paper (arXiv, publisher, or DOI), not a secondary summary, wherever possible.
- **Citations:** author(s), title, year, link. Never cite a source that has not been verified to exist.
- **Naming:** lowercase, hyphen-separated file names. Dated items start with `YYYY-MM-DD`; experiments use `EXP-NNN-short-slug.md`; paper notes use `YYYY-firstauthor-short-title.md`.
- **Indexes:** each folder's README keeps an index; add a row when you add a file.
- **Size:** prefer several small documents over one large one. Split when a document stops being navigable.
