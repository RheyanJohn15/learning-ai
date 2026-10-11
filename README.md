# learning-ai

> I want to learn AI down to its low-level architecture and mathematics.

A long-term, documented path from mathematical foundations to original AI research — built by a software engineer moving from *using* AI APIs to *understanding, implementing, evaluating, and improving* AI systems.

## Objective

To become an AI researcher who can:

- derive and implement neural networks, backpropagation, and Transformers from first principles;
- explain how models train and run on CPUs and GPUs, down to memory traffic and numerical precision;
- investigate what models learn internally (interpretability);
- study how agentic systems behave, fail, and can be constrained (agents, alignment, safety);
- read papers critically, reproduce results, and run original, evidence-based experiments.

## Philosophy

- **First principles before abstractions.** Derive it, implement it in NumPy, verify it numerically — then use the framework version.
- **Verification over belief.** Every implementation is checked against something independent: a numerical gradient, a reference implementation, a published result.
- **Hypotheses before experiments.** Write down what you expect, and what would prove you wrong, before running anything.
- **Honest records.** Unclear points, failed experiments, and negative results are documented like successes. Nothing is marked complete without evidence.
- **Minimal infrastructure.** Markdown, Python, and version control until something more is measurably needed.

## Roadmap

| Phase | Area | Builds |
|---|---|---|
| A | [Mathematical foundations](docs/roadmap/00-foundations.md) | Linear algebra, calculus, probability, information theory, numerics |
| B | [Neural networks from scratch](docs/roadmap/01-neural-networks.md) | Backpropagation by hand in NumPy; XOR network; a small autograd engine |
| C | [Deep learning frameworks](docs/roadmap/02-deep-learning-frameworks.md) | PyTorch mapped back to what you built; training engineering |
| D | [Transformers and LLMs](docs/roadmap/03-transformers-and-llms.md) | Bigram model → MLP → GPT from scratch; KV cache; GPT-2 weights in your own code |
| E | [Training and optimization](docs/roadmap/04-training-and-optimization.md) | Optimizers, mixed precision, memory and compute accounting |
| F | [Computer architecture and GPUs](docs/roadmap/05-computer-architecture-and-gpu-computing.md) | Memory hierarchy, roofline, C++ and CUDA kernels |
| G | [Interpretability](docs/roadmap/06-model-interpretability.md) | Probes, circuits, activation patching, sparse autoencoders |
| H | [Agentic AI and RL](docs/roadmap/07-agentic-ai-and-reinforcement-learning.md) | RL algorithms; a minimal agent loop with enforced permissions |
| I | [Alignment and safety](docs/roadmap/08-ai-alignment-and-safety.md) | Specification gaming, robustness, oversight, evaluation limits |
| J | [Research methodology](docs/roadmap/09-research-methodology.md) | Runs alongside everything: hypotheses, statistics, replication, writing |

A → B → C → D is the critical path; E, F, and G can be interleaved after D; H and I build on them; J runs throughout. See the [roadmap overview](docs/roadmap/README.md) for dependencies and how to work through a phase.

## Navigating the documentation

| Start here | For |
|---|---|
| [docs/README.md](docs/README.md) | Documentation map, progress conventions, claim labels |
| [docs/roadmap/](docs/roadmap/README.md) | Phases, dependencies, and the progress tracker |
| [docs/notation.md](docs/notation.md) | Shared symbols and tensor-shape conventions |
| [docs/resources/](docs/resources/README.md) | Cross-phase textbooks, courses, and the source policy |
| [AGENTS.md](AGENTS.md) | Instructions for AI assistants working in this repository |

## How work is recorded

| Record | Location | Template |
|---|---|---|
| What I studied and what is still unclear | [docs/learning-log/](docs/learning-log/README.md) | [template](docs/learning-log/TEMPLATE.md) |
| Critical notes on papers | [docs/papers/](docs/papers/README.md) | [template](docs/papers/TEMPLATE.md) |
| Pre-registered experiments and results | [docs/experiments/](docs/experiments/README.md) | [template](docs/experiments/TEMPLATE.md) |
| Concept notes synthesized across sources | [docs/research-notes/](docs/research-notes/README.md) | free-form |
| Questions, ideas, planned experiments | [docs/research-backlog.md](docs/research-backlog.md) | one line per item |

## Current status

- **Current phase:** Phase A — Mathematical foundations (not yet started).
- **Next milestone:** complete the [Phase A self-assessment](docs/roadmap/00-foundations.md#2-prerequisites), then milestone A1 (matrix multiplication from scratch, verified against NumPy).
- **Completed phases:** none.
- **Repository contents:** documentation, plus a first scaffold in [implementations/](implementations/README.md) (B3 XOR, `backward()` not yet written).

## Tracking progress

Phase status lives in one place: the [progress tracker](docs/roadmap/README.md#progress-tracker). Each phase document lists numbered milestones (A1, B3, …) and completion criteria as checkboxes. A box is ticked only with a link to evidence — a learning-log entry, an experiment record, or code — following the [progress conventions](docs/README.md#progress-conventions).
