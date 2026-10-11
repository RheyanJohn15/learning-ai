# Resources

Cross-phase resources and guidance on choosing sources. Phase-specific textbooks, papers, and code are listed in each phase document's **Resources** section; this page does not duplicate them.

## Contents

- [Source policy](#source-policy)
- [Backbone textbooks](#backbone-textbooks)
- [Courses and lecture series](#courses-and-lecture-series)
- [Readable reference implementations](#readable-reference-implementations)
- [Finding and tracking papers](#finding-and-tracking-papers)
- [Research publications and blogs](#research-publications-and-blogs)
- [Resources by phase](#resources-by-phase)

## Source policy

Prefer, in order:

1. **Original papers** (arXiv, conference proceedings, journals) and **official documentation**.
2. **Textbooks and university courses** from recognized authors and institutions.
3. **Source code** of widely used, maintained implementations.
4. **Technical blogs** by researchers who show their work (code, derivations, experiments).

Before relying on a source:

- Verify that it exists and that title, authors, and year are correct.
- Check whether a claim is the paper's main result, a side remark, or a citation of someone else.
- Note whether results were replicated, and whether the paper is peer-reviewed or a preprint (preprints are normal in ML; still note it).
- For anything fast-moving (library APIs, hardware specifications), check the date and current official documentation.

AI assistants working in this repository must follow the same policy and must never invent references (see [AGENTS.md](../../AGENTS.md)).

## Backbone textbooks

Free online versions are linked where the authors provide them.

| Book | Use |
|---|---|
| Deisenroth, Faisal, Ong — [*Mathematics for Machine Learning*](https://mml-book.github.io/) (2020) | Phase A reference |
| Goodfellow, Bengio, Courville — [*Deep Learning*](https://www.deeplearningbook.org/) (2016) | Foundations; dated on Transformers |
| Simon Prince — [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (2023) | Modern, clear, well illustrated |
| Bishop and Bishop — [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/) (2024) | Rigorous modern alternative |
| Zhang, Lipton, Li, Smola — [*Dive into Deep Learning*](https://d2l.ai/) | Code-first, runnable examples |
| Sutton and Barto — [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed., 2018) | Phase H |
| Bryant and O'Hallaron — [*Computer Systems: A Programmer's Perspective*](https://csapp.cs.cmu.edu/) (3rd ed.) | Phase F, CPU side |
| Hwu, Kirk, El Hajj — *Programming Massively Parallel Processors* (4th ed., 2022) | Phase F, GPU side |

## Courses and lecture series

- Andrej Karpathy — [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html): from micrograd to GPT; aligns with Phases B–D.
- Stanford CS231n — [course notes](https://cs231n.github.io/): backpropagation and neural networks.
- Stanford CS224n — [Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/).
- MIT 18.06 — [Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (Gilbert Strang).
- 3Blue1Brown — [linear algebra](https://www.3blue1brown.com/topics/linear-algebra), [calculus](https://www.3blue1brown.com/topics/calculus), and [neural networks](https://www.3blue1brown.com/topics/neural-networks) series.
- David Silver — [UCL reinforcement learning course](https://www.davidsilver.uk/teaching/).
- ARENA — [interpretability and alignment curriculum](https://www.arena.education/).
- GPU MODE — [lecture materials](https://github.com/gpu-mode/lectures) on GPU programming.

## Readable reference implementations

Small enough to read in full; use them to check your own implementations, not to replace them.

- [micrograd](https://github.com/karpathy/micrograd) — scalar autograd (Phase B).
- [makemore](https://github.com/karpathy/makemore), [nanoGPT](https://github.com/karpathy/nanoGPT), [minbpe](https://github.com/karpathy/minbpe) — language models and tokenization (Phase D).
- [llm.c](https://github.com/karpathy/llm.c) — GPT-2 training in C/CUDA (Phases D and F).
- [tinygrad](https://github.com/tinygrad/tinygrad) — a small deep-learning framework (Phase C).
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file RL algorithms (Phase H).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — interpretability tooling (Phase G).

## Finding and tracking papers

- [arXiv](https://arxiv.org/) — most ML papers appear here first (categories `cs.LG`, `cs.CL`, `cs.AI`, `stat.ML`).
- [Semantic Scholar](https://www.semanticscholar.org/) — citation graphs; find who built on a paper.
- [OpenReview](https://openreview.net/) — peer reviews and author responses for several major venues; reviews are often an excellent critical reading aid.
- [ACL Anthology](https://aclanthology.org/) — NLP venue proceedings.
- [Proceedings of Machine Learning Research](https://proceedings.mlr.press/) — ICML, AISTATS, and other proceedings.

## Research publications and blogs

- [Distill](https://distill.pub/) — interactive ML research articles (no longer publishing, archive remains valuable).
- [Transformer Circuits Thread](https://transformer-circuits.pub/) — interpretability research.
- [Lil'Log](https://lilianweng.github.io/) — Lilian Weng's detailed literature reviews.
- [Andrej Karpathy's blog](https://karpathy.github.io/).
- [Alignment Forum](https://www.alignmentforum.org/) — alignment research discussion; quality varies, so apply the source policy.

## Resources by phase

- [Phase A — Mathematical foundations](../roadmap/00-foundations.md#7-resources)
- [Phase B — Neural networks from scratch](../roadmap/01-neural-networks.md#7-resources)
- [Phase C — Deep learning frameworks](../roadmap/02-deep-learning-frameworks.md#7-resources)
- [Phase D — Transformers and LLMs](../roadmap/03-transformers-and-llms.md#7-resources)
- [Phase E — Training and optimization](../roadmap/04-training-and-optimization.md#7-resources)
- [Phase F — Computer architecture and GPU computing](../roadmap/05-computer-architecture-and-gpu-computing.md#7-resources)
- [Phase G — Model interpretability](../roadmap/06-model-interpretability.md#7-resources)
- [Phase H — Agentic AI and reinforcement learning](../roadmap/07-agentic-ai-and-reinforcement-learning.md#7-resources)
- [Phase I — AI alignment and safety](../roadmap/08-ai-alignment-and-safety.md#7-resources)
- [Phase J — Research methodology](../roadmap/09-research-methodology.md#7-resources)
