# Phase H — Agentic AI and Reinforcement Learning

> **Phase H** · File `07-agentic-ai-and-reinforcement-learning.md` · Depends on: [Phase C](02-deep-learning-frameworks.md) for the reinforcement-learning track; [Phase D](03-transformers-and-llms.md) for the language-model agent track · Next: [Phase I — AI Alignment and Safety](08-ai-alignment-and-safety.md)
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

Reinforcement learning (RL) is the mathematical framework for learning from actions and their consequences. It underpins RLHF ([Phase D §4.7](03-transformers-and-llms.md#47-post-training-objectives)), reward hacking ([Phase I](08-ai-alignment-and-safety.md)), and the training of reasoning and agentic models. Separately, language-model agents — models wrapped in software that lets them take actions — are where many current safety questions become concrete: multi-step error compounding, prompt injection, tool misuse, and policy violations. Your backend and DevOps experience is directly relevant here: authorization, sandboxing, and least privilege are software-engineering problems.

This phase has two tracks. The **RL track** can begin after Phase C. The **agent track** needs Phase D.

## 2. Prerequisites

- [Phase A](00-foundations.md): probability, expectation, Markov chains (learn just in time).
- [Phase B](01-neural-networks.md) and [Phase C](02-deep-learning-frameworks.md): function approximation with neural networks, PyTorch.
- [Phase D](03-transformers-and-llms.md) (agent track): sampling, context windows, post-training.
- [Phase J](09-research-methodology.md): RL and agent results are high-variance; confidence intervals are mandatory.

## 3. Learning objectives

### 3.1 Core — reinforcement learning

- Markov decision processes (MDPs): states, actions, transition dynamics, rewards, discount factor; partially observable MDPs.
- Policies, returns, state-value and action-value functions; Bellman equations.
- Dynamic programming: policy evaluation, policy iteration, value iteration.
- Exploration versus exploitation: $\epsilon$-greedy, upper confidence bounds, Thompson sampling (bandits).
- Model-free learning: Monte Carlo, temporal-difference learning, SARSA, Q-learning.
- Function approximation: deep Q-networks (DQN).
- Policy gradients: REINFORCE, baselines, actor–critic, generalized advantage estimation, PPO.
- The link to language models: tokens as actions, sequence-level rewards, KL-regularized RL.

### 3.2 Core — language-model agents

- **The three-layer distinction** (§4.1): model, agent framework, environment.
- The agent loop: observation → model call → parse action → execute tool → new observation.
- Planning (chain-of-thought, task decomposition), memory (context, retrieval, scratchpads), tool calling via structured outputs.
- Multi-step feedback loops and error compounding.
- Instruction conflicts, instruction hierarchy, direct and indirect prompt injection.
- Tool permissions, authorization, sandboxing, human confirmation, audit logging.
- Evaluation: task success, partial credit, policy violations, cost, number of steps, and variance across runs.

### 3.3 Optional advanced

Model-based RL, offline RL, inverse RL, multi-agent RL, RL for reasoning with verifiable rewards.

## 4. Theory

### 4.1 Model, agent framework, and environment

These three are routinely conflated; keep them separate in your notes and experiments.

- **Language model:** a function from a token sequence to a probability distribution over the next token. It does not execute anything, has no memory beyond its input, and has no access to anything outside that input.
- **Agent framework (scaffold, harness):** ordinary software that builds prompts, calls the model, parses outputs into tool calls, executes those calls, feeds results back as new input, manages memory, enforces step limits, and (should) enforce permissions.
- **Environment:** everything the framework's tools touch — file systems, shells, APIs, browsers, databases, users — with its own state and side effects.

In RL terms, the *policy* is the composite of model and framework, mapping observation histories to actions. Capabilities and failures arise from the composition. A security property ("the agent cannot delete files outside `/tmp/work`") must be enforced by the framework or environment; text in a prompt is not an enforcement mechanism.

### 4.2 MDPs and Bellman equations

The return from time $t$ is $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$. For a policy $\pi$:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^\pi(s')\right]
$$

$$
Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \,\middle|\, s, a\right]
$$

Q-learning moves toward this fixed point from samples:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
$$

### 4.3 Policy gradients

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big)\right]
$$

A baseline $b(s_t)$ that does not depend on the action leaves the gradient unbiased while reducing variance. PPO constrains each update with a clipped probability ratio $\rho_t = \pi_\theta(a_t \mid s_t)/\pi_{\theta_{\text{old}}}(a_t \mid s_t)$:

$$
L^{\text{CLIP}} = \mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,\ \operatorname{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon)\hat{A}_t\right)\right]
$$

For a language model, each token is an action, the state is the prefix, and the reward often arrives only at the end of the sequence.

### 4.4 Error compounding

Under the simplifying assumption that each step of an $n$-step task succeeds independently with probability $p$, the whole task succeeds with probability $p^n$: at $p = 0.95$ and $n = 20$, about $0.36$. Real agents can recover from errors, and errors are correlated, so this is a baseline model to test against, not a prediction.

### 4.5 Prompt injection

A model receives instructions and data in the same token stream and has no hard boundary between them. **Indirect prompt injection** places instructions inside data the agent retrieves (web pages, documents, tool outputs). Prompt-level defenses (delimiters, warnings, instruction-hierarchy training) reduce attack success rates but have not been shown to eliminate them. Robust mitigations constrain what the agent *can* do: least-privilege tools, argument validation, confirmation for consequential actions, and isolating untrusted data from control flow.

## 5. Implementation

All agent experiments use harmless tools, fixture data, and sandboxed execution. No real credentials, no real external side effects.

**RL track (NumPy first, then PyTorch)**

- [ ] **H1** Multi-armed bandits: $\epsilon$-greedy, UCB, and Thompson sampling; plot regret curves.
- [ ] **H2** A gridworld MDP with value iteration and policy iteration from scratch; confirm they produce the same policy.
- [ ] **H3** Tabular Q-learning and SARSA; reproduce the qualitative cliff-walking difference described by Sutton and Barto.
- [ ] **H4** REINFORCE with a baseline on CartPole (via Gymnasium).
- [ ] **H5** PPO (or DQN) from scratch on CartPole; compare learning curves with a CleanRL reference implementation across seeds.

**Agent track**

- [ ] **H6** A minimal agent loop **without an agent framework**: model call, tool registry, output parser, executor, observation feedback, step limit. Tools: a calculator and a read-only file reader restricted to a fixture directory.
- [ ] **H7** A permission layer: tool allowlist, argument validation, confirmation for write actions, and an audit log; tests showing disallowed actions are blocked regardless of what the model outputs.
- [ ] **H8** A prompt-injection testbed: fixture documents containing injected instructions that try to trigger a harmless **canary** action (for example writing the string `CANARY` to a note tool). Measure attack success rates.
- [ ] **H9** An evaluation harness: tasks with programmatic success checks, $N$ trials per configuration, success rates with confidence intervals, and violation counts.

## 6. Experiments

1. **Exploration schedules.** Regret under different $\epsilon$ schedules and UCB constants.
2. **RL variance.** Train H5 with at least 10 seeds; report interquartile mean and confidence intervals (Agarwal et al., 2021). How many seeds are needed to distinguish two hyperparameter settings?
3. **Reward shaping side effects.** Add a shaping term to the gridworld reward and observe whether the learned policy changes in unintended ways (bridges to [Phase I](08-ai-alignment-and-safety.md)).
4. **Error compounding.** Agent success rate versus task length; compare with the $p^n$ baseline from §4.4.
5. **Model versus framework attribution.** Same model across two frameworks, and the same framework across two models; which factor changes success and violation rates more?
6. **Prompt-injection defenses.** Canary-action success rate with no defense, with delimiters and warnings, with an instruction-hierarchy system prompt, and with permission enforcement. Hypothesis: prompt-level defenses reduce but do not eliminate success; enforced permissions block the canary action entirely.

## 7. Resources

**Core — reinforcement learning**

- Sutton and Barto — *Reinforcement Learning: An Introduction* (2nd ed., 2018), free: <http://incompleteideas.net/book/the-book-2nd.html>.
- OpenAI — *Spinning Up in Deep RL*: <https://spinningup.openai.com/>.
- David Silver — UCL reinforcement learning course: <https://www.davidsilver.uk/teaching/>.
- Gymnasium (environments): <https://gymnasium.farama.org/>.
- CleanRL (single-file reference implementations): <https://github.com/vwxyzjn/cleanrl>; Huang et al., *JMLR* (2022): <https://jmlr.org/papers/v23/21-1342.html>.

**Papers — reinforcement learning**

- Williams — "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (REINFORCE, 1992): <https://doi.org/10.1007/BF00992696>.
- Mnih et al. — "Human-level control through deep reinforcement learning", *Nature* (2015): <https://www.nature.com/articles/nature14236>.
- Schulman et al. — "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015): <https://arxiv.org/abs/1506.02438>.
- Schulman et al. — "Proximal Policy Optimization Algorithms" (2017): <https://arxiv.org/abs/1707.06347>.
- Henderson et al. — "Deep Reinforcement Learning that Matters" (2017): <https://arxiv.org/abs/1709.06560>.
- Agarwal et al. — "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (2021): <https://arxiv.org/abs/2108.13264>.

**Core — agents**

- Anthropic — "Building effective agents" (2024): <https://www.anthropic.com/research/building-effective-agents>.
- Model Context Protocol specification (a common tool-integration protocol): <https://modelcontextprotocol.io/>.
- OWASP — Top 10 for LLM Applications: <https://genai.owasp.org/llm-top-10/>.
- Saltzer and Schroeder — "The Protection of Information in Computer Systems" (1975), the classic statement of least privilege: <https://doi.org/10.1109/PROC.1975.9939>.

**Papers — agents and prompt injection**

- Wei et al. — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022): <https://arxiv.org/abs/2201.11903>.
- Yao et al. — "ReAct: Synergizing Reasoning and Acting in Language Models" (2022): <https://arxiv.org/abs/2210.03629>.
- Schick et al. — "Toolformer" (2023): <https://arxiv.org/abs/2302.04761>.
- Shinn et al. — "Reflexion" (2023): <https://arxiv.org/abs/2303.11366>.
- Perez and Ribeiro — "Ignore Previous Prompt: Attack Techniques For Language Models" (2022): <https://arxiv.org/abs/2211.09527>.
- Greshake et al. — "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (2023): <https://arxiv.org/abs/2302.12173>.
- Wallace et al. — "The Instruction Hierarchy" (2024): <https://arxiv.org/abs/2404.13208>.
- Debenedetti et al. — "AgentDojo" (2024): <https://arxiv.org/abs/2406.13352>.
- Debenedetti et al. — "Defeating Prompt Injections by Design" (CaMeL, 2025): <https://arxiv.org/abs/2503.18813>.
- Ruan et al. — "Identifying the Risks of LM Agents with an LM-Emulated Sandbox" (ToolEmu, 2023): <https://arxiv.org/abs/2309.15817>.

**Papers — agent evaluation**

- Jimenez et al. — "SWE-bench" (2023): <https://arxiv.org/abs/2310.06770>.
- Zhou et al. — "WebArena" (2023): <https://arxiv.org/abs/2307.13854>.
- Liu et al. — "AgentBench" (2023): <https://arxiv.org/abs/2308.03688>.
- Kinniment et al. — "Evaluating Language-Model Agents on Realistic Autonomous Tasks" (2023): <https://arxiv.org/abs/2312.11671>.

## 8. Completion criteria

- [ ] Derive the Bellman expectation equation and the policy-gradient theorem on paper.
- [ ] H2: value iteration and policy iteration converge to the same policy.
- [ ] H5: PPO or DQN solves CartPole across at least 5 seeds, with variance reported.
- [ ] H6: a minimal agent loop built without a framework, with each component explained in a learning-log entry.
- [ ] H7: a permission layer with tests showing enforcement is independent of model output.
- [ ] H8/Experiment 6: an injection experiment report with success rates and confidence intervals.
- [ ] Explain the model / framework / environment distinction, with an example of a failure originating in each layer.

## 9. Research directions

- How reliable are agent benchmarks, and how much do scores vary across runs, frameworks, and prompt formats?
- Which prompt-injection defenses hold up under adaptive attacks, and what can be guaranteed by system design rather than model behavior?
- Monitoring agent trajectories: can a second model reliably flag policy violations, and how is it evaded?
- Reward hacking in RL-trained agents (connects to [Phase I](08-ai-alignment-and-safety.md)).
- How does RL with sequence-level rewards change a language model's internal computation? (Connects to [Phase G](06-model-interpretability.md).)

## 10. Common misconceptions

- **"The language model executes tools."** The model emits text; the framework decides whether and how to execute it.
- **"The system prompt is a security boundary."** It is input text with no enforcement power; enforce permissions in code.
- **"An agent that succeeded once is reliable."** Report success rates over many trials with confidence intervals.
- **"The reward is what we want."** The reward is what we wrote down; agents optimize what is written (see [Phase I](08-ai-alignment-and-safety.md)).
- **"The agent wants to achieve its goal."** Describe observed behavior and the optimization that produced it. Whether and when it is useful to model a system as goal-directed is a research question.
- **"A high benchmark score implies real-world capability."** Benchmarks have narrow coverage, contamination risk, and framework-dependent results.

---

Previous: [Phase G — Interpretability](06-model-interpretability.md) · Next: [Phase I — AI Alignment and Safety](08-ai-alignment-and-safety.md) · Up: [Roadmap](README.md)
