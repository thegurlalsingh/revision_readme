# Agentic AI — Interview Preparation

A condensed, interview-ready handbook covering agent architectures, tool use, memory, multi-agent systems, frameworks, safety, and evaluation.

## Table of Contents

1. [Agent Fundamentals](#1-agent-fundamentals)
2. [Core Agent Reasoning Patterns](#2-core-agent-reasoning-patterns)
3. [Tool Use & Function Calling](#3-tool-use--function-calling)
4. [Multi-Agent Systems](#4-multi-agent-systems)
5. [Agent Memory](#5-agent-memory)
6. [Agentic RAG](#6-agentic-rag)
7. [Agent Frameworks](#7-agent-frameworks)
8. [Reliability, Safety & Security](#8-reliability-safety--security)
9. [Observability & Evaluation](#9-observability--evaluation)
10. [Cost & Latency Control](#10-cost--latency-control)
11. [Enterprise & Organizational Concerns](#11-enterprise--organizational-concerns)
12. [Classical AI Agents vs Modern LLM Agents](#12-classical-ai-agents-vs-modern-llm-agents)
13. [Real-World Agent Examples](#13-real-world-agent-examples)
14. [Agent System Design Framework](#14-agent-system-design-framework)
15. [Common Interview Questions](#15-common-interview-questions)
16. [Interview Traps](#16-interview-traps)
17. [Quick Revision Cheat Sheet](#17-quick-revision-cheat-sheet)

---

## 1. Agent Fundamentals

### What is an Agent?

> **An agent is not just an LLM that gives an answer — it's an LLM embedded in a loop where it can observe state, decide what to do next, take an action, observe the result, and continue until a stopping condition is reached.**

**Single LLM call:** `Input → LLM → Output` — the developer controls the entire flow.

**Agent:** `Goal → LLM decides action → Act → Observe → LLM decides next action → ...` — the model has control over the next step.

⚠️ **Interview Trap:** "An agent is an LLM that uses tools" is incomplete. Tool use matters, but the deeper distinction is **autonomous, multi-step decision-making**, not a single tool call.

### The Four Elements That Make Something an Agent

| Element | Meaning |
|---|---|
| **Persistent state** | Remembers what happened across steps |
| **Decision-making** | Decides the next action itself |
| **Environment interaction** | Can use tools/APIs/browser/code execution |
| **Termination condition** | Knows when to stop |

### The Basic Agent Loop

```
Goal
 ↓
Observe current state
 ↓
Reason / decide next action
 ↓
Act (tool call)
 ↓
Observe result
 ↓
Update state
 ↓
Goal achieved? ──No──┐
   │Yes               │
   ↓                  │
Output ◄───────────────┘ (loop back to Reason)
```

**Agent state** typically contains: `goal, conversation_history, current_plan, tool_outputs, intermediate_results, errors, retry_count, final_answer`.

**Agent state ≠ conversation history.** Conversation history is what was said; state is the broader execution context (goal, budget, retries, selected options, etc.) — conversation history is just one part of it.

### Termination Conditions

An agent needs *multiple* stopping mechanisms, not just one:

| Condition | Example |
|---|---|
| Goal achieved | Explicit check against original constraints |
| Max steps/iterations | `max_steps = 10` |
| Token/cost budget | `max_tokens = 50K`, `max_cost = $0.50` |
| Time limit | `max_execution_time = 60s` |
| Model signals "done" | Structured `{"status": "completed"}` |
| Human intervention | Approval required before continuing |
| Error/retry threshold | `if tool_failed > 3: escalate` |

### Agent vs Workflow

| | Workflow | Agent |
|---|---|---|
| Control flow | Predefined by developer (`A → B → C`) | Model decides dynamically |
| Predictability | High | Lower |
| Adaptability | Low | High |

### When to Use an Agent vs Simpler Alternatives

Autonomy is a **spectrum**: `Deterministic code → LLM chain → RAG pipeline → Tool-using LLM → Agent → Multi-agent`. As you move right: flexibility ↑, but cost, latency, and unpredictability ↑ too.

**Golden rule:** *Use the least autonomous architecture that reliably solves the problem.*

**Use an agent when:**
- The next action depends on intermediate results (tool selection can't be predetermined)
- The environment is uncertain/dynamic and requires adaptation/recovery
- The task is a multi-step goal under uncertainty

**Prefer a chain / RAG / deterministic script when:**
- The workflow is fixed and known in advance
- No real-time decision-making is required
- Reliability, cost, and auditability matter more than flexibility

**Decision framework:**
```
Can plain code solve it?              → Yes → use code
Can one LLM call/chain solve it?      → Yes → use a chain
Can simple RAG solve it?              → Yes → use RAG
Can a single agent + tools solve it?  → Yes → use single agent
Needs specialization/parallelism?     → Yes → multi-agent
```

⚠️ **Interview Trap:** Agentic RAG ≠ "RAG + agent by default." Most retrieval tasks are fine with classic retrieve-then-generate RAG; only add agentic control when retrieval strategy genuinely needs to adapt.

---

## 2. Core Agent Reasoning Patterns

These describe **how a single agent reasons/acts**, as opposed to Section 4 (how *multiple* agents coordinate).

### ReAct (Reason + Act)

**Idea:** Interleave `Thought → Action → Observation` in the same trajectory instead of generating a full answer in one shot.

```
Thought → Action (tool call) → Observation → Thought → Action → ... → Final Answer
```

**Why it works:** Each observation *grounds* the next reasoning step in real data instead of letting the model hallucinate intermediate facts.

```
Thought: I need the current weather in Goa.
Action:  get_weather("Goa")
Observation: 29°C, 15% rain probability
Thought: Rain probability is low.
Final Answer: Umbrella optional.
```

⚠️ **Interview Trap:** Don't say "ReAct eliminates hallucination." Better: *"ReAct grounds reasoning in tool observations, which reduces (but doesn't guarantee against) hallucination — the model can still misinterpret a correct observation."*

⚠️ **Interview Trap:** Modern production systems rarely expose literal `Thought:` text — they use native function-calling with reasoning kept in message history/state. The *pattern* (reason → act → observe → reason) is what matters, not the literal words.

**Strengths:** flexible, good for dynamic/uncertain environments, debuggable trajectory, handles partial failures gracefully.

**Weaknesses:** many LLM calls → high token cost/latency; can get stuck retrying a failing tool; context window growth; reasoning drift from the original goal.

**When NOT to use pure ReAct:** long predictable workflows, strict cost/latency budgets, high-stakes irreversible actions (prefer plan + human approval first).

### Plan-and-Execute

**Idea:** Generate a full (or hierarchical) plan upfront, *then* execute it — rather than deciding the next action after every observation.

```
Goal → PLANNER → Plan [Step1, Step2, Step3...] → EXECUTOR → Output
```

- **Planner** and **Executor** can be different models (strong model plans, cheap model/deterministic code executes) — reduces cost.
- Doesn't mean "no reasoning during execution" — the executor can still retry/adapt locally.

**Advantages:** global view of task, easier progress tracking, plan can be reviewed/approved before side effects, cheaper (strong planner + cheap executor), predictable.

**Disadvantages:** bad initial plan breaks execution, less adaptive than ReAct, requires replanning when environment changes, planning itself costs tokens (don't plan for trivial tasks).

**Hybrid (most practical in production):**
```
Plan → Execute a few steps → Observe → Still valid? ── Yes → Continue
                                                     └── No  → Replan
```

### Reflection / Reflexion / Self-Critique

**Idea:** After generating output, evaluate it against criteria and revise — a *quality-improvement loop*, distinct from ReAct's *decision loop*.

```
Generate → Evaluate/Critique → Good enough? ── Yes → Final
                             └── No → Revise → Evaluate → ...
```

| Pattern | Mechanism |
|---|---|
| **Self-reflection** | Same model generates and critiques itself |
| **Generator–Critic** | Separate model/component critiques |
| **Reflexion** | Reflections are **stored in memory** so *future* attempts improve (doesn't require retraining/weight updates) |

⚠️ **Interview Trap:** Self-reflection has a fundamental weakness — the same model is both generator and judge, so it can rate a wrong answer as "logically consistent." External evaluators (unit tests, compilers, retrieval evidence, human review) are more reliable than pure self-critique.

⚠️ **Interview Trap:** Reflection doesn't always improve output — a revision can score *worse* than the original. Robust systems compare scores and keep the best version, plus cap iterations (`max_reflections` or a quality threshold) to avoid infinite reflection loops.

### Tree-of-Thoughts (ToT)

**Idea:** Explore multiple reasoning paths as a tree, evaluate intermediate states, prune weak branches — instead of following one linear chain.

```
Generate candidate thoughts → Evaluate → Prune bad ones → Expand promising ones → Repeat
```

**Best for:** complex reasoning/search problems with multiple viable approaches (puzzles, planning, combinatorial problems).
**Downside:** expensive — branching factor multiplies LLM calls/tokens.

### ReWOO (Reasoning WithOut Observation)

**Idea:** Generate the full plan of tool calls **before** seeing any results, execute them (potentially in parallel), then reason over all collected observations at once.

```
Plan tool calls → Execute tools (parallel where independent) → Reason over results → Answer
```

**Best for:** predictable, independent tool calls (e.g., calling 5 unrelated APIs) — reduces repeated LLM round-trips and enables parallelism.
**Downside:** less adaptive — if an early tool result should change the plan, ReWOO can't react mid-execution (favor ReAct when actions are *dependent*).

### Hierarchical Planning

**Idea:** A high-level planner decomposes a large goal into sub-goals; sub-planners/agents handle each sub-goal in more detail.

```
Goal → High-level plan → Sub-goals → Sub-plans → Execution (tools/agents)
```

**Best for:** long-horizon, large tasks (e.g., "build and launch an e-commerce platform"). Connects naturally to multi-agent systems (each sub-goal → a specialized agent).
**Downside:** more orchestration, more failure points, errors at the top level can amplify downward.

### Comparison Table

| Pattern | Mental Question | Core Idea | Best For | Main Downside |
|---|---|---|---|---|
| **ReAct** | "What should I do NEXT?" | Interleaved reason→act→observe | Dynamic/uncertain tasks | High token cost, drift, loops |
| **Plan-and-Execute** | "What should I do OVERALL?" | Plan first, then execute | Structured/predictable tasks | Brittle if plan is wrong |
| **Reflection** | "Was my result GOOD ENOUGH?" | Generate→critique→revise | Quality improvement | Critic can be superficial/wrong |
| **Tree-of-Thoughts** | "Which PATH is best?" | Explore & prune branches | Complex search/reasoning | Expensive (branching) |
| **ReWOO** | "Can I plan tools upfront?" | Plan tools → execute → reason | Independent/parallel tool calls | Less adaptive |
| **Hierarchical Planning** | "How do I BREAK this down?" | Multi-level decomposition | Long-horizon complex tasks | Orchestration overhead |

**These are composable, not mutually exclusive.** A real system might use hierarchical planning at the top, ReAct within each sub-task, ReWOO for parallel independent lookups, and reflection for final quality checking.

---

## 3. Tool Use & Function Calling

### How Tool Calling Works

The LLM **does not execute tools itself** — it requests a structured call; the **runtime/application** validates and executes it.

```
LLM: "I want to call get_weather(city='Goa')"
        ↓
Runtime: validate → execute → handle errors/timeout
        ↓
Tool Result → back to LLM as an Observation
```

> **Golden sentence:** *"The LLM decides what tool to call; the runtime is responsible for safely validating and executing that call."*

### Tool Schema Design

A tool schema is effectively **part of the prompt** — treat the description as prompt engineering.

```json
{
  "name": "get_current_weather",
  "description": "Returns current weather for a city. Use when the user asks about current conditions. Do NOT use for historical/multi-day forecasts.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}
```

**Best practices:**
- Clear, descriptive names (`search_customer_orders`, not `search()` or `func1()`)
- Rich descriptions: **when** to use it, **when not to**, what it returns
- Examples inside the description (acts like few-shot guidance)
- Strict types + **enums** for constrained values + required vs optional fields
- **Side-effect warnings** for tools with real-world consequences (`send_email`, `book_flight`, `delete_file`)
- Keep the tool count small — too many tools (e.g. 50+) confuses tool selection; use a **router** to expose only relevant tools

**Never trust the model's arguments blindly** — always run:
1. **Schema validation** (types, required fields, ranges)
2. **Business/authorization validation** (is this account allowed to transfer this amount?)

### Error Handling & Resilience

| Technique | Purpose |
|---|---|
| **Input validation** | Reject malformed/invalid arguments before execution |
| **Output validation** | Type/range/business-logic checks on tool results before trusting them |
| **Timeouts** | Never let the agent wait forever for a tool |
| **Retries + exponential backoff** | Retry *transient* failures (503, timeout) with increasing delay; cap retry count |
| **Circuit breaker** | Stop hitting a repeatedly-failing service (`CLOSED → OPEN → HALF-OPEN`) |
| **Fallback tools / degraded strategy** | Alternative API, cached data, or graceful degradation when primary fails |
| **Sandboxing** | Isolate code execution / shell / browser / filesystem tools in containers with restricted permissions |
| **Feed errors back to the model** | Return explicit error observations (not silent "no results") so the agent can reason about recovery instead of hallucinating success |

⚠️ **Interview Trap:** Don't retry everything — retry *transient* errors (timeouts, rate limits); don't retry *deterministic* errors (invalid API key, bad permissions) since retrying won't fix them.

### MCP (Model Context Protocol)

**Problem it solves:** Without a standard, every framework builds its own custom integration for every tool (GitHub, Slack, Postgres...) → fragmentation/duplication.

**What it is:** An open protocol standardizing how AI applications discover and connect to external tools, data, and context.

```
AI Application → MCP Client → MCP Protocol → MCP Server → [GitHub | Slack | Database]
```

- **Host** = the AI application running the model
- **Client** = component that speaks MCP
- **Server** = exposes standardized capabilities (**tools**, **resources**, **prompts**)

⚠️ **Interview Trap:** "MCP replaces function calling" — wrong. Function calling is the *mechanism* by which a model requests a tool invocation; MCP is the *protocol* standardizing how the application discovers/connects to those tools. They're complementary, not competitors.

MCP does **not** solve security/authorization by itself — you still need auth, permissions, rate limiting, and validation around it.

**One-line map:**
```
Tool Calling → WHICH tool + WHAT arguments?
MCP          → HOW do I connect to tools/context in a standardized way?
```

---

## 4. Multi-Agent Systems

### Why Multi-Agent?

> **Multi-agent ≠ automatically better.** Coordination has real cost.

| Reason | Example |
|---|---|
| **Specialization** | Researcher, Writer, Critic each focus on one job |
| **Parallelism** | Independent subtasks run simultaneously → lower wall-clock latency |
| **Separation of concerns** | Each component easier to reason about/debug |
| **Different tools per agent** | Coding agent has IDE/tests; DB agent has SQL; no agent needs every tool |

> **Rule of thumb:** *Use multi-agent only when roles are clearly distinct and the coordination overhead is justified. Otherwise prefer a well-designed single agent with good tools and memory.*

### Topologies

```
Supervisor/Orchestrator          Sequential/Pipeline        Parallel
      ┌────┼────┐                A → B → C → D          ┌→ A ┐
      ▼    ▼    ▼                                        ├→ B ┤→ Synthesizer
   Researcher Coder Critic                                └→ C ┘

Hierarchical                     Peer-to-Peer / Conversational
     Manager                          A ↔ B
    /       \                         ↕   ↕
 Sub-mgr   Sub-mgr                    C ↔ D
  /  \      /  \
 W    W    W    W
```

| Topology | Description | Trade-off |
|---|---|---|
| **Supervisor + Workers** | Central coordinator routes tasks, orders steps, decides termination | Supervisor can become a latency/token bottleneck |
| **Sequential / Pipeline** | Each agent's output feeds the next | Simple, easy to debug; latency = sum of all stages |
| **Parallel** | Independent agents work simultaneously, then synthesize | Lower latency; risk of duplicated/conflicting work |
| **Hierarchical** | Manager → sub-managers → workers | Scales to very large tasks; more layers = more failure points |
| **Peer-to-peer / Conversational** | Agents message each other directly (AutoGen-style) | Flexible for debate/exploration; can loop endlessly, hard to trace |

### Communication Mechanisms

| Mechanism | Description |
|---|---|
| **Shared state/memory** | All agents read/write one common state object (LangGraph-style) |
| **Message passing** | Agents send messages directly to each other |
| **Blackboard** | Shared workspace agents read/write to incrementally, without direct messaging |
| **Structured handoff** | Agents pass structured JSON (not free text) — easier to validate/debug than prose handoffs |

### When Does Multi-Agent Actually Help?

Ask:
1. Can one agent + tools reliably solve this? → prefer single agent.
2. Are there clearly distinct roles? → multi-agent may help.
3. Can subtasks run independently? → parallel agents reduce latency.
4. Does the benefit outweigh coordination cost? → only then, multi-agent.

### Failure Modes

| Failure | Cause | Mitigation |
|---|---|---|
| Endless arguing (peer-to-peer) | No termination rule | `max_rounds`, supervisor makes final call |
| Duplicated work | Agents unaware of each other's progress | Shared state/task registry |
| Conflicting actions | Two agents modify same resource | Ownership, locking, priority rules |
| Hard-to-trace decisions | Many hops between agents | Tracing, agent IDs, structured logs |
| Higher token burn | Every hop repeats context | Structured handoffs, minimal shared state |

---

## 5. Agent Memory

### The Four Things to Never Confuse

| Concept | Answers |
|---|---|
| **Short-term memory** | "What is happening right now?" |
| **Long-term memory** | "What should I remember for future tasks?" |
| **RAG** | "What external information is relevant to this query?" |
| **Checkpointing** | "Where was my workflow when it stopped?" (execution/system state, not knowledge) |

### Short-Term / Working Memory

The information currently in the LLM's context window: recent messages, current goal, latest tool observations, intermediate state.

⚠️ **Interview Trap:** "Short-term memory means the LLM remembers things" — wrong. LLMs are stateless between calls; the **application** maintains and re-sends the context every call.

**Implementation approaches:** simple conversation buffer → sliding window (keep last N) → token-limited buffer (keep what fits a budget) → summarization of older turns.

**Limitation:** bounded by context window; cost grows with length; doesn't persist across sessions; important info can get "buried" even if technically still in context.

### Long-Term Memory: Three Types

> **Episodic = what happened. Semantic = what is true. Procedural = how to do it.**

| Type | Stores | Example | Typical Implementation |
|---|---|---|---|
| **Episodic** | Specific past experiences/trajectories and outcomes | "Last time, DFS gave a TLE on this problem type — use BFS instead" | DB/vector store of past trajectories, retrieved by similarity |
| **Semantic** | General facts, preferences, domain knowledge (not tied to one event) | "User prefers morning flights" | Vector DB + embeddings (RAG-style), or knowledge graph for relationships |
| **Procedural** | Skills / how-to workflows / tool-use strategies | "Always check cancellation policy before booking" | Text skills, workflow definitions, code, or (rarely) fine-tuning |

### Implementation Techniques

| Technique | Purpose |
|---|---|
| **Summarization/compression** | Periodically compress older history into a summary; trades detail for context efficiency |
| **Vector stores** (Pinecone, Weaviate, Chroma, FAISS, LanceDB, pgvector) | Store embeddings, retrieve by semantic similarity |
| **Knowledge graphs** | Nodes/edges for entities+relationships; needed for multi-hop/structured queries ("which employees worked on X before announcement") |
| **Hybrid retrieval** | Dense (embeddings, semantic) + sparse (BM25, exact keyword match, e.g. error codes) combined |
| **Reranker** | Re-scores/reorders retrieved candidates for better relevance (retriever finds candidates; reranker improves ordering) |
| **Checkpointing/state persistence** | Save execution state so a long-running agent can resume after crash (system-level, not knowledge) |
| **Memory write strategies** | Decide *what* to store: store everything (noisy) vs. importance filtering vs. LLM-decides vs. importance scoring vs. reflection-based writing |

```
Dense (embeddings) ──┐
                      ├─→ Merge → Reranker → Top-k → LLM
Sparse (BM25) ────────┘
```

### Challenges

| Challenge | Description |
|---|---|
| Context window limits | Forces truncation/summarization |
| Forgetting / retrieval failure | Memory exists but wasn't retrieved when needed |
| Retrieval quality | Irrelevant or wrong-but-similar memories mislead the agent — **bad memory can be worse than no memory** |
| Cost | Storage + embedding + retrieval + added LLM context tokens all cost money |
| Staleness/consistency | Old preferences contradict new ones; needs update/versioning/recency-weighting |
| Privacy & security | Long-term memory may hold PII/secrets — needs access control, retention policy, deletion |
| Memory pollution | A hallucinated fact gets written to memory and becomes "true" for future turns — mitigate with importance filtering, confidence scores, provenance tracking |
| Management complexity | What to store / update / forget / resolve-conflicts-between is a decision problem, not just a DB problem |

---

## 6. Agentic RAG

| | Classical RAG | Agentic RAG |
|---|---|---|
| Flow | `Question → Retrieve once → Generate` | Agent decides *whether*, *what*, *from where*, and *how many times* to retrieve |
| Control | Fixed pipeline | Model controls the retrieval loop |
| Best for | Single-hop, predictable queries | Complex, multi-hop, ambiguous, or dynamic queries |
| Cost | Lower | Higher (more LLM + retrieval calls) |

```
Question → Agent reasons → Retrieve → Observe → Sufficient? ──No──→ Retrieve again
                                                             └─Yes─→ Answer
```

**Multi-hop retrieval:** each retrieval step depends on the previous result.
```
Find company → find its acquirer → find acquirer's CEO → find CEO's recent statement
```

**Self-RAG:** the model reflects on whether retrieved evidence (or its own draft answer) is actually sufficient, and retrieves again rather than confidently answering on weak evidence. It was designed specifically to improve retrieval + generation quality. The whole system is optimized around retrieval quality and grounded generation.

Self-RAG can do:
```
✓ Decide whether retrieval is needed
✓ Evaluate retrieved information
✓ Evaluate grounding
✓ Revise generation
✓ Potentially retrieve again
```

But Self-RAG does not inherently mean:

```
✗ Planning arbitrary workflows
✗ Calling multiple unrelated tools
✗ SQL queries
✗ API calls
✗ Calculator
✗ Sending emails
✗ Executing actions
✗ Multi-agent collaboration
```

Those are agentic capabilities. Its focus is autonomous problem solving.

Agentic RAG is a broader orchestration pattern, while Self-RAG is a specific approach focused on adaptive retrieval and self-critique. It can incorporate Self-RAG techniques, but Agentic RAG doesn't inherently require them.

⚠️ **Interview Trap:** "Agentic RAG = multiple retrievals" is incomplete. The defining feature is that the **model controls the retrieval strategy** (deciding whether/what/when to retrieve), not merely that retrieval happens more than once.


---

## 7. Agent Frameworks

### LangChain vs LangGraph

**LangChain** = building blocks/integrations for LLM apps (prompts, models, chains, tools, retrievers, RAG components, agent abstractions).

**LangGraph** = explicit graph-based orchestration for **stateful, cyclic** agent workflows.

| Concept | Meaning |
|---|---|
| **Node** | Unit of work (LLM call, tool call, validation, human approval) |
| **Edge** | Where execution goes next (can be conditional/looping) |
| **State** | Shared data persisted and updated across the whole workflow |
| **Checkpoint** | Persisted snapshot of state so execution can resume after failure |
| **Durable execution** | Workflow survives interruption and resumes from persisted state |

```
START → Agent Node → Tool required? ──Yes──→ Tool Node → update state ─┐
                                └──No──→ END                            │
                    ▲──────────────────────────────────────────────────┘
```

**Why LangGraph over a plain LangChain agent:** explicit control over branching, loops, retries, checkpointing, and human-in-the-loop interruptions — essential once workflows aren't linear.

> *"I'd use LangGraph when I need explicit state, non-trivial control flow (branching/cycles), and the ability to persist/resume a long-running agent workflow."*

### CrewAI

**Role-based multi-agent framework.** Core concepts: `Agent (role, goal, backstory, tools)`, `Task (what work)`, `Crew (team)`, `Process (Sequential / Parallel / Hierarchical)`.

- **Opinionated & high-level** — very fast to prototype role-based teams (Researcher + Writer + Critic).
- Less fine-grained control/observability than LangGraph.
- Everything CrewAI does *can* be built in LangGraph — CrewAI just gives a ready-made abstraction (`WHO does WHAT, HOW`) instead of raw primitives (state/nodes/edges).

### AutoGen

**Conversational multi-agent framework** — agents coordinate through **messages**, not a predefined graph or fixed role/task list.

- Strong for dynamic, multi-turn, exploratory collaboration and natural human-in-the-loop participation (a human can join the chat).
- More free-form than CrewAI's structured roles or LangGraph's explicit graph → harder to constrain, debug, and control cost.

### The Three-Framework Mental Model

| Framework | Primary abstraction | Ask yourself |
|---|---|---|
| **LangGraph** | State + Nodes + Edges | "How does my workflow execute?" |
| **CrewAI** | Roles + Tasks + Crew + Process | "Who are my agents and what's their job?" |
| **AutoGen** | Messages / conversation | "How do my agents talk to each other?" |

> **Memory trick:** CrewAI = *Team*. AutoGen = *Conversation*. LangGraph = *Workflow*.

### Other Frameworks / Platforms (name-drop level)

| Platform | Strength |
|---|---|
| **LlamaIndex** | RAG-heavy / data-intensive agents (indexing, retrieval) |
| **Semantic Kernel** (Microsoft) | Enterprise, Microsoft ecosystem, planner + plugins/skills |
| **Google ADK / Vertex AI Agents** | Google Cloud/Gemini-native agent building & deployment |
| **AWS Bedrock Agents** | Managed cloud agents with built-in tools + knowledge bases |
| **Salesforce Agentforce** | Enterprise CRM/customer-facing agents |

### Named Pattern Collections (Recognition Level)

Two well-known ways people categorize the same underlying ideas:

**Andrew Ng's four agentic patterns:** Reflection, Tool Use, Planning, Multi-Agent Collaboration.

**Anthropic's workflow patterns:** Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer.

| Anthropic pattern | Description | Closest mapping |
|---|---|---|
| **Prompt Chaining** | `LLM1 → LLM2 → LLM3`, fixed sequence | Deterministic workflow (not really "agentic") |
| **Routing** | Classify input → send to ONE specialized path | Tool/workflow selection (not ongoing orchestration like a Supervisor) |
| **Parallelization** | Independent work run simultaneously, then combined | Multi-Agent Collaboration |
| **Orchestrator-Workers** | Manager decomposes → delegates → collects → synthesizes | = Supervisor/Worker topology; Multi-Agent + Planning |
| **Evaluator-Optimizer** | Generate → evaluate → improve | = Reflection |

⚠️ **Interview Trap:** These are *conceptual mappings*, not strict 1:1 equivalences — e.g., Prompt Chaining isn't inherently agentic at all. If an interviewer describes an unnamed architecture, practice recognizing it: *"This looks like an Evaluator-Optimizer pattern because one component generates, another evaluates, and feedback drives revision."*

---

## 8. Reliability, Safety & Security

### Common Failure Modes

| Failure | What happens | Mitigation |
|---|---|---|
| **Infinite loops** | Agent keeps acting without progress | Max iterations, token budget, detect repeated identical tool calls |
| **Tool misuse** | Wrong tool, wrong arguments, unnecessary calls | Clear schemas, descriptions, examples, input validation |
| **Goal drift** | Agent optimizes an easier proxy and forgets original constraints | Keep goal explicit in state; validate final result against original constraints |
| **Compounding errors** | An early small mistake cascades through later steps | Validation checkpoints between steps, verify critical intermediate results |
| **Prompt injection** | Malicious instructions from user or *untrusted content* (webpages, docs, tool output) hijack behavior | Treat external content as untrusted data, least privilege, human approval |
| **High cost/latency** | Too many LLM/tool calls | Budgets, caching, model routing, early stopping |
| **Non-determinism** | Same input → different trajectories across runs | Lower temperature where appropriate, mocked/fixed test environments, trajectory replay |
| **Hallucinated tool results** | Model invents an observation instead of using the real one | Strict separation of tool output vs. model text; validate outputs |
| **Premature stopping** | Agent stops before satisfying all constraints | Termination checklist against original goal/constraints |
| **Ignoring observations** | Agent receives correct data but acts as if it didn't | Structured tool outputs; evaluation/reflection catches this |
| **Privilege escalation via tools** | Legitimate tools *combined* grant unintended capability | Least privilege, per-resource authorization, sandboxing |

> **The core distinction:** a normal LLM usually fails at *generation*; an agent can fail at *any point* in a multi-step trajectory (planning, tool selection, arguments, interpreting observations, recovery, termination, authorization) — and these errors **compound**. This is why agents need stronger validation, observability, and evaluation than single-shot LLM calls.

### Goal Misalignment / Specification Gaming

> **The agent satisfies the literal metric/specification while violating the actual human intention.**

Classic examples:
- "Maximize sales" → spam customers with unrealistic discounts
- "Resolve tickets fast" → close tickets without solving them
- "Book the cheapest flight" → picks a 30-hour, 4-layover itinerary

**Why it happens:** humans leave constraints implicit (common sense); metrics are easier to optimize than actual intentions.

**Mitigation — separate objective from constraints:**
```
Objective:      minimize price
Hard constraints:  arrival before 8 PM, ≤1 layover     ← must never be violated
Soft preferences:  prefer morning departure             ← nice-to-have
```

Also connects to **reward hacking** (same idea in RL terms): the system finds an unintended loophole that maximizes the reward signal without achieving the true goal.

⚠️ **Interview Trap:** Goal misalignment (bad/incomplete objective specification) is a different failure from **goal drift** (a correct objective gets lost/diluted *during* execution). They can co-occur, but they're distinct causes.

### Security: Three Connected Attack Vectors

```
1. Prompt Injection  →  2. Privilege Escalation  →  3. Data Exfiltration
   (malicious webpage)     (read secret + email)      (secret leaves system)
```

| Attack | Description | Mitigation |
|---|---|---|
| **Prompt Injection** (direct/indirect) | Malicious instructions from the user *or* from retrieved/untrusted content (webpage, email, tool output) override agent behavior. **Indirect** injection is more dangerous for agents because it enters via *data* the agent processes, not a direct chat message. | Treat external content as untrusted data (never as instructions); least privilege; policy/authorization checks before high-impact tool calls; human approval for risky actions |
| **Tool Privilege Escalation** | Individually-safe tools *combined* let the agent do something never intended (e.g., `read_file(secret)` + `execute_code()` → exfiltrate) | Least privilege, separate credentials per tool, sandboxing, tool-level authorization (not just "can it call X" but "can it access *this* record/destination") |
| **Data Exfiltration** | Sensitive data leaves its authorized boundary (email, HTTP request, file upload) — can be attacker-triggered *or* accidental agent inference | Data classification, destination checks before sending data externally, DLP-style content scanning, least privilege |

> **Core principle:** *Never rely solely on the LLM to behave correctly.* Critical safety controls must live **outside** the model, at the runtime/infrastructure level — an instruction telling the LLM "don't delete files" is a *soft* control; runtime authorization that blocks the call is a *hard* control.

### Safeguards & Production Controls (Defense in Depth)

| Layer | Mechanism |
|---|---|
| **Bounded autonomy** | Agent acts freely *within* limits (tools, budget, time, steps). Autonomy is a spectrum: **Supervised** (approve every action) → **Collaborative** (routine auto, risky → human) → **Delegated** (full workflow, still bounded) |
| **Hard limits** | Max steps, token budget, time budget, cost cap |
| **Human-in-the-loop / approval gates** | Required for payments, deletions, irreversible actions, external comms — often **threshold-based** (e.g. auto-approve <₹1,000) |
| **Least privilege** | Every tool/agent gets only the minimum permissions it needs |
| **Guardrails** | Input guardrails (detect malicious/unsafe input), output guardrails (block leaking secrets), allowed-tool lists, policy checks |
| **Sandboxing** | Isolate code execution/shell/browser/filesystem tools in restricted containers |
| **Audit logs / tracing** | Full trajectory logged (decision → tool → args → result) — must be reconstructible after the fact, without necessarily storing raw private chain-of-thought |
| **Kill switches / credential revocation** | Instant ability to stop an agent and revoke its access — works *outside* the LLM, doesn't depend on the model "deciding to stop" |
| **Monitoring & alerting** | Real-time detection of loops, cost spikes, repeated failures, policy violations |
| **Supervisor/meta-agent** | A separate component that can pause/replan/escalate — but is not a silver bullet (itself needs the same controls) |

---

## 9. Observability & Evaluation

### Observability

> Agents can take 10–20 actions before a final answer — you cannot debug by looking only at the final output. You need the **full trajectory**.

**What to capture per step:** LLM call (input/output/tokens/latency), tool call (name/args/start-end time/result-or-error), state before/after, final answer.

```
Logs    = individual events ("tool X called")
Metrics = numerical aggregates (avg latency, error rate, avg cost)
Traces  = the entire multi-step request journey
```

**Replayability:** ability to re-run a past trajectory to reproduce a bug — harder for agents than pure functions because external state (web, APIs, time) may have changed since the original run; mock external dependencies in test environments.

**Tools to name:** LangSmith, Langfuse, Helicone, Phoenix (Arize), or custom logging + OpenTelemetry-style tracing.

⚠️ Don't log everything carelessly — full trajectories can capture PII/secrets; apply redaction, access control, and retention policies. Prefer logging **structured decision metadata** (`selected_tool`, `reason`) over raw private chain-of-thought.

### Trajectory Evaluation vs Final-Answer Evaluation

> **Final answer evaluation** only checks the last output. **Trajectory evaluation** examines the entire path — tool choices, arguments, error recovery, efficiency, safety.

An agent can reach the *correct* answer via a *bad* trajectory:
- Wrong tool first, then corrected (wasted cost)
- "Lucky correctness" — random pick among conflicting sources happened to be right
- Booked correctly, but charged the card *before* asking for confirmation (safety violation)

**Evaluate:** final correctness · tool selection · argument validity · error recovery · efficiency (steps/tokens/cost) · safety (forbidden actions?) · appropriate termination.

> **Interview line:** *"Evaluating only the final answer is insufficient for agents — I always look at the full trajectory, because an agent can reach the right answer through an unsafe, wasteful, or non-reproducible path."*

### Evaluation Methods & Datasets

```
Build golden dataset → Run agent → Evaluate (rules + LLM judge + human) →
   Find failures → Improve prompts/tools/guardrails → Re-run
```

| Method | Description |
|---|---|
| **Golden dataset** | Curated test cases with expected outcomes *or* expected behavior/constraints (not always one exact trajectory — multiple valid paths can satisfy a rubric). Grows from real production failures (regression testing). |
| **LLM-as-judge** | Strong LLM scores trajectories/answers against a rubric. Cheap & scalable, but can be biased, inconsistent, or overly generous — needs a clear rubric and calibration against humans. |
| **Human evaluation** | Gold standard for nuanced/high-stakes cases; used to validate/calibrate the automated judge, not to review everything forever. |
| **Deterministic checks** | Rules/unit tests for objective things (correct tool called? valid arguments?) |

**Benchmarks to recognize (not memorize scores):**

| Benchmark | Tests |
|---|---|
| **GAIA** | General-purpose agent capability |
| **AgentBench** | Agent performance across diverse environments |
| **ToolBench** | Tool selection/use |
| **WebArena** | Web navigation/task completion |
| **SWE-bench** | Coding agents resolving real repo issues |

⚠️ **Interview Trap:** Evaluation isn't just accuracy — compare agents on correctness **+** cost **+** latency **+** safety **+** tool efficiency together; a 1% accuracy gain isn't worth 200× the cost.

---

## 10. Cost & Latency Control

| Technique | Effect |
|---|---|
| **Max steps / max iterations** | Hard cap prevents infinite loops / runaway cost |
| **Token budgets** | Per-run, per-user, or per-tool-call limits |
| **Caching** (tool results, LLM responses, embeddings) | Avoid redundant calls; needs a **TTL** to avoid serving stale data |
| **Parallel/speculative execution** | Run independent calls simultaneously to cut latency (don't parallelize *dependent* or side-effecting calls) |
| **Model routing** | Cheap/fast model for simple steps (classification, extraction); strong model for complex planning/reasoning |
| **Early stopping** | Stop as soon as the goal is achieved, don't burn remaining budget |
| **Rate limiting** | Cap requests/period to a shared or external service |
| **Concurrency limits** | Cap simultaneous operations against a resource (e.g., DB) |

> Cost/latency optimization isn't "make everything cheaper" — it's *balancing* quality, latency, reliability, and cost for the application's actual requirements. Some techniques help both (caching); others trade one for the other (speculative execution cuts latency but can raise cost).

---

## 11. Enterprise & Organizational Concerns

| Concern | Question it answers |
|---|---|
| **Compliance & regulatory** | Are we legally allowed to do this? (GDPR, SOC 2, industry-specific rules for finance/healthcare/HR) |
| **Auditability** | Can we reconstruct exactly what the agent did and why, after the fact? (Log tool calls, args, results, policy decisions — not necessarily raw chain-of-thought) |
| **Multi-agent conflict resolution** | What happens when two agents modify the same resource? → Ownership, locking, priority rules, or a supervisor |
| **Governance** | Who can create/modify agents and tools? How are new tools reviewed? How are incidents handled? → Versioning of prompts/tools/policies, security review pipelines, incident response process |

---

## 12. Classical AI Agents vs Modern LLM Agents

| | Classical / GOFAI Agents | Modern LLM Agents |
|---|---|---|
| Reasoning | Rules, symbolic planning (STRIPS/HTN), expert systems, **BDI** (Beliefs–Desires–Intentions) | LLM as the reasoning engine + tools + memory + loop (ReAct etc.) |
| Input | Structured | Natural language / unstructured |
| Behavior | Deterministic, predictable | Probabilistic |
| Explainability | Easier | Harder |
| Flexibility/generalization | Low — brittle outside programmed cases | High — handles ambiguity and novel phrasing |
| Failure mode | Brittle rules | Hallucination, tool misuse, goal drift |

> **The one-sentence trade-off:** *Classical agents trade flexibility for predictability; modern LLM agents trade some predictability for flexibility and generalization* — which is exactly why modern agents need the safety layer covered in Section 8.

**They combine well in production:** LLM handles flexible interpretation/reasoning; a deterministic policy/authorization layer enforces hard guarantees before any tool executes.
```
LLM: "I want to do X" → Deterministic policy layer: "Are you allowed to do X?" → Tool
```

---

## 13. Real-World Agent Examples

| Category | Description |
|---|---|
| **Coding agents** (Devin-style) | Inspect a repo, write/modify code, run tests, debug failures, submit a PR — essentially ReAct + tools + code execution + state + evaluation |
| **Research agents** (Auto-GPT lineage) | Decompose a research goal, search/browse iteratively, cross-check sources, synthesize a report — Planning + Agentic RAG + Reflection |
| **Customer support / workflow agents** | Read tickets, retrieve knowledge + order/account data, update CRM, escalate above their authority threshold |
| **Booking / personal assistant agents** | Search & compare under constraints (budget, dates, preferences); **searching is reversible, booking is not** → approval gate before purchase |
| **Enterprise agents** (e.g. Agentforce-style) | Integrate with CRM/business systems; the hard part is controlled action + integration, not the chat itself |
| **Browser-using agents** | Click/type/navigate real websites instead of clean APIs — unpredictable environment → needs stronger sandboxing, injection defenses, and confirmation steps |

All of these reduce to the same underlying architecture: **LLM + tools + state/memory + decision loop + constraints**, applied to a different domain.

---

## 14. Agent System Design Framework

When asked to design an agent (e.g. *"Design an agent that books a family trip to Goa under ₹80,000"*), walk through these **9 steps**:

```
1. Goal / success criteria         → What exactly counts as "done"? Constraints?
2. Agent loop                      → Observe → Reason → Act → Observe → Replan
3. LLM / brain                     → Strong model for planning; cheap model for simple steps
4. Tools                           → List them; separate READ (safe) vs WRITE (side-effecting)
5. State / memory                  → Short-term (current trip) vs long-term (user preferences)
6. Planning strategy                → ReAct / Plan-and-Execute / Hybrid — justify the choice
7. Termination conditions          → Explicit success criteria AND hard safety limits
8. Failure recovery & safety       → Retries, fallbacks, human approval before irreversible actions
9. Observability & evaluation      → What to log; test cases including edge cases & failures
```

**Key framing points that impress interviewers:**
- Separate **read-only** tools (safe, autonomous) from **write/side-effecting** tools (require confirmation)
- Distinguish **hard constraints** (budget ≤ ₹80,000 — never violate) from **soft preferences** (prefer morning flights)
- Don't send every raw tool result back to the LLM — filter/summarize locally, only pass the top candidates
- Mention a small test set covering normal cases, edge cases, tool failures, and safety-sensitive cases — not just "does the final answer look good"

---

## 15. Common Interview Questions

- What makes an AI system "agentic," as opposed to just calling an LLM once?
- Walk me through how an agent actually works, end to end.
- What's the difference between ReAct and Plan-and-Execute? When would you choose one over the other?
- Why can ReAct require more LLM calls than Plan-and-Execute?
- What is Reflexion, and does it require fine-tuning the model?
- How do agents decide which tool to use, and how do you improve tool selection?
- How do you prevent an agent from entering an infinite loop or making excessive tool calls?
- What's the difference between final-answer evaluation and trajectory evaluation, and why does it matter for agents?
- How would you evaluate an agent you built?
- When should you use a multi-agent system instead of a single agent?
- RAG vs fine-tuning — and RAG vs agentic RAG?
- How do you manage/design memory for an agent (short-term vs long-term, episodic/semantic/procedural)?
- How do you defend against (indirect) prompt injection?
- What safeguards would you put around a production agent?
- When would you use an agent vs. a simple chain or RAG pipeline?
- How is LangGraph different from a simple LangChain agent? From CrewAI? From AutoGen?
- What is MCP, and how is it different from function calling?
- Design an agent for [booking / customer support / research] — walk through your architecture.

---

## 16. Interview Traps

> ⚠️ An LLM that can call a tool is not automatically a fully autonomous agent — agentic behavior also requires state, iteration, decision-making, and a termination condition.

> ⚠️ ReAct does **not** eliminate hallucination — it grounds reasoning in observations, but the model can still misinterpret or ignore a correct observation.

> ⚠️ Don't assume modern agents expose literal `Thought:`/chain-of-thought text — production systems typically use native function calling with reasoning kept in structured state, not necessarily shown to the user.

> ⚠️ Self-reflection has a blind spot: the same model is both generator and judge, so it can rate its own wrong answer as fine. External evaluators are more reliable.

> ⚠️ Reflexion does not require retraining the model — it stores feedback in memory/context, not in model weights.

> ⚠️ "Multi-agent" is not inherently smarter than "single agent + good tools" — more agents can mean more cost, more latency, and harder debugging for no accuracy gain.

> ⚠️ Agentic RAG isn't just "retrieve more than once" — the defining trait is that the model *controls* the retrieval strategy (whether/what/when).

> ⚠️ MCP does not replace function calling, and it doesn't solve security/authorization by itself.

> ⚠️ Correct final answer ≠ good agent behavior — a trajectory can be unsafe, wasteful, or "luckily correct." Always evaluate the full trajectory, not just the output.

> ⚠️ Never rely on a prompt instruction alone for safety (e.g., "don't delete files") — critical controls (authorization, permissions, sandboxing) must live outside the model, at the runtime/infrastructure level.

> ⚠️ Goal drift (losing track of a correct objective mid-execution) and goal misalignment/specification gaming (the objective itself was incomplete) are different failures with different fixes.

---

## 17. Quick Revision Cheat Sheet

### Core Concepts

| Concept | One-Line Definition |
|---|---|
| **Agent** | An LLM in a loop that observes, decides, acts, and stops based on a termination condition |
| **Tool** | An external capability (API/function) the LLM can request; the runtime executes it |
| **State** | The agent's persisted working data across steps (broader than conversation history) |
| **Short-term memory** | The current context window |
| **Long-term memory** | Persistent knowledge across sessions (episodic/semantic/procedural) |
| **Planning** | Deciding the sequence of steps needed to reach a goal |
| **Reflection** | Critiquing and revising one's own output against criteria |
| **RAG** | Retrieve relevant external info, then generate an answer using it |
| **MCP** | A standardized protocol for AI apps to connect to tools/data/context |
| **Checkpointing** | Persisting execution state so a workflow can resume after failure |
| **Guardrail** | A rule constraining what an agent can receive, produce, or do |

### Architecture Patterns

| Pattern | Key Idea |
|---|---|
| **ReAct** | Reason → Act → Observe, interleaved, incremental |
| **Plan-and-Execute** | Plan the whole sequence upfront, then execute |
| **Reflection** | Generate → critique → revise loop |
| **Tree-of-Thoughts** | Explore & prune multiple reasoning branches |
| **ReWOO** | Plan all tool calls first, execute (parallel), reason once at the end |
| **Hierarchical Planning** | Decompose a big goal into sub-goals/sub-plans |
| **Multi-Agent (Supervisor)** | Central coordinator delegates to specialized workers |

### Important Comparisons

| A | B | Key Difference |
|---|---|---|
| Agent | LLM call | Agent has a loop, state, and multi-step decision-making; LLM call is one-shot |
| Agent | Workflow | Agent = model-driven control flow; Workflow = developer-predefined sequence |
| RAG | Fine-tuning | RAG injects external knowledge at query time (fresh, cheap to update); fine-tuning bakes behavior/knowledge into weights (static, expensive to update) |
| RAG | Agentic RAG | RAG retrieves once in a fixed pipeline; Agentic RAG lets the model control the retrieval loop |
| ReAct | Plan-and-Execute | ReAct decides locally after each observation; Plan-and-Execute commits to a global plan upfront |
| Reflection | Planning | Reflection asks "was this good enough?"; Planning asks "what should I do?" |
| Single-agent | Multi-agent | Multi-agent only worth it when roles are distinct and coordination benefit > overhead |
| Short-term memory | Long-term memory | Short-term = current context window; Long-term = persists across sessions |
| Tool calling | MCP | Tool calling = which tool + what args; MCP = standardized connection protocol |
| LangGraph | CrewAI | LangGraph = explicit state/graph control; CrewAI = high-level role/task/team abstraction |
| CrewAI | AutoGen | CrewAI = structured roles/tasks; AutoGen = free-form agent conversation |
| Observability | Evaluation | Observability = what happened; Evaluation = was it good |
| Goal drift | Specification gaming | Goal drift = correct goal lost during execution; Specification gaming = the goal itself was incomplete |

### Production Safety — 6 Layers to Remember

```
BOUND    → limit steps/tokens/time/cost
PERMIT   → least-privilege tools
APPROVE  → human approval for risky actions
ISOLATE  → sandbox dangerous tools
OBSERVE  → logs + monitoring + tracing
KILL     → kill switch + credential revocation
```

### The 9-Step Design Framework

```
Goal → Agent Loop → Brain (LLM) → Tools → State/Memory →
Planning Strategy → Termination → Safety/Recovery → Observability/Evaluation
```
