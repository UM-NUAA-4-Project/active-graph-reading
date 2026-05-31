# Decisions

This file is an append-only record of project decisions.

## 2026-05-15

### Direction

**Decision**: Use active / budgeted graph reading as the concrete project direction.

**Rationale**: Plain graph classification is crowded. The budgeted reading formulation gives the project a clearer research angle: learn which graph evidence to reveal before graph-label prediction, analogous to how LLM / coding agents retrieve context.

**Implications**:

- The first stage should not start from a complicated new GNN architecture.
- The first deliverables are toy examples, related-work contrast, and a benchmark protocol.
- Evaluation should focus on performance under matched observation budget.

### Initial method style

**Decision**: Start from simple reveal policies and trajectory distillation before more complex methods.

**Rationale**: Random, BFS, centrality, and heuristic policies make the problem concrete and provide sanity-check baselines. If there is no headroom over these baselines, a learned policy is unlikely to be persuasive.

**Implications**:

- Build the reveal process first.
- Plot budget-performance curves early.
- Only add a learned policy after the benchmark shows meaningful headroom.

### Collaboration workflow

**Decision**: Keep project documents, meeting notes, progress logs, and later experiment summaries in this repo.

**Rationale**: Repo-first collaboration makes asynchronous supervision easier. A coding agent or LLM can read the repo and recover project status quickly.

**Implications**:

- Use `docs/meeting-notes/` after each sync.
- Use `docs/progress/daily/` for short every-1-2-day updates.
- Use `docs/decisions.md` for decisions that should be easy to recover later.

## 2026-05-18

### Benchmark scope

**Decision**: Treat MUTAG / molecular graphs as toy illustrations only. The first real benchmark uses OGBG-PPA, or another graph-classification dataset with comparable per-graph scale (hundreds of nodes, thousands of edges per graph).

**Rationale**: The project motivation is active reading under a limited observation budget on large, noisy, or expensive-to-inspect graphs. Small molecular graphs are usually cheap to inspect fully, so they do not strongly support the core premise.

**Note on terminology**: "PPI" in common usage refers to the inductive PPI dataset, which is a node-classification benchmark. Our task is graph classification, so we anchor on OGBG-PPA, not PPI.

**Implications**:

- MUTAG examples can remain in progress notes as explanation material only.
- M3 targets OGBG-PPA unless setup is unexpectedly blocked, in which case the substitute dataset must also satisfy the scale criterion above and be documented.
- The first report should include per-graph node/edge distributions to justify why active graph reading is needed.

### Cost-model scope for v0

**Decision**: Defer the full atomic cost model. For v0, use a single reveal action (`reveal_node`) at unit cost.

**Relation to 5/17 atomic cost model**: The 5/17 4-atomic-action cost model is retained as design scaffolding for M4 and later. v0 is intentionally narrower so we can get a measurement surface running quickly. Richer actions and cost accounting are added back once v0 budget-performance curves are reproducible.

**Rationale**: A comprehensive action/cost formalism is more useful once we have empirical evidence about which actions move the budget-performance curve. Designing it earlier risks optimizing the formalism against intuition rather than data.

**Implications**:

- v0 implementation uses one reveal action at unit cost: `reveal_node(v)` reveals v as an observed node, the edge features on edges from v to already-revealed nodes, and v's incident edge list / neighbor IDs so the frontier can expand. Features of newly discovered neighbors remain hidden until those nodes are revealed.
- Budget is measured as number of revealed nodes, also reported as a fraction of total nodes per graph.
- Initial policy comparison: uniform random over frontier, BFS from seed, max known-frontier degree. Report averages over multiple random seeds per graph because k = 1 seed choice can dominate early performance.

### v0 initial observation

**Decision**: For v0, `G_0` is Plan D with k = 1 — a single randomly chosen seed node, with incident edge list / neighbor IDs visible. Raw node features should be treated as absent for OGBG-PPA unless the dataset check shows otherwise. The set of all node IDs is **not** revealed up front.

**Rationale**: Plan A ("all node IDs visible") is borderline at PPA scale and arguably itself an unrealistic free read. Plan D with a small seed matches the real "start from one entry point" regime, and is a cleaner starting condition for sequential expansion policies (BFS, frontier-degree).

**Implications**:

- Policies operate on the *known frontier* (neighbors of revealed nodes that are themselves not yet revealed); they do not see the full node set.
- "Frontier degree" used by the max-degree policy is the count of edges from a frontier node back into the revealed subgraph, not true graph degree (which is hidden).
- OGBG-PPA's useful observed features are expected to be edge features, not node features, so v0 should avoid assuming raw node attributes.
