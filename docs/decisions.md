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
