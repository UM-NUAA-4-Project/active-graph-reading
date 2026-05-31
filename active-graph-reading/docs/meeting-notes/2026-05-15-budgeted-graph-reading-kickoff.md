# Meeting Notes: 2026-05-15 - Active Graph Reading Direction

**Date**: 2026-05-15  
**Participants**: 瞿锰 (Meng), 黄智辉  
**Context**: Weekly sync. The project direction was narrowed from general graph classification / security graph ideas to active / budgeted graph reading for graph classification.

## Summary

Meng proposed a more concrete research direction: instead of assuming the full graph is available for graph classification, study how a model can actively reveal useful graph evidence under a fixed budget before making a graph-level prediction.

The key analogy is LLM / coding agents. An agent working with a large codebase does not read every file. It searches, retrieves, inspects, and accumulates useful evidence. A graph classifier may need a similar reading strategy when the graph is large, noisy, or only partially observable.

## Proposed Problem

The model starts from a partial graph observation `G_0`. At each step, it chooses what graph evidence to reveal next:

```text
Start from G_0
for b = 1...B:
  choose what node / edge / evidence to reveal next
predict graph label from the revealed partial graph
```

The goal is to predict the graph label accurately under a fixed observation budget.

## Why This Direction

- Plain graph classification is already crowded.
- Active / budgeted graph reading gives the project a clearer and more recent angle.
- The direction connects graph ML with evidence acquisition and agent-style context retrieval.
- It can start from simple public graph-classification benchmarks before moving to more domain-specific graphs.

## Initial Method Intuition

Start with simple policies that generate graph-reading trajectories:

- random reveal;
- BFS / neighborhood expansion;
- degree or centrality heuristic;
- task-informed heuristic or oracle upper bound if feasible.

Then score trajectories by final prediction quality or information gain. If some trajectories are clearly better, train a model to imitate useful reading behavior.

## Immediate Guidance

智辉 should not start by coding a complex model. The first task is to understand and formalize the problem:

1. Read the project proposal with an LLM.
2. Ask the LLM to generate toy examples of the budgeted graph-reading process.
3. Map related-work categories and explain how this project differs from each.
4. Identify a suitable first benchmark dataset and likely setup blockers.

## Decisions

1. The concrete candidate direction is **active / budgeted graph reading for graph classification**.
2. The first stage is problem understanding and related-work grounding, not model engineering.
3. The project should use repo-first collaboration: project docs, meeting notes, progress logs, and experiments should live in the repo.
4. The initial benchmark should be simple and public, likely OGBG-style.

## Action Items

| Who | Action | Timing |
|---|---|---|
| Meng | Upload project documents and meeting note to the repo | After meeting |
| 智辉 | Read the proposal with an LLM and write toy examples | Next few days |
| 智辉 | Summarize related-work categories and differences | Before next sync |
| 智辉 | Identify first benchmark candidates and setup blockers | Before first coding pass |
| Meng + 智辉 | Decide first dataset and baseline protocol | Next phase |
