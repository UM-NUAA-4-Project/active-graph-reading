# Roadmap

This roadmap tracks the near-term execution plan for Active / Budgeted Graph Reading.

## Current Phase

**Phase 0: Problem definition and related-work grounding**

Goal: make the project concrete enough that a first benchmark and baseline implementation can start without ambiguity.

## Milestones

### M0: Repo and Document Setup

**Target**: 2026-05-15 **Status**: in progress

- [X]  Add `docs/` structure.
- [X]  Add proposal.
- [X]  Add kickoff / direction-setting meeting note.
- [X]  Add roadmap, decisions, and open questions.
- [X]  Add first daily progress note after the student starts reading / setup.

### M1: Problem Understanding

**Target**: 2026-05-18 **Status**:  started

- [X]  Write 2-3 toy examples showing the reveal process step by step.
- [X]  Define what is visible in `G_0`.
- [X]  Define candidate reveal actions: node, edge, neighborhood, feature, or evidence unit.
- [X]  Define budget `B` and what counts as one unit of cost.
- [X]  Explain the task in plain language and in mathematical notation.

### M2: Related-work Map

**Target**: 2026-05-20 **Status**: started

- [X]  List closest papers in graph classification.
- [X]  List closest papers in active learning on graphs.
- [X]  List closest papers in subgraph sampling / partial graph observation.
- [X]  List closest papers in graph explainability / rationale extraction.
- [X]  For each category, write how this project is different.

### M3: First Benchmark Protocol

**Target**: late May 2026 **Status**: pending M1 / M2

- [X]  Confirm benchmark dataset: OGBG-PPA (graph classification, hundreds of nodes per graph). Substitute only if blocked, and only with a dataset of comparable per-graph scale (graph classification, not node classification — so not PPI).
- [X]  Document per-graph node / edge distributions, node / edge feature availability, label setup, and train/val/test split sizes; explicitly confirm that OGBG-PPA has edge features and whether node features are absent.
- [X]  Explicitly keep MUTAG / molecular graphs as toy illustrations only, not the main benchmark.
- [X]  Implement data loading and full-graph classifier baseline (GIN or GCN) — this is the upper-bound number every later curve is read against.
- [X]  Freeze v0 protocol: `G_0` = Plan D with k=1 random seed (incident edge list / neighbor IDs visible; no raw node-feature assumption); action = `reveal_node(v)` revealing v as observed, edge features to already-revealed nodes, and incident edge list / neighbor IDs so the frontier can expand; unit cost; budget = number of revealed nodes, reported also as fraction of total nodes per graph.
- [X]  Implement reveal process and partial-graph representation.
- [X]  Implement three policies under the v0 action: uniform random over frontier; BFS from seed; max known-frontier degree (edges from a frontier node back into the revealed subgraph, not true graph degree).
- [ ]  Average each policy over multiple random seeds per graph.
- [ ]  Plot budget-performance curves vs. the full-graph upper bound.
- [ ]  Kill-criterion check: if BFS-from-seed reaches ~90% of full-graph accuracy at <10% reveal ratio, the dataset is too easy — switch dataset or harden `G_0` before M4.

### M4: Teacher / Trajectory Baselines

**Target**: after first benchmark protocol works**Status**: not started

- [ ]  Generate trajectories from random, BFS, degree / centrality, and heuristic policies.
- [ ]  Score trajectories by final prediction quality or information gain.
- [ ]  Estimate oracle / upper-bound headroom without leaking labels into the test-time policy.
- [ ]  Decide whether trajectory distillation is worth pursuing.

### M5: First Learned Reading Policy

**Target**: after M4 shows headroom**Status**: not started

- [ ]  Train a policy to imitate useful reading trajectories.
- [ ]  Compare learned reading against random / heuristic baselines under matched budget.
- [ ]  Evaluate robustness across budget values and graph sizes.
- [ ]  Decide paper framing and target venue.

## Submission Direction

The first paper should emphasize:

- a clear graph-learning problem formulation;
- a simple but reproducible benchmark protocol;
- evidence that adaptive graph reading improves fixed-budget graph classification;
- a method that is easier to understand than a large new GNN architecture.

Likely route:

- internal prototype first;
- workshop / arXiv if the formulation and benchmark are interesting but results are early;
- main-track submission if the learned policy clearly improves over strong baselines and the relation to prior work is crisp.
