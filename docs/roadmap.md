# Roadmap

This roadmap tracks the near-term execution plan for Active / Budgeted Graph Reading.

## Current Phase

**Phase 0: Problem definition and related-work grounding**

Goal: make the project concrete enough that a first benchmark and baseline implementation can start without ambiguity.

## Milestones

### M0: Repo and Document Setup

**Target**: 2026-05-15  
**Status**: in progress

- [x] Add `docs/` structure.
- [x] Add proposal.
- [x] Add kickoff / direction-setting meeting note.
- [x] Add roadmap, decisions, and open questions.
- [ ] Add first daily progress note after the student starts reading / setup.

### M1: Problem Understanding

**Target**: 2026-05-18  
**Status**: not started

- [ ] Write 2-3 toy examples showing the reveal process step by step.
- [ ] Define what is visible in `G_0`.
- [ ] Define candidate reveal actions: node, edge, neighborhood, feature, or evidence unit.
- [ ] Define budget `B` and what counts as one unit of cost.
- [ ] Explain the task in plain language and in mathematical notation.

### M2: Related-work Map

**Target**: 2026-05-20  
**Status**: not started

- [ ] List closest papers in graph classification.
- [ ] List closest papers in active learning on graphs.
- [ ] List closest papers in subgraph sampling / partial graph observation.
- [ ] List closest papers in graph explainability / rationale extraction.
- [ ] For each category, write how this project is different.

### M3: First Benchmark Protocol

**Target**: late May 2026  
**Status**: pending M1 / M2

- [ ] Choose first dataset, likely OGBG-PPA or another convenient OGBG graph-classification dataset.
- [ ] Implement data loading and full-graph classifier baseline.
- [ ] Implement reveal process and partial-graph representation.
- [ ] Implement random reveal and BFS / neighborhood expansion baselines.
- [ ] Plot budget-performance curves.

### M4: Teacher / Trajectory Baselines

**Target**: after first benchmark protocol works  
**Status**: not started

- [ ] Generate trajectories from random, BFS, degree / centrality, and heuristic policies.
- [ ] Score trajectories by final prediction quality or information gain.
- [ ] Estimate oracle / upper-bound headroom without leaking labels into the test-time policy.
- [ ] Decide whether trajectory distillation is worth pursuing.

### M5: First Learned Reading Policy

**Target**: after M4 shows headroom  
**Status**: not started

- [ ] Train a policy to imitate useful reading trajectories.
- [ ] Compare learned reading against random / heuristic baselines under matched budget.
- [ ] Evaluate robustness across budget values and graph sizes.
- [ ] Decide paper framing and target venue.

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
