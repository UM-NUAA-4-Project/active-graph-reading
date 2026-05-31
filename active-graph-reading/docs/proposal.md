# Proposal: Active / Budgeted Graph Reading

## One-line Idea

Learn how to read a graph under a fixed observation budget before making a graph-level prediction.

## Motivation

Most graph classification methods assume the full graph is available at prediction time. This is convenient for benchmarks but less natural for large, noisy, or expensive-to-inspect graphs. In security provenance graphs, program-analysis graphs, biological graphs, or large interaction graphs, the discriminative evidence may live in a small subgraph while the full graph contains many irrelevant nodes and edges.

Modern LLM / coding agents face a similar constraint when working with large documents or codebases. They do not read everything at once. They search, retrieve, inspect, and accumulate evidence until they have enough context to answer. This project imports that evidence-acquisition view into graph classification.

## Problem Setup

Given a graph `G` and graph label `y`, the model starts with a partial observation `G_0`. At each step `b = 1...B`, it chooses an action that reveals more graph evidence:

```text
a_b = policy(G_<b)
G_b = reveal(G_<b, a_b)
```

After `B` reveal steps, the model predicts:

```text
y_hat = classifier(G_B)
```

The goal is to maximize graph-level prediction performance under a fixed reveal budget.

## Why This Is Different

- **Not standard graph classification**: the full graph is not given for free.
- **Not only subgraph sampling**: the evidence is chosen sequentially and adaptively.
- **Not standard active learning**: the label budget is not the primary constraint; the graph-observation budget is.
- **Not only explainability**: the model must acquire evidence before prediction, not explain a prediction after seeing the full graph.

## Initial Method Direction

Start with teacher-student / trajectory distillation:

1. Generate graph-reading trajectories using simple policies:
   - random reveal;
   - BFS / neighborhood expansion;
   - degree or centrality heuristics;
   - task-informed heuristic or oracle upper bound if feasible.
2. Score trajectories by final prediction quality, information gain, or budget-performance tradeoff.
3. Train a learned policy to imitate useful trajectories.
4. Evaluate whether learned graph reading outperforms random / heuristic reading under the same budget.

## First Benchmark Candidate

Start from an OGBG-style graph-classification dataset, with OGBG-PPA as a candidate if setup and graph sizes are convenient. The first benchmark should prioritize:

- easy public access;
- clear graph-level labels;
- non-trivial graph size;
- fast enough training / evaluation loop;
- simple baselines that can be implemented quickly.

## First Deliverables

1. Toy examples of the reveal process.
2. Related-work map covering active learning, partial graph observation, subgraph sampling, graph explanation, and graph classification.
3. A minimal benchmark protocol with random / BFS baselines.
4. A first learned or imitation-based policy if the protocol shows meaningful headroom.
