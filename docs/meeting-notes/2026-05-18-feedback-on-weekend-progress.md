# Feedback: 2026-05-18 - Weekend Progress Review

**Date**: 2026-05-18
**Topic**: Feedback on May 16-17 formulation notes and next benchmark direction.

## Summary

The May 16-17 notes are a strong consolidation pass on the problem formulation. In particular:

- The G_0 design-space survey (Plans A-F) and the action-type taxonomy give us a clean map of the problem.
- The 4-atomic-action cost model is a clean abstraction that should generalize across later experiments.
- The plain-language + symbolic formalization makes the decision process unambiguous.
- The MUTAG toy examples make the reveal process concrete and easy to explain.

This is enough for problem understanding. The next step should move from formulation to a real benchmark surface.

## Main Feedback

### Treat MUTAG / molecular graphs as toy illustrations only

The project motivation is large, noisy, or expensive-to-inspect graphs where reading the full graph is not free. Small molecular graphs (MUTAG averages ~18 atoms) are usually cheap to inspect completely, so they do not strongly motivate active graph reading.

The first real benchmark should use **OGBG-PPA, or another graph-classification dataset with comparable per-graph scale** (hundreds of nodes per graph and thousands of edges per graph).

Note: "PPI" in common usage refers to the inductive PPI dataset, which is a node-classification benchmark. Our task is graph classification, so we should anchor on OGBG-PPA, not PPI.

### Stop expanding the cost-model formalism for now

The 5/17 atomic cost model is retained as design scaffolding for M4 and later. For v0, we use a single action with unit cost. The reason is that a comprehensive action/cost formalism is more useful after we know which actions actually move the budget-performance curve. Right now we have no empirical evidence to design the formalism against.

## Frozen v0 Protocol

Commit to one choice per axis. Do not leave options open for v0:

1. **Dataset**: OGBG-PPA. Only switch if setup is unexpectedly blocked, and document the switch.
2. **`G_0` (initial observation)**: Plan D, small k-seed. Start with k = 1 randomly chosen seed node, with incident edge list / neighbor IDs visible; raw node features should be treated as absent for OGBG-PPA unless the dataset check shows otherwise. Justification: at PPA scale, "all node IDs visible" (Plan A) is itself an unrealistic free read; k-seed matches the real "start from one entry point" regime.
3. **Reveal action**: one action only — `reveal_node(v)` reveals v as an observed node, the edge features on edges from v to already-revealed nodes, and v's incident edge list / neighbor IDs so the frontier can expand. Features of newly discovered neighbors remain hidden until those nodes are revealed. Unit cost = 1.
4. **Budget**: number of revealed nodes. Report B both as an absolute count and as a fraction of total nodes per graph (so budget-performance curves are comparable across graphs of different sizes).
5. **Policies to compare under the same action** (all pick the next node from the *known frontier* — neighbors of already-revealed nodes):
   - Uniform random over the frontier.
   - BFS order from the seed.
   - Max known-frontier degree, where "degree" is the number of edges from the frontier node back into the revealed subgraph (not true graph degree, which is hidden).
6. **Seed averaging**: because k = 1 random seed can dominate early performance, report each policy averaged over multiple random seeds per graph.
7. **Full-graph upper bound**: train a standard GIN or GCN classifier on the full graph and report its accuracy. This is the most important grounding number — every budget-performance curve is read relative to it.

## Next Deliverable

By the next sync (target **Fri 2026-05-22**), produce a short benchmark-protocol report containing:

1. OGBG-PPA dataset stats: graph-size distribution (nodes, edges), node / edge feature availability, label setup, train/val/test split sizes; explicitly confirm that OGBG-PPA has edge features and whether node features are absent.
2. Frozen v0 reveal protocol restated for the record: `G_0`, action, budget, metric.
3. Data loader + full-graph baseline status (accuracy number if running; blockers if not).
4. Either (a) initial random / BFS / max-frontier-degree budget-performance curves, even ugly, or (b) a concrete plan with the implementation blocker named.
5. One line explicitly stating MUTAG is only a toy illustration, not the M3 benchmark.

## Risk To Avoid

The main risk is spending another week designing a richer cost/action formalism before showing that active graph reading has headroom on a real large-graph benchmark. The project should prioritize a minimal empirical proof point over a comprehensive formulation.

## Kill Criterion

If BFS-from-seed already reaches ~90% of full-graph accuracy at a small reveal ratio (e.g. <10% of nodes), the dataset is too easy for this problem — the question of *which* reveal policy to use does not matter enough. In that case, switch to a harder dataset or harder `G_0` before investing in M4 / M5.
