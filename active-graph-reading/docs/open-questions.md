# Open Questions

## Problem Formulation

- What is the cleanest definition of the initial partial graph `G_0`?
- What exactly can the policy reveal: nodes, edges, node features, neighborhoods, or learned evidence units?
- Does one reveal action always have unit cost, or should cost depend on the amount of graph structure revealed?
- Should the policy know the full graph topology without features, or should both topology and features be partially hidden?

## Dataset and Benchmark

- Is OGBG-PPA the best first dataset, or is another OGBG dataset faster and clearer for proof of concept?
- What graph sizes and label types make budgeted reading non-trivial?
- Should the first benchmark use synthetic hiding / reveal on public datasets, or a domain where partial observation is natural?
- What full-graph baseline should be used for reference?

## Baselines

- How strong are random reveal, BFS / neighborhood expansion, degree / centrality, and simple heuristic policies?
- What is a fair oracle / upper bound that does not leak labels into the deployable test-time policy?
- Should we compare against fixed-size subgraph sampling methods?
- Should explainability-style rationale extraction be used as a teacher or only as related work?

## Learning Method

- Is imitation learning from successful trajectories enough for a first method?
- How should trajectory quality be scored: final accuracy, confidence gain, information gain, or marginal utility?
- Should the classifier and reading policy be trained jointly or in stages?
- How do we prevent the learned policy from overfitting to dataset-specific graph artifacts?

## Paper Framing

- Is the core contribution a new task, a benchmark protocol, or a learning method?
- Should the paper be framed as graph ML, active evidence acquisition, or agentic graph reasoning?
- Which target venue makes sense if the first results are promising?
