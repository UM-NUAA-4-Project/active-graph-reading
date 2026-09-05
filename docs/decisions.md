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

## 2026-09-05

### Classifier readout for subgraph robustness (GIN+)

**Decision**: The upper-bound classifier must work on both full graphs and small revealed subgraphs (it is the RL reward source and the evidence-reader reference `acc_clf_*`). `GINEEncoder` therefore gains a configurable graph readout: `mean` (default, unchanged behavior), `meanmax` (mean+max concat), `sum`. The GIN+ classifier (residual + ffn head + meanmax readout, no virtual node) is trained with `meanmax`.

**Rationale**: Virtual-node classifiers aggregate global graph statistics, which shift drastically on tiny revealed subgraphs (5-15% budget = a handful of nodes) — measured as severe `acc_clf_*` degradation. Sum pooling scales readout magnitude with node count, so it is rejected for the same distribution-shift reason: on an 8-node subgraph its magnitude is ~20x smaller than on a 150-node full graph. Mean and per-dim max are both invariant to node count, so `meanmax` adds discriminative power without introducing scale drift between full-graph training and subgraph evaluation.

**Implications**:

- `ClassifierConfig.readout` ('mean' | 'meanmax' | 'sum') is serialized in checkpoints; 'mean' stays the default and keeps old configs/JSON unchanged (old checkpoints load strictly, no migration).
- `GINEEncoder.out_dim` doubles under 'meanmax'; `GraphClassifier` sizes its head from `encoder.out_dim` (falling back to last hidden width for backbones without the attribute).
- Diagnostic tool: `code/diagnosis/classifier_subgraph_eval.py` measures a classifier checkpoint's full-graph accuracy plus per-budget accuracy under fixed strategies (bfs/random) with protocol seeds — the acceptance gate for any new reward/reference classifier.

### classifier package: GNN+ building-block architecture (plus/)

**Decision**: Reorganize `code/classifier` into a block-based architecture modeled on the GNN+ unified enhancement framework ("Can Classic GNNs Be Strong Baselines for Graph-level Tasks?", ICML 2025, arXiv 2502.09263). The six components map to pluggable blocks controlled by `ClassifierConfig` fields: edge feature integration (`plus/edge.py`, config `edge_encoder`), normalization (`plus/norm.py`, config `gnn_norm`), dropout (in-loop stage, config `dropouts`), residual connections (in-loop stage, config `residual`), per-layer FFN (`plus/ffn.py`, config `ffn`/`ffn_dropout`, faithful to Eq.10 `BN(σ(hW₁)W₂ + h)` with its own residual), and positional encoding (`plus/pos_enc.py`, RWSE per Eq.12, config `pos_enc`/`pos_enc_ksteps`). Graph readout lives in `plus/readout.py` (mean/meanmax/sum). `GINEEncoder.forward` follows the paper's Eq.11 pipeline: MP(edge-integrated) → norm → σ → dropout → residual → FFN.

**Rationale**: The classifier needs continuous capacity growth (GIN+ experiments on cifar10 showed meanmax readout + residual fixes the subgraph collapse that virtual-node causes) without each new technique entangling the encoder loop. Blocks are parameter-controlled, default-off, and default settings reproduce pre-refactor numerics bit-for-bit. Backbones declare supported blocks via `layers.BACKBONE_KWARGS`; `GraphClassifier` filters kwargs by the whitelist and rejects enabled-but-unsupported blocks loudly (no silent ignoring).

**Implications**:

- Dependency direction: `plus/` (torch/dgl only) ← `layers/` ← `graph_classifier.py` (renamed from `gin_classifier.py`) ← `factory.py`. `plus/` must not import config/layers.
- All new config fields serialize only when non-default; old checkpoints (GIN, GIN-res-vn, GIN-plus, PPA) rebuild and load strictly with identical state_dict keys — verified by golden key-sequence and fixed-input logit tests in `tests/test_plus_blocks.py`, captured pre-refactor.
- RWSE is injected as `[x ‖ pe] → node_proj` (input dim = node_feat_dim + ksteps); it is subgraph-computable and scale-robust, unlike Laplacian PE (sign/axis ambiguity under graph truncation), which is intentionally not implemented.
- Virtual node remains legacy code inside `gin.py` (not a GNN+ component; collapses to chance on revealed subgraphs — see `result/cifar10/classifier/GIN-res-vn/subgraph_eval.json`).
- New backbones: implement an encoder (whitelisted kwargs + `out_dim` attribute), register in `BACKBONES` + `BACKBONE_KWARGS`, organize the layer loop per Eq.11 with default-off blocks reproducing original behavior.
