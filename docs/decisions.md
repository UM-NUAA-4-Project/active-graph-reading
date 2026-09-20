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

### Unified eval JSON schema (multi-split × multi-strategy, one file per source)

**Decision**: `eval.py` evaluates all available held-out splits (default `--splits valid test`; datasets without a validation split such as cifar10 are auto-skipped with a warning) in a single run and writes ONE JSON per source: fixed strategies `--strategy random bfs` → `result/{ds}/fixed/eval_fixed.json`, reader checkpoints → `<checkpoint dir>/eval_learnable.json`. The schema merges the old per-file outputs: top level `{valid,test} → {random,bfs,learnable} → {s0..sN, mean} → "<budget:.2f>" → {mean,std,n}`, plus `per_class_{split}`, and evidence-reader references renamed to `acc_clf_reference_{split}`. `config.full_graph_ref` is now measured live (frozen classifier forward on each evaluated split's full graphs) instead of hardcoded null, so `plot_three_stage.py` draws the upper-bound dashed lines without `--ref-valid/--ref-test`.

**Rationale**: The previous one-JSON-per-(split, strategy) convention required 4 invocations and 4 files for one fixed-strategy baseline, and `plot_three_stage.py` had to be fed multiple files whose split coverage could silently diverge. The merged schema is a strict superset of the m5 reference files (`fixed_eval/eval_fixed.json`, `analysis/eval_learnable.json`), so the comparison scripts' loader (`data[split][strategy]['mean'][f'{b:.2f}']`) works unchanged.

**Implications**:

- `merge_eval_json` (train_reader/core/eval_loop.py) is the single merge implementation, reused by `eval.py` and by `tools/migrate_eval_json.py`, which consolidates the three historical JSON generations (m7 `eval_heldout_{split}_{strategy|reader}.json`, m6 `evaluate/eval_heldout_{split}.json`, old fixed-name files) into the canonical names; dry-run by default, `--write` to apply, `--archive` moves consumed files into `_legacy_eval/`.
- Historical eval JSONs are treated as merge sources, never deleted by default; conflicts (same split+strategy from multiple files) resolve to the newest mtime with a printed warning.
- `plot_three_stage.py` / `plot_eval_results.py` now bootstrap `sys.path` and resolve relative input paths against `code/` (previously only the output dir was resolved).
- Known legacy quirk surfaced by the migration report: collab fixed-strategy evals used budgets 0.05/0.10/0.15 while collab reader evals used 0.10/0.15/0.20 — three-stage plots on collab need `--budgets 0.10 0.15` (the intersection).

## 2026-09-19

### Eval entry-point split (reader vs baselines)

**Decision**: Split `train_reader/eval.py` into `eval_reader.py` (--checkpoint mode → `<checkpoint dir>/eval_learnable.json`, plus `--export-confidence` calibration export) and `eval_baselines.py` (--strategy mode → `result/{ds}/fixed/eval_fixed.json`, owning the baseline/ strategy registration and scorer-checkpoint wiring). Shared plumbing (common CLI args, per-split on-demand dataset loading, incremental JSON save, memory release) lives in `train_reader/core/eval_common.py` with an `eval_split(split, graphs, labels) -> [(key, results)]` callback contract. The old `eval.py` remains as a dispatch-only shim (reads `--checkpoint` vs `--strategy` from argv) so existing docs/commands keep working; new code should call the dedicated entry points.

**Rationale**: The unified script had grown to carry two unrelated argument families (reader chunking/calibration vs baseline scorer checkpoints/MCTS knobs), and every baseline-affecting change (e.g. agentnet fast path) forced edits into a file whose reader half was untouched. Splitting keeps CLI surfaces honest while the JSON schema, output paths, and `merge_eval_json` stay byte-identical.

**Implications**:

- `config` in each eval JSON is now written by its own entry point (`mode: 'reader' | 'fixed'`); the `{split} → {strategy_key}` schema and `full_graph_ref`/`n_graphs` accumulation are unchanged.
- `AGENTS.md` usage examples now point at the two new entries.

### Protocol audit of the eval chain (RNG discipline)

**Decision**: Three reproducibility / evaluation-correctness violations found and fixed during a protocol audit of the eval chain: (1) `RandomStrategy` (`--strategy random`) drew from the unseeded global `random` module, making random-baseline results differ run-to-run — it now seeds a per-rollout RNG as `random.Random(seed_node * 1_000_003 + n_nodes_total)` (same formula as random-walk, principle 4) and picks from the sorted frontier; (2) `evaluate_reader` reseeded a shared `random.Random(0)` per chunk for teleports, so a state's teleport depended on its position inside the chunk — teleports now draw from a per-state RNG seeded by `(seed_node, budget_limit)`, invariant to `--chunk-size` and chunk composition; (3) `AttentionReader` did not override `train`/`eval`, so `reader.eval()` (MRO: `BaseReader.eval` shadows `nn.Module.eval`, which only toggles `_training`) never reached its GNN submodules — **dropout stayed active during all attention-reader evaluations**, making reader eval nondeterministic run-to-run and systematically degraded. `AttentionReader` now overrides `train`/`eval` exactly like `LearnableReader` (`nn.Module.eval(self)` + `_training` flag).

**Rationale**: Principle 4 ("the same (graph, seed_i) must produce the same trajectory") applies to stochastic baselines and reader rollouts alike; a chunk-position-dependent stream silently changes results when `--chunk-size` changes, and an unseeded baseline cannot be re-run for the paper.

**Implications**:

- Random-baseline, reader-teleport, and **all attention/evidence reader** results shift vs JSONs produced before 2026-09-19; affected rows need re-running (reader rows because dropout no longer fires at eval — accuracy typically moves up slightly and becomes reproducible). bfs / max-frontier-degree / agentnet / sagpool / softmask / subgraphx pick-paths are deterministic and unaffected.
- Known, documented asymmetry (not changed): fixed-strategy rollouts end when the frontier empties (`rollout_strategy` never teleports), while reader rollouts teleport on empty frontiers with budget remaining — on disconnected graphs baselines may spend less budget than readers; interpretation caveat, see `baseline/protocol-adaptation.md`.

### AgentNet own-head degeneracy: root cause is training collapse, not an eval bug

**Decision**: Confirmed (2026-09-19) that `acc_own_*` ≈ class prior on all three datasets is not caused by the evaluation chain or by the porting of the forward pass: (1) the artifacts' own training history shows full-graph train/test accuracy frozen at chance from epoch 1 (cifar10 10.0%, collab 51.9% = majority prior), and the trained model outputs bitwise-identical logits for any input graph (cross-graph std = 0.0) — a constant function; (2) the eval chain is verified faithful (forward_batch ≡ forward within 8e-9; own_predict preprocessing reproduces training preprocessing bitwise on a fully-revealed graph; symmetrize dedupe/feature-lookup verified against the pre-refactor implementation); (3) instrumented mini-training locates the collapse mechanism: classification gradient reaching the input path (~3e-4) is two orders of magnitude below the weight-decay pressure (0.01), so input-path weights decay to zero and the model converges to the constant-output attractor (loss rises above ln C).

**Rationale**: A controlled experiment running the *unmodified original AgentNet code* (PyG, local AgentNet-main) on the same cifar10-superpixel data shows it learns where our training configuration collapses: README configuration (K=18 agents, global_agent_pool, reduce='log', mlp_width_mult=2, batch 32) reaches 20.7% train acc within 800 steps and climbs; a stripped K=1 configuration through the original code still learns (15.9%). The port's forward is structurally faithful (attention/BSEU/straight-through gumbel/updates/readout all match; the K=1 readout `cat[max,mean]=[v,v]` is linearly equivalent to our `fc(v)`), so the degeneracy is attributable to the single-agent training regime and omitted configuration, not to coding errors.

**Implications**:

- Undocumented deviations from the original working recipe found during this audit: `global_agent_pool` missing, `reduce='sum'` instead of `'log'`, `mlp_width_mult` 1 vs 2, 20 epochs vs 350, single-graph batches vs 32. The module docstring's "超参对齐原版 README" listed only the matching subset.
- New original-code finding: the original `--no_time_cond` switch is broken (forward unconditionally references `time_emb`, model.py:411), so removing time conditioning in our adaptation was forced, not optional.
- `acc_own_*` stands as an honest measurement ("the learning-based baseline degenerates to constant-class output under this protocol"); if a discriminative own head is ever needed, the training side must be made faithful (global_agent_pool/reduce/width/batching) and retrained — an architecture change that invalidates existing agentnet checkpoints.

### AgentNet AGR integration v2: single-agent constraint, restored original mechanisms

**Decision**: The AGR integration of AgentNet is locked to **one agent** (protocol: one node revealed per step). After a line-by-line review of the original source (local AgentNet-main, PyG) and the paper (arXiv 2206.11010), the previously undocumented training-side deviations are corrected: `mlp_width_mult=2` and `reduce='log'` (conv aggregation sum/log2(deg+1)) are restored as config-driven model options (defaults remain legacy values so existing checkpoints load strictly); training defaults move to the original recipe (batch 32 via gradient accumulation, Adam lr 1e-4 / wd 0.01 / clip 1.0, 100 epochs). `global_agent_pool` is intentionally not implemented: with a single agent it degenerates to self-broadcast (no information gain). The own-head readout walk now starts at the seed's row in the revealed subgraph (was: first node).

**Rationale**: The paper itself (Theorem 12) proves single-agent walks are strictly weaker on some structures, so the expressiveness loss is an accepted, documented protocol constraint — not something to paper over with the multi-agent machinery. The training collapse diagnosed earlier traced to the missing width/reduce plus batch-1 gradient noise, not to the single-agent choice; restoring the recipe is the minimal faithful fix. The original `--no_time_cond` switch is broken upstream (forward references `time_emb` unconditionally), and variable eval-time step counts make a fixed step table unusable, so time conditioning stays removed.

**Implications**:

- Old agentnet checkpoints load unchanged (config keys default to legacy architecture; strict state_dict validation); new checkpoints record `mlp_width_mult`/`reduce`/`batch_size`.
- The full-graph test accuracy of the collapsed artifacts (cifar10 10.0%, collab 51.9%) was a training failure, not an evaluation artifact; retraining with the restored recipe is expected to move `acc_own_*` off class-prior. cifar10/collab/PPA agentnet rows should be re-run after retraining.
- Original test protocol note: `test_argmax(τ→1e-6)` in the original still samples with gumbel noise (hard selection = argmax of src+gumbels); our evaluation is noiseless argmax — strictly more deterministic than the original.

### Frozen-reader state-attention aggregation head (offline reference track)

**Decision**: A new offline track (`code/train_reader/train_state_attention.py`, 2026-09-20) trains a state-attention aggregation head on top of a **frozen** evidence reader instead of modifying the reader architecture. The head attends over the per-step evidence states x_t = [z_t; h_t] (collected read-only via a new `return_states=True` optional parameter of `forward_states_evidence`; default False keeps the existing signature and behavior byte-identical) plus a normalized time scalar t/T, with masked softmax pooling (query zero-initialized so training starts at the uniform mean-pool floor), no terminal-state concatenation (the terminal forward is the last sequence element, attention can weight it), and a small MLP classifier. Terminal-only accuracy of the frozen reader is reported on the same rollouts alongside a uniform-pooling twin of the head (learned-vs-uniform ablation is therefore free). wd=0 for the head (dense-gradient small model; the AgentNet collapse lesson applies to end-to-end training, not here). Ablations are deferred unless the head is promoted into the method; all head variants share one collected feature cache (`--cache`), so promotion-time ablations retrain only small heads.

**Rationale**: Per-step predictions are already computed at every rollout step but the terminal judge consumes only the final supplementary forward; an attention pool can reweight trustworthy steps (time scalar exposes budget depth) without any reader retraining and without re-introducing end-to-end training instability. Replacing the reader's evidence head outright (per-step causal attention) is deferred: the frozen-head result decides whether the retraining cost across datasets and the checkpoint-compat churn is justified.

**Implications**:

- `forward_states_evidence` gains an optional 4th return only when `return_states=True`; all existing callers (BC/RL/eval loops) are untouched (9 evidence unit tests pass).
- Collection is lockstep-batched across episodes (`--collect-batch`, eval_loop-style; per-state hc/teleport RNG independent). Budget/step/hc semantics match a batch-1 reference, but trajectories are NOT bitwise identical: batch-shape GEMM noise (~1e-7) flips near-tie argmax picks and diverges whole rollouts (cifar10: 30/120 episodes, T always equal). Lockstep is the canonical collector and is bitwise run-to-run deterministic for a fixed batch composition (T/X/terminal all zero-diff in the check); the budget sentinel follows the `rollout_strategy` compound condition (budget checked every iteration — an earlier draft only checked it on empty frontier and silently revealed the whole graph; caught by T mean 117.9 ≈ n mean vs expected ~13 in the cifar10 smoke).
- Accuracies from this track are reference diagnostics (`acc_ens`-style), not protocol `acc_*`; the evidence reader's decision-classification-integrated terminal judgment remains the paper's method until a promotion decision.


### Attention reader removed (deprecated architecture cleanup)

**Decision**: The attention reader (belief-state architecture: `reader/attention/` package, `ReaderConfig.kind='attention'`, `m`/`n_heads` config fields, and all `use_belief` duck-typed lockstep branches in `core/bc.py`/`core/rl.py`/`core/eval_loop.py`/`core/common.py` and the three diagnosis scripts) is deleted (2026-09-20). `--reader-kind` choices are now `base|evidence`; `train_rl.py`'s default kind changes from `attention` to `evidence` (behavior change: an RL run with no explicit `--reader-kind` now trains evidence, not attention). `infer_reader_config` no longer infers `attention`, so legacy attention checkpoints fail validation with a clean message instead of attempting an import. `strip_classifier_keys` (belief-classifier key stripping in `tools/ckpt_io.py`) is kept — it is harmless and documents old-ckpt handling. All attention-reader experiment artifacts (`result/{cifar10,ppa}/experiments/*/attention/`, ~378 MB, 30 checkpoints) and attention-derived comparison outputs (`ppa/comparison/bc_gap*`, `teacher_gap*`, `cifar10/diagnosis/rl/diagnosis.json`) are deleted; tracked PNGs removed via git rm. The inventory/checklist is preserved at `docs/cleanup/attention-reader-cleanup.md`.

**Rationale**: Evidence reader supersedes attention (design rationale in `docs/design/evidence-reader.md` SS1/SS9); additionally the 2026-09-19 protocol audit found `AttentionReader` never overrode `train()`/`eval()`, so dropout stayed active during all its evaluations — its recorded results are invalid and unreferenced by any table in `docs/Summary of results.md`. Removing the dead branches eliminates the duck-typed `use_belief`/`hasattr(reader, 'init_beliefs')` forks from every training/eval/diagnosis loop.

**Implications**:

- `ReaderConfig` loses `m`/`n_heads`; old (base/evidence) checkpoints whose embedded configs contain these keys still load — `from_dict` ignores unknown keys.
- Legacy attention checkpoints are no longer loadable (kind rejected by `validate()`); their git history remains (JSONs up to 2026-09-09, PNGs at HEAD before this cleanup).
- The state-attention offline track (`train_state_attention.py`/`eval_state_attention.py`, `experiments/state-attn/`) is unrelated and untouched (verified: zero imports of the removed package).
