"""
Phase 1 smoke test
==================

End-to-end: Reader(BFS, B=10) -> SubgraphBuilder -> GINEEncoder -> GraphClassifier

Goal: forward pass, no shape errors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import dgl
from ogb.graphproppred import DglGraphPropPredDataset

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_loop import RevealLoop
from active_graph_reader.reader.fixed.registry import get_strategy
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.gnn.gin_encoder import GINEEncoder
from active_graph_reader.gnn.classifier import GraphClassifier


def smoke_test():
    print("=" * 60)
    print("Phase 1 smoke test: Reader -> Encoder -> Classifier")
    print("=" * 60)

    # ---- 1. Load data -------------------------------------------------------
    print("\n[1] Loading OGBG-PPA (graph 0 only)...")
    ds = DglGraphPropPredDataset(name='ogbg-ppa', root=r'D:\ogb-master\dataset')
    n_classes = int(ds.num_classes)
    g, label = ds[0]
    g = dgl.add_self_loop(g)
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()
    has_edge_feat = 'feat' in g.edata
    edge_feat_dim = g.edata['feat'].shape[1] if has_edge_feat else 0
    label = label.item()

    print(f"  |V|={n_nodes}, |E|={n_edges}, edge_feat_dim={edge_feat_dim}, label={label}")

    # ---- 2. Run Reader ------------------------------------------------------
    print("\n[2] Running Reader (BFS, budget=10, seed=42)...")
    state = GraphState(g, seed_node=42)
    strategy = get_strategy('bfs')
    loop = RevealLoop(state, strategy)
    stats = loop.run(budget=10)
    print(f"  revealed nodes: {stats['n_revealed']}/{stats['n_total']} "
          f"({stats['reveal_ratio']:.2%})")
    print(f"  revealed edge feats: {stats['n_revealed_edge_feats']}")

    # ---- 3. Build subgraph --------------------------------------------------
    print("\n[3] SubgraphBuilder: GraphState -> DGLGraph...")
    builder = SubgraphBuilder()
    sub_g, original_ids = builder.build(state)
    print(f"  subgraph nodes: {sub_g.num_nodes()}, edges: {sub_g.num_edges()}")
    print(f"  edge feat shape: {sub_g.edata['feat'].shape if 'feat' in sub_g.edata else 'N/A'}")

    # ---- 4. Create encoder --------------------------------------------------
    print("\n[4] Creating GINEEncoder (hidden=32, layers=2)...")
    encoder = GINEEncoder(
        edge_feat_dim=edge_feat_dim,
        hidden_dim=32,
        n_layers=2,
        dropout=0.0,
    )
    encoder.eval()

    batch_g = dgl.batch([sub_g])
    node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long)

    with torch.no_grad():
        g_emb = encoder(batch_g, node_ids)
    print(f"  graph embedding shape: {g_emb.shape}  (expected: (1, 32))")
    assert g_emb.shape == (1, 32), f"Shape mismatch: {g_emb.shape}"

    # ---- 5. Create classifier -----------------------------------------------
    print("\n[5] Creating GraphClassifier (n_classes=37)...")
    model = GraphClassifier(
        edge_feat_dim=edge_feat_dim,
        n_classes=n_classes,
        hidden_dim=32,
        n_layers=2,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        logits = model(batch_g, node_ids)
    print(f"  Logits shape: {logits.shape}  (expected: (1, 37))")
    assert logits.shape == (1, n_classes), f"Shape mismatch: {logits.shape}"

    # ---- 6. Batch test (2 subgraphs) ----------------------------------------
    print("\n[6] Batch test (2 subgraphs with different budgets)...")

    state1 = GraphState(g, seed_node=42)
    loop1 = RevealLoop(state1, get_strategy('bfs'))
    loop1.run(budget=5)
    sub_g1, ids1 = builder.build(state1)

    state2 = GraphState(g, seed_node=42)
    loop2 = RevealLoop(state2, get_strategy('bfs'))
    loop2.run(budget=20)
    sub_g2, ids2 = builder.build(state2)

    batch_g2 = dgl.batch([sub_g1, sub_g2])
    node_ids2 = torch.zeros(batch_g2.num_nodes(), dtype=torch.long)

    with torch.no_grad():
        logits2 = model(batch_g2, node_ids2)
    print(f"  Batch logits shape: {logits2.shape}  (expected: (2, 37))")
    assert logits2.shape == (2, n_classes), f"Shape mismatch: {logits2.shape}"

    # ---- 7. Edge test: no-edge subgraph ------------------------------------
    print("\n[7] Edge test: subgraph with no revealed edges (budget=0)...")
    state0 = GraphState(g, seed_node=42)
    sub_g0, ids0 = builder.build(state0)
    print(f"  subgraph nodes: {sub_g0.num_nodes()}, edges: {sub_g0.num_edges()}")
    batch_g0 = dgl.batch([sub_g0])
    node_ids0 = torch.zeros(batch_g0.num_nodes(), dtype=torch.long)
    with torch.no_grad():
        logits0 = model(batch_g0, node_ids0)
    print(f"  Logits shape: {logits0.shape}  (expected: (1, 37))")
    assert logits0.shape == (1, n_classes)

    print("\n" + "=" * 60)
    print("Smoke test passed.")
    print("=" * 60)


if __name__ == '__main__':
    smoke_test()
