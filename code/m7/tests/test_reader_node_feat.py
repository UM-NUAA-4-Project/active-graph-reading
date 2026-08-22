"""LearnableReader / AttentionReader 节点特征支持测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/

import torch
import dgl

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader.attention.attention_reader import AttentionReader
from active_graph_reader.engine.graph_state import GraphState


def _make_state(n_nodes=6, feat_dim=5, edge_dim=1):
    src = [i for i in range(n_nodes - 1)] * 2
    dst = [i + 1 for i in range(n_nodes - 1)] + [i for i in range(n_nodes - 1)]
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.randn(n_nodes, feat_dim)
    g.edata['feat'] = torch.randn(g.num_edges(), edge_dim)
    g = dgl.add_self_loop(g)
    state = GraphState(g, seed_node=0)
    state._budget_limit = 2
    return state


def test_base_reader_node_feat():
    reader = LearnableReader(edge_feat_dim=1, hidden_dim=32, n_layers=1,
                             dropout=0.0, node_feat_dim=5)
    state = _make_state()
    scores, probs, frontier = reader.forward(state)
    assert frontier, "expected non-empty frontier"
    assert scores.shape == (len(frontier),)
    assert probs.shape == (len(frontier),)
    assert abs(probs.sum().item() - 1.0) < 1e-4


def test_attention_reader_node_feat():
    reader = AttentionReader(edge_feat_dim=1, hidden_dim=32, n_layers=1,
                             dropout=0.0, node_feat_dim=5, m=2, n_heads=2)
    state = _make_state()
    beliefs = reader.init_beliefs(1)
    results = reader.forward_states([state], beliefs=beliefs)
    assert len(results[0]) == 3
    scores, probs, frontier = results[0]
    assert frontier and probs.shape == (len(frontier),)


if __name__ == '__main__':
    test_base_reader_node_feat()
    test_attention_reader_node_feat()
    print("OK: readers support node_feat_dim=5")
