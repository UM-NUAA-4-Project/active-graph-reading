"""GINEEncoder node_feat_dim 支持测试：有节点特征时用 Linear 投影，无时保持 Embedding 行为。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/

import torch
import dgl

from active_graph_reader.gnn.gin_encoder import GINEEncoder
from active_graph_reader.gnn.classifier import GraphClassifier


def test_node_feat_path():
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g.ndata['feat'] = torch.randn(3, 5)
    g.edata['feat'] = torch.randn(3, 1)
    batch = dgl.batch([g])
    enc = GINEEncoder(edge_feat_dim=1, node_feat_dim=5, hidden_dim=16,
                      n_layers=2, dropout=0.0)
    assert hasattr(enc, 'node_proj') and not hasattr(enc, 'node_emb')
    out = enc(batch, torch.zeros(3, dtype=torch.long))
    assert out.shape == (1, 16), out.shape


def test_no_node_feat_path():
    g = dgl.graph(([0, 1], [1, 0]))
    g.edata['feat'] = torch.randn(2, 7)
    batch = dgl.batch([g])
    enc = GINEEncoder(edge_feat_dim=7, hidden_dim=16, n_layers=1, dropout=0.0)
    assert hasattr(enc, 'node_emb') and not hasattr(enc, 'node_proj')
    out = enc(batch, torch.zeros(2, dtype=torch.long))
    assert out.shape == (1, 16), out.shape


def test_classifier_node_feat():
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g.ndata['feat'] = torch.randn(3, 5)
    g.edata['feat'] = torch.randn(3, 1)
    batch = dgl.batch([g])
    clf = GraphClassifier(edge_feat_dim=1, n_classes=10, hidden_dim=16,
                          n_layers=1, dropout=0.0, node_feat_dim=5)
    out = clf(batch, torch.zeros(3, dtype=torch.long))
    assert out.shape == (1, 10), out.shape


if __name__ == '__main__':
    test_node_feat_path()
    test_no_node_feat_path()
    test_classifier_node_feat()
    print("OK: GINEEncoder/GraphClassifier node_feat_dim support")
