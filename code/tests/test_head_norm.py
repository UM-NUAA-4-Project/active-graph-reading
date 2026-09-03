"""分类头（linear/ffn）与 GNN 层间归一化（gnn_norm）单测。

覆盖：模块类型与 state_dict 键名、前向（含 batch=1 eval）、config 往返
与非法值校验、旧 config（无新键）的向后兼容（strict load 通过）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgl
import torch
import torch.nn as nn

from classifier import ClassifierConfig, GraphClassifier


def _batch(n_graphs=3, n_nodes=(8, 12, 6), n_edges=20, edge_feat_dim=7,
           node_feat_dim=0):
    graphs = []
    for i in range(n_graphs):
        g = dgl.rand_graph(n_nodes[i % len(n_nodes)], n_edges)
        g = dgl.add_self_loop(g)
        g.edata['feat'] = torch.randn(g.num_edges(), edge_feat_dim)
        if node_feat_dim > 0:
            g.ndata['feat'] = torch.randn(g.num_nodes(), node_feat_dim)
        graphs.append(g)
    return dgl.batch(graphs)


def _cfg(**kw):
    base = dict(edge_feat_dim=7, n_classes=10, hidden_dims=[32, 32],
                dropouts=[0.0, 0.0])
    base.update(kw)
    return ClassifierConfig(**base)


def test_gin_default_batchnorm_linear():
    clf = GraphClassifier(config=_cfg())
    assert isinstance(clf.encoder.bns[0], nn.BatchNorm1d)
    assert clf.encoder.norm_type == 'batchnorm'
    assert isinstance(clf.head, nn.Linear)
    sd = clf.state_dict()
    assert 'encoder.bns.0.weight' in sd and 'head.weight' in sd
    print('  GIN 默认: BatchNorm 层间 + Linear 头 OK')


def test_gin_norm_none_and_layernorm():
    clf = GraphClassifier(config=_cfg(gnn_norm='none'))
    assert isinstance(clf.encoder.bns[0], nn.Identity)
    assert 'encoder.bns.0.weight' not in clf.state_dict()
    clf = GraphClassifier(config=_cfg(gnn_norm='layernorm'))
    assert isinstance(clf.encoder.bns[0], nn.LayerNorm)
    assert 'encoder.bns.0.weight' in clf.state_dict()
    # batch=1 eval 前向（BN/LN 的边界情形）
    g = _batch(n_graphs=1)
    clf.eval()
    with torch.no_grad():
        logits = clf(g, torch.zeros(g.num_nodes(), dtype=torch.long))
    assert logits.shape == (1, 10)
    print('  GIN gnn_norm=none/layernorm + batch=1 eval OK')


def test_gat_norm_default_and_batchnorm():
    clf = GraphClassifier(config=_cfg(backbone='gat', n_heads=[4, 4]))
    assert clf.encoder.norm_type == 'none'
    assert isinstance(clf.encoder.bns[0], nn.Identity)
    assert not any(k.startswith('encoder.bns.') for k in clf.state_dict())
    clf = GraphClassifier(config=_cfg(backbone='gat', n_heads=[4, 4],
                                      gnn_norm='batchnorm'))
    assert isinstance(clf.encoder.bns[0], nn.BatchNorm1d)
    assert 'encoder.bns.0.weight' in clf.state_dict()
    g = _batch()
    with torch.no_grad():
        logits = clf(g, torch.zeros(g.num_nodes(), dtype=torch.long))
    assert logits.shape == (g.batch_size, 10)
    print('  GAT 默认无归一化 / batchnorm 显式开启 OK')


def test_head_ffn():
    clf = GraphClassifier(config=_cfg(head='ffn'))
    assert isinstance(clf.head, nn.Sequential)
    sd = clf.state_dict()
    assert 'head.0.weight' in sd and 'head.2.weight' in sd
    assert 'head.weight' not in sd
    g = _batch()
    with torch.no_grad():
        logits = clf(g, torch.zeros(g.num_nodes(), dtype=torch.long))
    assert logits.shape == (g.batch_size, 10)
    print('  ffn 分类头: Sequential 键名 + 前向 OK')


def test_config_roundtrip_and_validation():
    cfg = _cfg(head='ffn', gnn_norm='layernorm')
    d = cfg.to_dict()
    assert d['head'] == 'ffn' and d['gnn_norm'] == 'layernorm'
    cfg2 = ClassifierConfig.from_dict(d)
    assert cfg2.head == 'ffn' and cfg2.gnn_norm == 'layernorm'

    # 非默认值才落盘：默认 config 不含新键，from_dict 还原默认
    d2 = _cfg().to_dict()
    assert 'head' not in d2 and 'gnn_norm' not in d2
    cfg3 = ClassifierConfig.from_dict(d2)
    assert cfg3.head == 'linear' and cfg3.gnn_norm is None

    for bad in ({'head': 'mlp'}, {'gnn_norm': 'groupnorm'}):
        try:
            ClassifierConfig.from_dict({**d2, **bad})
            raise AssertionError(f'{bad} 应触发 ValueError')
        except ValueError:
            pass
    print('  config 往返 + 非法值校验 OK')


def test_backward_compat_old_config():
    """旧 checkpoint 的内嵌 config 无 head/gnn_norm 键 → 重建后 strict load 通过。"""
    clf = GraphClassifier(config=_cfg(virtual_node=True, residual=True))
    old_cfg = _cfg(virtual_node=True, residual=True).to_dict()
    assert 'head' not in old_cfg and 'gnn_norm' not in old_cfg
    sd = clf.state_dict()
    rebuilt = GraphClassifier(config=ClassifierConfig.from_dict(old_cfg))
    rebuilt.load_state_dict(sd, strict=True)

    # 真实旧 ckpt（finetune3，VN+residual）也应照常重建
    from classifier import classifier_config_from_ckpt, load_classifier
    ckpt_path = (Path(__file__).resolve().parents[1] / 'result' / 'cifar10'
                 / 'classifier' / 'finetune3' / 'best_cifar10_classifier.pt')
    if ckpt_path.exists():
        real = load_classifier(str(ckpt_path), device='cpu')
        real.eval()
        g = _batch(edge_feat_dim=real.config.edge_feat_dim,
                   node_feat_dim=real.config.node_feat_dim)
        with torch.no_grad():
            logits = real(g, torch.zeros(g.num_nodes(), dtype=torch.long))
        assert logits.shape == (g.batch_size, real.config.n_classes)
        assert classifier_config_from_ckpt({'config': {
            'classifier': real.config.to_dict()}}).head == 'linear'
        print('  向后兼容: 旧 config strict load + finetune3 重建 OK')
    else:
        print('  向后兼容: 旧 config strict load OK（finetune3 不在本机，跳过）')


if __name__ == '__main__':
    test_gin_default_batchnorm_linear()
    test_gin_norm_none_and_layernorm()
    test_gat_norm_default_and_batchnorm()
    test_head_ffn()
    test_config_roundtrip_and_validation()
    test_backward_compat_old_config()
    print('HEAD/NORM TESTS PASSED')
