"""GAT backbone 单测：前向 shape、边特征/无边特征、residual/virtual_node、
结构约束报错、config 序列化、checkpoint 重建（config 为准 + shape 校验）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgl
import torch

from classifier import (
    ClassifierConfig,
    GraphClassifier,
    classifier_config_from_ckpt,
    load_classifier,
)


def _batch(n_graphs=3, n_nodes=(8, 12, 6), n_edges=20, edge_feat_dim=7):
    graphs = []
    for i in range(n_graphs):
        g = dgl.rand_graph(n_nodes[i % len(n_nodes)], n_edges)
        g = dgl.add_self_loop(g)
        g.edata['feat'] = torch.randn(g.num_edges(), edge_feat_dim)
        graphs.append(g)
    return dgl.batch(graphs)


def test_gat_forward_shapes():
    torch.manual_seed(0)
    cfg = ClassifierConfig(backbone='gat', edge_feat_dim=7, n_classes=10,
                           hidden_dims=[64, 32], dropouts=[0.0, 0.0],
                           n_heads=[4, 2])
    clf = GraphClassifier(config=cfg)
    g = _batch()
    nids = torch.zeros(g.num_nodes(), dtype=torch.long)
    logits = clf(g, nids)
    assert logits.shape == (g.batch_size, 10), logits.shape
    emb = clf.get_embedding(g, nids)
    assert emb.shape == (g.batch_size, 32), emb.shape
    print(f'  forward logits {tuple(logits.shape)}, embedding {tuple(emb.shape)} OK')


def test_gat_zero_in_degree_and_no_edge_feat():
    torch.manual_seed(0)
    cfg = ClassifierConfig(backbone='gat', edge_feat_dim=1, n_classes=5,
                           hidden_dims=[16], dropouts=[0.0])
    clf = GraphClassifier(config=cfg)
    g = dgl.graph(([0, 1], [1, 2]))          # 节点 0 无入边
    logits = clf(g, torch.zeros(3, dtype=torch.long))
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()
    print('  zero-in-degree + 零边特征占位 OK')


def test_gat_residual_virtual_node():
    torch.manual_seed(0)
    for vn, res in ((True, False), (False, True), (True, True)):
        cfg = ClassifierConfig(backbone='gat', edge_feat_dim=7, n_classes=4,
                               hidden_dims=[32, 32], dropouts=[0.0, 0.0],
                               virtual_node=vn, residual=res, n_heads=4)
        clf = GraphClassifier(config=cfg)
        g = _batch()
        logits = clf(g, torch.zeros(g.num_nodes(), dtype=torch.long))
        assert logits.shape == (g.batch_size, 4)
    print('  virtual_node / residual 组合 OK')


def test_gat_invalid_structure_raises():
    base = dict(backbone='gat', edge_feat_dim=7, n_classes=3)
    # hidden 不能被 heads 整除（64 % 5 != 0）
    try:
        GraphClassifier(config=ClassifierConfig(
            backbone='gat', edge_feat_dim=7, n_classes=3,
            hidden_dims=[64], dropouts=[0.5], n_heads=[5]))
        raise AssertionError('hidden 64 / heads 5 应报整除错误')
    except ValueError as e:
        assert '整除' in str(e)
    # n_heads 长度不匹配
    try:
        GraphClassifier(config=ClassifierConfig(
            backbone='gat', edge_feat_dim=7, n_classes=3,
            hidden_dims=[64, 64], dropouts=[0.5, 0.5], n_heads=[4]))
        raise AssertionError('n_heads 长度不匹配应报错')
    except ValueError as e:
        assert 'n_heads 长度' in str(e)
    # virtual_node 要求各层同宽
    try:
        GraphClassifier(config=ClassifierConfig(
            **base, hidden_dims=[64, 32], dropouts=[0.5, 0.5],
            n_heads=[4, 4], virtual_node=True))
        raise AssertionError('不同宽 + virtual_node 应报错')
    except ValueError as e:
        assert '同宽' in str(e)
    print('  非法结构（整除/长度/同宽）报错 OK')


def test_config_roundtrip():
    cfg = ClassifierConfig(backbone='gat', edge_feat_dim=3, n_classes=7,
                           hidden_dims=[64, 64, 32], dropouts=[0.1, 0.2, 0.3],
                           n_heads=[8, 8, 4])
    d = cfg.to_dict()
    assert d['n_heads'] == [8, 8, 4] and 'hidden_dims' in d
    cfg2 = ClassifierConfig.from_dict(d)
    assert cfg2.to_dict() == d

    uniform = ClassifierConfig(backbone='gat', edge_feat_dim=3, n_classes=7,
                               hidden_dims=[64, 64], dropouts=[0.5, 0.5],
                               n_heads=[4, 4])
    du = uniform.to_dict()
    assert du['n_heads'] == 4 and 'hidden_dim' in du and 'n_layers' in du
    assert ClassifierConfig.from_dict(du).to_dict() == du
    # gin 的 to_dict 不写 n_heads
    assert 'n_heads' not in ClassifierConfig(
        backbone='gin', edge_feat_dim=7, n_classes=7,
        hidden_dims=[64], dropouts=[0.5]).to_dict()
    print('  config 序列化往返（列表/折叠标量/gin 不含 n_heads）OK')


def test_ckpt_rebuild_config_first(tmp_dir=None):
    """B 方案：重建一律以内嵌 config 为准；shape 校验负责一致性。"""
    import tempfile

    torch.manual_seed(0)
    cfg = ClassifierConfig(backbone='gat', edge_feat_dim=5, n_classes=9,
                           hidden_dims=[64, 32], dropouts=[0.25, 0.5],
                           n_heads=[4, 2])
    clf = GraphClassifier(config=cfg)
    sd = clf.state_dict()

    # 新版内嵌 {'classifier': {...}} / 旧版扁平 dict 两种形式都能读
    restored = classifier_config_from_ckpt({'config': {'classifier': cfg.to_dict()}})
    assert restored.to_dict() == cfg.to_dict()
    flat = classifier_config_from_ckpt({'config': cfg.to_dict()})
    assert flat.to_dict() == cfg.to_dict()

    # 无内嵌 config → 明确报错并指向迁移脚本
    try:
        classifier_config_from_ckpt({'model_state_dict': sd})
        raise AssertionError('无 config 应报错')
    except ValueError as e:
        assert 'migrate_legacy_classifier_ckpt' in str(e)

    # save → load_classifier 往返（写临时文件）
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'gat_clf.pt'
        torch.save({'model_state_dict': sd,
                    'config': {'classifier': cfg.to_dict()},
                    'val_acc': 0.5}, p)
        loaded = load_classifier(str(p), freeze=False)
        assert loaded.config.to_dict() == cfg.to_dict()
        assert next(loaded.parameters()).requires_grad is True  # freeze=False
        g = _batch(edge_feat_dim=5)
        logits = loaded(g, torch.zeros(g.num_nodes(), dtype=torch.long))
        assert logits.shape == (g.batch_size, 9)

        # config 与权重不符（篡改 config 的层宽）→ shape 校验大声报错
        bad = dict(cfg.to_dict())
        bad['hidden_dims'] = [128, 32]
        bad['hidden_dim'] = 128
        torch.save({'model_state_dict': sd,
                    'config': {'classifier': bad}}, p)
        try:
            load_classifier(str(p))
            raise AssertionError('config 与权重不符应报错')
        except RuntimeError as e:
            assert '不符' in str(e)
    print('  ckpt 重建（config 为准/缺 config 报错/shape 校验拦截篡改）OK')


def test_gin_ckpt_not_misdetected():
    torch.manual_seed(0)
    cfg = ClassifierConfig(edge_feat_dim=7, n_classes=4,
                           hidden_dims=[64, 64], dropouts=[0.5, 0.5])
    sd = GraphClassifier(config=cfg).state_dict()
    assert 'encoder.convs.0.attn_l' not in sd  # GIN 无注意力参数
    assert 'encoder.convs.0.fc.weight' not in sd
    print('  GIN state_dict 不含 GAT 键 OK')


if __name__ == '__main__':
    test_gat_forward_shapes()
    test_gat_zero_in_degree_and_no_edge_feat()
    test_gat_residual_virtual_node()
    test_gat_invalid_structure_raises()
    test_config_roundtrip()
    test_ckpt_rebuild_config_first()
    test_gin_ckpt_not_misdetected()
    print('GAT TESTS PASSED')
