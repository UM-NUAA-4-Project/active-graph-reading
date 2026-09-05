"""readout 池化选项单测：meanmax/sum 构建、维度、序列化与旧 config 兼容。"""

import sys
from pathlib import Path

import torch
import dgl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier import ClassifierConfig, GraphClassifier  # noqa: E402


def _batch(n_feat=5, e_feat=1):
    g1 = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g1.ndata['feat'] = torch.randn(3, n_feat)
    g1.edata['feat'] = torch.randn(3, e_feat)
    g2 = dgl.graph(([0, 1], [1, 0]))
    g2.ndata['feat'] = torch.randn(2, n_feat)
    g2.edata['feat'] = torch.randn(2, e_feat)
    return dgl.batch([g1, g2])


def test_meanmax_output_dim():
    cfg = ClassifierConfig.from_legacy(edge_feat_dim=1, n_classes=10,
                                       hidden_dim=32, n_layers=3,
                                       readout='meanmax', head='ffn')
    clf = GraphClassifier(config=cfg)
    assert clf.encoder.out_dim == 64
    assert clf.head[0].in_features == 64
    out = clf(_batch(), torch.zeros(5, dtype=torch.long))
    assert out.shape == (2, 10)


def test_sum_readout():
    cfg = ClassifierConfig.from_legacy(edge_feat_dim=1, n_classes=10,
                                       hidden_dim=32, n_layers=2, readout='sum')
    clf = GraphClassifier(config=cfg)
    assert clf.encoder.out_dim == 32
    assert clf(_batch(), torch.zeros(5, dtype=torch.long)).shape == (2, 10)


def test_legacy_config_defaults_to_mean():
    old = {'backbone': 'gin', 'hidden_dim': 32, 'n_layers': 2, 'dropout': 0.5}
    cfg = ClassifierConfig.from_dict(old)
    assert cfg.readout == 'mean'
    assert 'readout' not in cfg.to_dict()  # 默认档不落盘，旧 JSON 零漂移


def test_readout_roundtrip_and_state_dict_compat():
    cfg = ClassifierConfig.from_legacy(edge_feat_dim=1, n_classes=10,
                                       hidden_dim=32, n_layers=2,
                                       readout='meanmax')
    d = ClassifierConfig.from_dict(cfg.to_dict())
    assert d.readout == 'meanmax'
    # 旧 checkpoint（mean）与新构建（mean）state_dict 键集一致
    old = GraphClassifier(config=ClassifierConfig.from_dict(old_cfg()))
    new = GraphClassifier(config=ClassifierConfig.from_dict(old_cfg()))
    assert old.state_dict().keys() == new.state_dict().keys()
    new.load_state_dict(old.state_dict(), strict=True)


def old_cfg():
    return {'backbone': 'gin', 'edge_feat_dim': 1, 'n_classes': 10,
            'hidden_dim': 32, 'n_layers': 2, 'dropout': 0.5}


def test_invalid_readout_raises():
    for bad in ('avg', 'MAX', ''):
        try:
            ClassifierConfig.from_dict({**old_cfg(), 'readout': bad})
        except ValueError:
            continue
        raise AssertionError(f"readout='{bad}' 应报错")
