"""Phase1 基础库单测：ReaderConfig / ClassifierConfig 往返与旧格式兼容。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from active_graph_reader import ReaderConfig, build_reader
from active_graph_reader.config import infer_reader_config
from classifier import ClassifierConfig, GraphClassifier


def test_reader_config_roundtrip():
    cfg = ReaderConfig.from_legacy('attention', hidden_dim=96, n_layers=3,
                                   dropout=0.2, m=6, n_heads=8)
    cfg2 = ReaderConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    # 逐层列表往返
    cfg3 = ReaderConfig.from_dict({'kind': 'base', 'hidden_dims': [256, 128],
                                   'dropouts': [0.2, 0.1], 'edge_feat_dim': 5,
                                   'node_feat_dim': 5})
    assert cfg3.encoder_hidden_dims == [256, 128]
    assert cfg3.to_dict()['hidden_dims'] == [256, 128]


def test_reader_legacy_state_dict_keys():
    reader = build_reader(ReaderConfig.from_legacy('base'))
    keys = [k for k in reader.state_dict() if k.startswith('reader_mlp')]
    assert keys == ['reader_mlp.mlp.0.weight', 'reader_mlp.mlp.0.bias',
                    'reader_mlp.mlp.3.weight', 'reader_mlp.mlp.3.bias',
                    'reader_mlp.mlp.6.weight', 'reader_mlp.mlp.6.bias'], keys


def test_reader_validate():
    try:
        ReaderConfig(kind='attention', encoder_hidden_dims=[128, 64]).validate()
        raise AssertionError('同宽校验应报错')
    except ValueError:
        pass


def test_classifier_legacy_keys():
    clf = GraphClassifier()  # PPA 默认
    keys = list(clf.state_dict().keys())
    assert keys[0] in ('encoder.node_emb.weight', 'encoder.node_proj.weight')
    assert keys[-1] == 'head.bias'
    assert clf.head.in_features == 300 and clf.head.out_features == 37


def test_classifier_per_layer():
    cfg = ClassifierConfig.from_dict({'hidden_dims': [64, 32], 'dropout': 0.3,
                                      'n_classes': 5, 'edge_feat_dim': 2})
    clf = GraphClassifier(config=cfg)
    assert clf.head.in_features == 32 and clf.head.out_features == 5
    assert clf.config.hidden_dims == [64, 32]


def test_infer_reader_config_from_weights():
    from classifier import ClassifierConfig as _  # noqa: F401
    reader = build_reader(ReaderConfig.from_legacy('attention', hidden_dim=64,
                                                   n_layers=2, m=3, n_heads=2))
    cfg = infer_reader_config(reader.state_dict())
    assert cfg.kind == 'attention'
    assert cfg.encoder_hidden_dims == [64, 64]
    assert cfg.edge_feat_dim == 7
    assert cfg.m == 3


def run_all():
    test_reader_config_roundtrip()
    test_reader_legacy_state_dict_keys()
    test_reader_validate()
    test_classifier_legacy_keys()
    test_classifier_per_layer()
    test_infer_reader_config_from_weights()
    print('test_config_reader: ALL PASSED')


if __name__ == '__main__':
    run_all()
