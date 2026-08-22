"""datasets 注册表 + 工具函数单测（不依赖 CIFAR10 下载，用构造数据）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/

import torch
import dgl

from m7.datasets.registry import (
    DatasetSpec, Dataset, register, get_dataset,
    stratified_split, has_self_loops,
)


def test_stratified_split():
    labels = [i % 10 for i in range(2000)]
    train_idx, valid_idx = stratified_split(labels, test_size=0.1, seed=42)
    assert len(train_idx) == 1800 and len(valid_idx) == 200
    assert not set(train_idx) & set(valid_idx)
    from collections import Counter
    c_train = Counter(labels[i] for i in train_idx)
    c_valid = Counter(labels[i] for i in valid_idx)
    assert len(c_train) == 10 and len(c_valid) == 10
    assert c_valid[0] == 20, c_valid  # 每类 200 个，10% → 20


def test_has_self_loops():
    g = dgl.graph(([0, 1], [1, 0]))
    assert not has_self_loops(g)
    g2 = dgl.add_self_loop(g)
    assert has_self_loops(g2)


def test_registry_roundtrip():
    spec = DatasetSpec(name='dummy', n_classes=3, node_feat_dim=0,
                       edge_feat_dim=1, needs_self_loop=False,
                       load=lambda s: Dataset(spec=s, train_graphs=[], train_labels=torch.tensor([]),
                                              valid_graphs=[], valid_labels=torch.tensor([]),
                                              test_graphs=[], test_labels=torch.tensor([]),
                                              train_indices=[], test_indices=[]))
    register(spec)
    ds = get_dataset('dummy')
    assert ds.spec.name == 'dummy'
    assert get_dataset('cifar10').spec.n_classes == 10
    assert get_dataset('cifar10').spec.node_feat_dim == 5
    assert get_dataset('cifar10').spec.edge_feat_dim == 1


if __name__ == '__main__':
    test_stratified_split()
    test_has_self_loops()
    test_registry_roundtrip()
    print("OK: registry utilities")
