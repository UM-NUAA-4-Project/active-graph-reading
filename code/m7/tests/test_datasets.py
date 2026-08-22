"""datasets 注册表 + 工具函数单测（不依赖 CIFAR10 下载，用构造数据）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # code/

import torch
import dgl

from m7.datasets.registry import (
    DatasetSpec, Dataset, register, get_dataset,
    has_self_loops,
)


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


def test_lazy_dataset_repr_eq_no_load():
    """repr/eq 防护：_LazyDataset 的 __repr__/__eq__ 不得触发数据懒加载。"""
    calls = {'n': 0}

    def load(spec):
        calls['n'] += 1
        return Dataset(spec=spec, train_graphs=[], train_labels=torch.tensor([]),
                       valid_graphs=[], valid_labels=torch.tensor([]),
                       test_graphs=[], test_labels=torch.tensor([]),
                       train_indices=[], test_indices=[])

    spec = DatasetSpec(name='lazy_no_load', n_classes=3, node_feat_dim=0,
                       edge_feat_dim=1, needs_self_loop=False, load=load)
    register(spec)
    ds = get_dataset('lazy_no_load')
    assert calls['n'] == 0
    assert 'lazy_no_load' in repr(ds)
    assert calls['n'] == 0
    assert ds == ds                      # 身份相等
    assert ds == get_dataset('lazy_no_load')  # 缓存同一对象
    assert not (ds == object())
    assert calls['n'] == 0


if __name__ == '__main__':
    test_has_self_loops()
    test_registry_roundtrip()
    test_lazy_dataset_repr_eq_no_load()
    print("OK: registry utilities")
