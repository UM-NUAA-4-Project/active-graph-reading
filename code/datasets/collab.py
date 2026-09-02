"""COLLAB 数据集加载（TUDataset，DGL 内置）。

科学合作网络图分类：5,000 图 / 3 类，平均 74.5 节点 / ~2,458 边（稠密）。
无节点特征、无边特征、无官方划分——本加载器：
- 边特征：注入 1 维零占位（协议管线要求 g.edata['feat'] 存在）；随后
  dgl.add_self_loop 会给新增自环边填 feat=1.0（dgl 行为），即
  "原始边 feat=0、自环边 feat=1"，仍不携带协同关系信息——frontier
  区分度退化为纯拓扑，这正是 COLLAB 的实验点之一
- 划分：按类别分层抽样 80/10/10（固定种子 42，确定性；各 split 类别
  分布一致；整体再洗牌使 --max-graphs 截断仍为类别均衡子集）
- 自环：原始图无自环，加载后按协议统一 dgl.add_self_loop()（加一次）
- graph_idx 语义 = ds.train_graphs 本地位置（同 cifar10）

数据目录：--root 指向 TUDataset raw_dir（如 D:\\data\\COLLAB，
其下 DGL 会生成 COLLAB_<hash>/ 处理产物）。
"""

import random
from typing import List, Tuple

import torch
import dgl
from dgl.data import TUDataset
from tqdm import tqdm

from .registry import DatasetSpec, Dataset, register

SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)   # train / valid / test


def _load_all(raw_dir: str) -> Tuple[List, List[int]]:
    ds = TUDataset(name='COLLAB', raw_dir=raw_dir)
    graphs, labels = [], []
    for item in tqdm(ds, desc='Loading COLLAB'):
        g, y = item if isinstance(item, tuple) else (item, None)
        if y is None:
            y = int(g.ndata['label'][0].item()) if 'label' in g.ndata else 0
        # 无边特征：注入 1 维零占位（GraphState/FrontierEncoder 均要求 edata['feat']）
        if 'feat' not in g.edata:
            g.edata['feat'] = torch.zeros(g.num_edges(), 1)
        graphs.append(g)
        labels.append(int(y))
    return graphs, labels


def _split_indices(labels: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """按类别分层抽样 80/10/10（固定种子，确定性）。

    每类内部洗牌后按比例切分，保证各 split 类别分布一致；各 split 拼回
    后再整体洗牌一次，使 --max-graphs 截断（取前 N）仍是类别均衡的
    随机子集（TUDataset 原始顺序按类别聚集）。
    """
    by_class: dict = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    rng = random.Random(SPLIT_SEED)
    train: List[int] = []
    valid: List[int] = []
    test: List[int] = []
    for cls in sorted(by_class):
        idx = by_class[cls]
        rng.shuffle(idx)
        n_train = int(len(idx) * SPLIT_RATIOS[0])
        n_valid = int(len(idx) * SPLIT_RATIOS[1])
        train += idx[:n_train]
        valid += idx[n_train:n_train + n_valid]
        test += idx[n_train + n_valid:]

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def _finalize(graphs: List, labels: List[int], needs_self_loop: bool,
              desc: str) -> Tuple[List, torch.Tensor]:
    if needs_self_loop and graphs:
        # 注意：部分 COLLAB 原始图自带少量自环，直接 add_self_loop 会造成
        # 双自环（2026-07-28 PPA 同类 bug）——先 remove 再统一 add，
        # 保证每节点恰好 1 条自环。
        graphs = [dgl.add_self_loop(dgl.remove_self_loop(g))
                  for g in tqdm(graphs, desc=desc)]
    return graphs, torch.tensor(labels)


def load_collab(spec: DatasetSpec) -> Dataset:
    if not spec.root:
        raise ValueError(
            f"数据集 '{spec.name}' 数据目录未指定：必须通过 --root 传入"
            "（如 --root D:\\data\\COLLAB）")

    all_graphs, all_labels = _load_all(spec.root)
    train_idx, valid_idx, test_idx = _split_indices(all_labels)

    def _pick(indices):
        if spec.max_graphs > 0:
            indices = indices[:spec.max_graphs]
        return ([all_graphs[i] for i in indices],
                [all_labels[i] for i in indices])

    parts = {}
    for split, indices in (('train', train_idx), ('valid', valid_idx),
                           ('test', test_idx)):
        if split in spec.splits:
            gs, ls = _pick(indices)
            parts[split] = _finalize(gs, ls, spec.needs_self_loop,
                                      f'Self-loop {split}')
        else:
            parts[split] = ([], torch.tensor([]))

    train_graphs, train_labels = parts['train']
    valid_graphs, valid_labels = parts['valid']
    test_graphs, test_labels = parts['test']
    return Dataset(
        spec=spec,
        train_graphs=train_graphs,
        train_labels=train_labels,
        valid_graphs=valid_graphs,
        valid_labels=valid_labels,
        test_graphs=test_graphs,
        test_labels=test_labels,
        train_indices=train_idx[:spec.max_graphs if spec.max_graphs > 0 else None],
        valid_indices=valid_idx[:spec.max_graphs if spec.max_graphs > 0 else None],
        test_indices=test_idx[:spec.max_graphs if spec.max_graphs > 0 else None],
    )


register(DatasetSpec(
    name='collab',
    n_classes=3,
    node_feat_dim=0,   # 原始数据无节点特征（type embedding / 共享 Embedding 路径）
    edge_feat_dim=1,   # 无边特征：1 维零占位（见 _load_all）
    needs_self_loop=True,
    budget_options=(0.05, 0.10, 0.15),
    load=load_collab,
))
