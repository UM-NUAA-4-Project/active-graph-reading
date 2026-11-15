"""OGBG-PPA 数据集加载（选择性加载，避免 158k 图全量进内存）。

数据布局（DGL processed 格式，与 m5 时代一致）：
    <root>/ogbg_ppa/processed/dgl_data_processed     # 全量图 + labels
    <root>/ogbg_ppa/split/species/{train,valid,test}.csv.gz

关键语义：
- graph_idx 使用 OGB 全局索引（与 m4/m5/m6 全部既有轨迹数据兼容）；
  train_indices = 实际加载的 OGB 索引（升序）。
- 图从磁盘加载后统一 dgl.add_self_loop()（协议要求，见 AGENTS.md）。
- max_graphs > 0 时每 split 选 N 个索引（升序，确定性）：默认前 N 截断，
  stratified=True 时按类别比例分层抽样（registry.stratified_take）。
  PPA 全量 train 约 158k 图（>5GB 内存），训练/评估请务必配合 max_graphs。
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
import dgl
from tqdm import tqdm

from .registry import DatasetSpec, Dataset, register, stratified_take


def _ppa_paths(root: str):
    data_dir = Path(root) / 'ogbg_ppa'
    return (data_dir / 'processed' / 'dgl_data_processed',
            data_dir / 'split' / 'species')


def _read_split_ids(split_dir: Path, split: str) -> List[int]:
    df = pd.read_csv(split_dir / f'{split}.csv.gz', compression='gzip', header=None)
    return df.values.flatten().tolist()


def _load_graphs(proc_path: Path, ids: List[int], needs_self_loop: bool,
                 desc: str = 'Loading PPA graphs'):
    """按 OGB 全局索引加载图与标签；返回 (graphs(与 ids 对齐), labels)。"""
    graphs, label_dict = dgl.load_graphs(str(proc_path), idx_list=list(ids))
    labels = torch.tensor([int(label_dict['labels'][i].item()) for i in ids])
    if needs_self_loop:
        graphs = [dgl.add_self_loop(g) for g in tqdm(graphs, desc=desc)]
    return graphs, labels


def _ppa_all_labels(spec, min_len: int) -> torch.Tensor:
    """全量标签（按 OGB 全局索引对齐），分层抽样选图前需要。

    DGL 的 labels 为文件级存储：idx_list 加载也返回全量 label_dict，
    先用 idx_list=[0] 廉价读取并落盘缓存（labels_cache.pt）；若读到的
    长度覆盖不了 min_len（老版 DGL 会按 idx_list 切 label），回退一次性
    全量加载后重写缓存。
    """
    proc_path, _ = _ppa_paths(spec.root)
    cache = proc_path.parent / 'labels_cache.pt'
    labels = None
    if cache.exists():
        labels = torch.load(str(cache), weights_only=True).get('labels')
    if labels is not None and len(labels) >= min_len:
        return labels
    _, label_dict = dgl.load_graphs(str(proc_path), idx_list=[0])
    cand = label_dict.get('labels')
    if not isinstance(cand, torch.Tensor):
        cand = torch.as_tensor(cand)
    if len(cand) >= min_len:
        labels = cand
    else:
        print(f'  PPA labels: idx_list 读取不足（{len(cand)} < {min_len}），'
              '回退全量加载（一次性）')
        _, label_dict = dgl.load_graphs(str(proc_path))
        labels = label_dict['labels']
    torch.save({'labels': labels}, str(cache))
    return labels


def load_ppa(spec: DatasetSpec) -> Dataset:
    if not spec.root:
        raise ValueError(
            f"数据集 '{spec.name}' 数据目录未指定：必须通过 --root 传入"
            "（如 --root D:\\data\\ogbg_ppa）")

    proc_path, split_dir = _ppa_paths(spec.root)

    out = {'train': ([], []), 'valid': ([], []), 'test': ([], [])}
    kept_ids = {}
    for split in spec.splits:
        # 升序 + 截断：保证 graphs[i] 与 indices[i] 严格对齐
        ids = sorted(_read_split_ids(split_dir, split))
        if spec.max_graphs > 0 and ids:
            if spec.stratified:
                # 分层抽样：按类别比例从全 split 选 N（固定种子），仍升序
                all_labels = _ppa_all_labels(spec, min_len=max(ids) + 1)
                ids = stratified_take(ids, lambda i: int(all_labels[i]),
                                      spec.max_graphs)
            else:
                ids = ids[:spec.max_graphs]
        graphs, labels = _load_graphs(proc_path, ids, spec.needs_self_loop,
                                      desc=f"PPA {split}")
        out[split] = (graphs, labels)
        kept_ids[split] = ids

    train_graphs, train_labels = out['train']
    valid_graphs, valid_labels = out['valid']
    test_graphs, test_labels = out['test']
    return Dataset(
        spec=spec,
        train_graphs=train_graphs,
        train_labels=train_labels,
        valid_graphs=valid_graphs,
        valid_labels=valid_labels,
        test_graphs=test_graphs,
        test_labels=test_labels,
        train_indices=kept_ids.get('train', []),
        valid_indices=kept_ids.get('valid', []),
        test_indices=kept_ids.get('test', []),
    )


def load_ppa_subset(spec: DatasetSpec, needed: Iterable[int]) -> Dict[int, Tuple[dgl.DGLGraph, int]]:
    """按 OGB train 索引选择性加载（BC 轨迹/诊断用），返回 {idx: (graph, label)}。

    只加载训练集内的索引；训练集外的索引警告并跳过（与 m5 行为一致）。
    """
    if not spec.root:
        raise ValueError(
            f"数据集 '{spec.name}' 数据目录未指定：必须通过 --root 传入")
    proc_path, split_dir = _ppa_paths(spec.root)

    train_set = set(_read_split_ids(split_dir, 'train'))
    needed = set(int(i) for i in needed)
    to_load = sorted(needed & train_set)
    skipped = needed - train_set
    if skipped:
        print(f"  WARNING: {len(skipped)} graph_idx 不在训练集内，已忽略")

    graph_cache: Dict[int, Tuple[dgl.DGLGraph, int]] = {}
    if not to_load:
        return graph_cache

    graphs, label_dict = dgl.load_graphs(str(proc_path), idx_list=to_load)
    for i, ogb_idx in enumerate(tqdm(to_load, desc="Loading PPA subset")):
        g = dgl.add_self_loop(graphs[i]) if spec.needs_self_loop else graphs[i]
        graph_cache[ogb_idx] = (g, int(label_dict['labels'][ogb_idx].item()))
    return graph_cache


register(DatasetSpec(
    name='ppa',
    n_classes=37,
    node_feat_dim=0,
    edge_feat_dim=7,
    needs_self_loop=True,
    budget_options=(0.05, 0.10, 0.15),
    load=load_ppa,
    load_graph_subset=load_ppa_subset,
))
