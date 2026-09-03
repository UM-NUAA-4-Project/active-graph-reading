"""registry 数据集 → GAM 输入格式（baseline/data/{dataset}/{split}/*.json）。

GAM（KDD 2018, boazlee/benedekrozemberczki 实现）要求每个图一个 JSON：
  {"target": 图类别, "edges": [[u,v],...],
   "labels": {节点: 节点标签}, "inverse_labels": {节点标签: [节点]}}
节点标签必须是离散类别（GAM 的 step 网络按"标签注意力"选下一步走向）。

对齐决策（与 Active Graph Reader 协议的差异，见 baseline/README.md）：
- collab 无节点标签 → 默认按每图度数分桶（quantile，--label-buckets 桶）
  合成离散标签；--node-labels constant 可退化为全 0（此时 GAM 退化为
  随机游走 + LSTM 分类）。
- 边一律剔除协议自环（GAM 看原始图，不含 reader 协议加的自环）。
- 节点重映射为 0..n-1 连续 id。

用法:
  python baseline/convert_dataset.py --dataset collab --root D:\\data\\collab \
      [--node-labels degree] [--label-buckets 4] [--max-graphs 0]
"""

import argparse
import json
import random
import sys
from pathlib import Path

_BASELINE_DIR = Path(__file__).resolve().parent
_CODE_DIR = _BASELINE_DIR.parent / 'code'
sys.path.insert(0, str(_CODE_DIR))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from datasets import get_dataset  # noqa: E402

SPLIT_ORDER = ('train', 'valid', 'test')


def compute_degree_labels(edges, n: int, k_buckets: int) -> dict:
    deg = np.zeros(n, dtype=np.int64)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    if k_buckets > 1 and len(np.unique(deg)) > 1:
        qs = np.quantile(deg, np.linspace(0, 1, k_buckets + 1)[1:-1])
        return {i: int(np.searchsorted(qs, deg[i])) for i in range(n)}
    return {i: 0 for i in range(n)}


def graph_to_gam_json(g, target: int, labels: dict) -> dict:
    """DGLGraph → GAM JSON dict（剔除自环、去重双向边；labels 由调用方按模式算好）。"""
    n = g.num_nodes()
    u, v = g.edges()
    edges = sorted({(min(a, b), max(a, b))
                    for a, b in zip(u.tolist(), v.tolist()) if a != b})
    inverse = {}
    for node, lab in labels.items():
        inverse.setdefault(str(lab), []).append(node)
    return {
        'target': int(target),
        'edges': [[int(a), int(b)] for a, b in edges],
        'labels': {str(node): lab for node, lab in labels.items()},
        'inverse_labels': inverse,
    }


def main():
    parser = argparse.ArgumentParser(description='registry 数据集 → GAM JSON')
    parser.add_argument('--dataset', default='collab')
    parser.add_argument('--root', default='', help='数据集根目录')
    parser.add_argument('--splits', nargs='+', default=list(SPLIT_ORDER),
                        choices=list(SPLIT_ORDER))
    parser.add_argument('--node-labels', default='degree',
                        choices=['degree', 'constant'],
                        help='合成节点标签方式（collab/ppa 无节点标签；'
                             'degree=每图度数分桶，constant=全 0）')
    parser.add_argument('--label-buckets', type=int, default=4,
                        help='degree 模式的分桶数')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 最多导出图数（0=全部；冒烟测试用）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 baseline/data/{dataset}）')
    args = parser.parse_args()

    ds = get_dataset(args.dataset, root=args.root or None, splits=args.splits)
    out_dir = (Path(args.output_dir) if args.output_dir
               else _BASELINE_DIR / 'data' / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _labels_for(g, edges):
        if args.node_labels == 'constant':
            return {i: 0 for i in range(g.num_nodes())}
        return compute_degree_labels(edges, g.num_nodes(), args.label_buckets)

    rng = random.Random(args.seed)
    meta = {'dataset': args.dataset, 'node_labels': args.node_labels,
            'label_buckets': args.label_buckets if args.node_labels != 'constant' else 1,
            'seed': args.seed, 'splits': {}}
    for split in args.splits:
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if not graphs:
            print(f'{split}: 空，跳过')
            continue
        idx = list(range(len(graphs)))
        if args.max_graphs > 0 and len(idx) > args.max_graphs:
            rng.shuffle(idx)
            idx = idx[:args.max_graphs]
            print(f'{split}: 采样 {args.max_graphs}/{len(graphs)}')
        split_dir = out_dir / split
        split_dir.mkdir(exist_ok=True)
        n_by_class = {}
        for i in tqdm(idx, desc=f'convert {split}'):
            g = graphs[i]
            u, v = g.edges()
            edges = sorted({(min(a, b), max(a, b))
                            for a, b in zip(u.tolist(), v.tolist()) if a != b})
            data = graph_to_gam_json(g, int(labels[i]),
                                     _labels_for(g, edges))
            with open(split_dir / f'graph_{i}.json', 'w') as f:
                json.dump(data, f, separators=(',', ':'))
            n_by_class[data['target']] = n_by_class.get(data['target'], 0) + 1
        meta['splits'][split] = {'n_graphs': len(idx),
                                 'by_class': dict(sorted(n_by_class.items()))}
        print(f'{split}: {len(idx)} graphs → {split_dir} | classes {dict(sorted(n_by_class.items()))}')

    with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'Meta: {out_dir / "meta.json"}')


if __name__ == '__main__':
    main()
