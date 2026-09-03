"""数据集分布统计（registry 驱动，任意注册数据集）。

对每个非空 split 统计：类别分布表、图规模分布（节点数 / 边数，剔除自环）、
特征信息、划分占比与预算档对应的揭示规模。输出（样式参考
docs/progress/daily/figures/ppa_analysis_report.txt / ppa_analysis_plots.png）：
- {dataset}_analysis_report.txt   文本报告
- {dataset}_analysis_plots.png    2x3 面板图
- {dataset}_analysis_stats.json   机器可读统计

用法:
  python code/diagnosis/analyze_dataset_distribution.py --dataset cifar10 \
      --root D:\\data\\superpixels
  python code/diagnosis/analyze_dataset_distribution.py --dataset collab \
      --root D:\\data\\collab

输出目录: code/data/{dataset}/dataset-distribution/
"""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402

SPLIT_ORDER = ('train', 'valid', 'test')


def fmt(v: float, sep: bool = True) -> str:
    """均值等保留 1 位小数，千位分隔（对齐参考报告样式）。"""
    s = f'{v:,.1f}' if sep else f'{v:.1f}'
    return s


def graph_scale_stats(graphs):
    """逐图 (节点数, 剔除自环后的边数)。

    加载器保证自环图每节点恰 1 条自环（collab 先 remove 再 add；
    cifar10 原图无自环统一 add），故 real_edges = num_edges - num_nodes。
    首图校验该假设，不成立时退回逐图数边。
    """
    nodes = [g.num_nodes() for g in graphs]
    if graphs:
        u, v = graphs[0].edges()
        if int((u == v).sum()) == graphs[0].num_nodes():
            return nodes, [g.num_edges() - g.num_nodes() for g in graphs]
    edges = []
    for g in graphs:
        u, v = g.edges()
        edges.append(g.num_edges() - int((u == v).sum()))
    return nodes, edges


def edge_storage(graphs, n_sample: int = 3) -> str:
    """抽样判断非自环边是否双向存储（(u,v) 与 (v,u) 成对）。"""
    n_checked = n_bidir = 0
    for g in graphs[:n_sample]:
        u, v = g.edges()
        u, v = u[:5000], v[:5000]
        keep = u != v
        pairs = set(zip(u[keep].tolist(), v[keep].tolist()))
        n_checked += len(pairs)
        n_bidir += sum(1 for a, b in pairs if (b, a) in pairs)
    if n_checked == 0:
        return '无非自环边'
    ratio = n_bidir / n_checked
    if ratio >= 0.99:
        return '双向'
    if ratio >= 0.9:
        return f'双向为主（双向边占比 {ratio:.0%}）'
    return f'单向（双向边占比 {ratio:.0%}）'


def edge_feat_is_placeholder(graphs, n_sample: int = 20) -> bool:
    """采样检查边特征是否全零（占位注入）。"""
    for g in graphs[:n_sample]:
        if 'feat' not in g.edata or g.edata['feat'].numel() == 0:
            return True
        if bool(g.edata['feat'].any()):
            return False
    return True


def node_feat_stats(graphs) -> dict:
    g = graphs[0]
    if 'feat' not in g.ndata or g.ndata['feat'].numel() == 0:
        return {'present': False, 'dim': 0}
    return {'present': True, 'dim': int(g.ndata['feat'].shape[1])}


def dist_stats(arr) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {'min': float(a.min()), 'max': float(a.max()),
            'mean': float(a.mean()), 'median': float(np.median(a)),
            'std': float(a.std())}


def class_table(labels, n_classes: int) -> dict:
    counts = np.bincount(np.asarray(labels, dtype=np.int64),
                         minlength=n_classes)[:n_classes]
    return {c: int(counts[c]) for c in range(n_classes)}


def budget_lines(nodes, budget_options) -> list:
    lines = []
    a = np.asarray(nodes, dtype=np.float64)
    for r in budget_options:
        b = np.maximum(1, (a * r).astype(int))
        lines.append((r, float(b.mean()), float(np.median(b))))
    return lines


# ----------------------------------------------------------------------
# 报告渲染
# ----------------------------------------------------------------------

def render_report(spec, split_stats: dict, out_lines: list):
    p = out_lines.append
    total = sum(s['n_graphs'] for s in split_stats.values())
    first = split_stats.get('train') or next(iter(split_stats.values()))

    p('')
    p('=' * 70)
    p(f'  {spec.name} 数据集分析报告  |  Summary')
    p('=' * 70)
    p('')
    p('数据集基本信息:')
    p(f'  - 名称           : {spec.name}')
    p(f'  - 总图数         : {total:,}')
    p(f'  - 类别数         : {spec.n_classes}')
    p(f'  - 预算档位       : {", ".join(f"{r:g}" for r in spec.budget_options)}')
    p('')
    p('图结构统计（train split）:')
    ns, es = first['node_stats'], first['edge_stats']
    p(f'  - 节点数           : min={ns["min"]:,.0f}, max={ns["max"]:,.0f},')
    p(f'                       mean={fmt(ns["mean"])}, median={fmt(ns["median"], sep=False)}')
    p(f'  - 边数（剔除自环） : min={es["min"]:,.0f}, max={es["max"]:,.0f},')
    p(f'                       mean={fmt(es["mean"])}, median={fmt(es["median"], sep=False)}')
    p('')
    p('特征信息:')
    nf = first['node_feat']
    p(f'  - 节点特征        : {"存在 (维度=%d)" % nf["dim"] if nf["present"] else "缺失"}')
    ef = first['edge_feat']
    if ef['present']:
        extra = '，采样全零 → 占位注入' if ef['placeholder'] else ''
        p(f'  - 边特征          : 存在 (维度={ef["dim"]}{extra})')
    else:
        p('  - 边特征          : 缺失')
    p('')
    p('数据集划分:')
    for name in SPLIT_ORDER:
        if name in split_stats:
            s = split_stats[name]
            p(f'  - {name:<6}: {s["n_graphs"]:>8,} 图 '
              f'({s["n_graphs"] / total:.1%})')
    p('')
    cc = first['class_counts']
    vals = [v for v in cc.values() if v > 0]
    p('类别不平衡（train）:')
    p(f'  - max/min ratio   : {max(vals) / min(vals):.2f}')
    p('')
    p('核心结论:')
    p(f'  1. 边特征: {"是, 维度 = %d%s" % (ef["dim"], "（全零占位）" if ef["placeholder"] else "") if ef["present"] else "无"}')
    p(f'  2. 节点特征: {"存在, 维度 = %d" % nf["dim"] if nf["present"] else "缺失"}')
    p(f'  3. 边存储: {first["bidir"]}；自环: 加载器已按协议添加'
      f'（每节点 1 条），边数统计已剔除')
    p(f'  4. train 类别不平衡 max/min = {max(vals) / min(vals):.2f}')
    concl_no = 5
    for r, bm, bmed in first['budget_lines']:
        p(f'  {concl_no}. 预算 {r:g} → 平均揭示 B={bm:.1f} 节点（median {bmed:.0f}，'
          f'约 {bm / ns["mean"]:.1%} 规模）')
        concl_no += 1

    for name in SPLIT_ORDER:
        if name not in split_stats:
            continue
        s = split_stats[name]
        p('')
        p('=' * 70)
        p(f'=== {name.upper()} ({s["n_graphs"]:,} graphs) ===')
        p(' Class    Count      Pct')
        p('-' * 25)
        for c, n in s['class_counts'].items():
            p(f'{c:>6} {n:>9,} {n / s["n_graphs"]:>7.2%}')
        positive = {c: n for c, n in s['class_counts'].items() if n > 0}
        vals = list(positive.values())
        cmin = min(positive, key=lambda c: (positive[c], c))
        cmax = max(positive, key=lambda c: (positive[c], c))
        mean_c = float(np.mean(list(s['class_counts'].values())))
        std_c = float(np.std(list(s['class_counts'].values())))
        p('')
        p(f'  Min: {min(vals):,} (class {cmin}), Max: {max(vals):,} '
          f'(class {cmax}), Mean: {fmt(mean_c)}, Std: {fmt(std_c)}')
        p(f'  Imbalance ratio (max/min): {max(vals) / min(vals):.2f}')


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------

def render_plots(spec, split_stats: dict, out_path: Path):
    train = split_stats.get('train') or next(iter(split_stats.values()))
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(f'{spec.name} Dataset Analysis', fontsize=16, fontweight='bold')

    ax = axes[0][0]
    ax.hist(train['nodes'], bins=40, color='#5DADE2', edgecolor='none')
    ax.axvline(train['node_stats']['mean'], color='red', linestyle='--',
               label=f'Mean={train["node_stats"]["mean"]:.0f}')
    ax.axvline(train['node_stats']['median'], color='orange', linestyle='--',
               label=f'Median={train["node_stats"]["median"]:.0f}')
    ax.set_title('Node Count Distribution')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Frequency')
    ax.legend()

    ax = axes[0][1]
    ax.hist(train['edges'], bins=40, color='#52BE80', edgecolor='none')
    ax.axvline(train['edge_stats']['mean'], color='red', linestyle='--',
               label=f'Mean={train["edge_stats"]["mean"]:.0f}')
    ax.axvline(train['edge_stats']['median'], color='orange', linestyle='--',
               label=f'Median={train["edge_stats"]["median"]:.0f}')
    ax.set_title('Edge Count Distribution (excl. self-loops)')
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Frequency')
    ax.legend()

    ax = axes[0][2]
    ax.scatter(train['nodes'], train['edges'], s=4, alpha=0.3, color='#7D3C98')
    ax.set_title('Nodes vs Edges per Graph')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Number of Edges')

    ax = axes[1][0]
    cc = train['class_counts']
    ids = list(cc.keys())
    ax.bar(ids, [cc[i] for i in ids], color='#F5B041')
    uniform = train['n_graphs'] / max(1, sum(1 for v in cc.values() if v > 0))
    ax.axhline(uniform, color='red', linestyle='--',
               label=f'Uniform ({uniform:.0f})')
    ax.set_title(f'Class Distribution (train, {spec.n_classes} classes)')
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Number of Graphs')
    ax.legend()

    ax = axes[1][1]
    names = [n for n in SPLIT_ORDER if n in split_stats]
    sizes = [split_stats[n]['n_graphs'] for n in names]
    ax.pie(sizes, labels=names, autopct='%1.1f%%', startangle=90)
    ax.set_title('Dataset Split')

    ax = axes[1][2]
    sorted_nodes = np.sort(np.asarray(train['nodes']))
    ax.plot(sorted_nodes, np.arange(1, len(sorted_nodes) + 1)
            / len(sorted_nodes), color='#C2185B')
    ax.set_title('Node Count CDF (train)')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('CDF')
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='数据集分布统计（registry 驱动）')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', default='',
                        help='数据集根目录（如 D:\\data\\superpixels）')
    parser.add_argument('--splits', nargs='+', default=None,
                        choices=list(SPLIT_ORDER),
                        help='要统计的 split（默认 train valid test 全部）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 最多加载图数（0=全部；冒烟用）')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 code/data/{dataset}/dataset-distribution）')
    args = parser.parse_args()

    splits = tuple(args.splits or SPLIT_ORDER)
    ds = get_dataset(args.dataset, root=args.root or None, splits=splits,
                     max_graphs=args.max_graphs)
    spec = ds.spec
    print(f'Dataset: {spec.name} | n_classes={spec.n_classes} '
          f'| node_feat_dim={spec.node_feat_dim} | edge_feat_dim={spec.edge_feat_dim}')

    split_stats: dict = {}
    for name in SPLIT_ORDER:
        if name not in splits:
            continue
        graphs = getattr(ds, f'{name}_graphs')
        labels = getattr(ds, f'{name}_labels')
        if not graphs:
            print(f'  {name}: 空，跳过')
            continue
        print(f'  统计 {name}: {len(graphs):,} graphs ...')
        nodes, edges = graph_scale_stats(graphs)
        split_stats[name] = {
            'n_graphs': len(graphs),
            'nodes': nodes,
            'edges': edges,
            'node_stats': dist_stats(nodes),
            'edge_stats': dist_stats(edges),
            'node_feat': node_feat_stats(graphs),
            'edge_feat': {
                'present': 'feat' in graphs[0].edata
                and graphs[0].edata['feat'].numel() > 0,
                'dim': spec.edge_feat_dim,
                'placeholder': edge_feat_is_placeholder(graphs),
            },
            'bidir': edge_storage(graphs),
            'class_counts': class_table(labels, spec.n_classes),
            'budget_lines': budget_lines(nodes, spec.budget_options),
        }

    out_dir = (Path(args.output_dir) if args.output_dir
               else CODE_DIR / 'data' / spec.name / 'dataset-distribution')
    if not out_dir.is_absolute():
        out_dir = CODE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list = []
    render_report(spec, split_stats, lines)
    report_path = out_dir / f'{spec.name}_analysis_report.txt'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Report: {report_path}')

    plots_path = out_dir / f'{spec.name}_analysis_plots.png'
    render_plots(spec, split_stats, plots_path)
    print(f'Plots : {plots_path}')

    # 机器可读版（不含逐图数组）
    json_stats = {}
    for name, s in split_stats.items():
        json_stats[name] = {k: v for k, v in s.items()
                            if k not in ('nodes', 'edges')}
    stats_path = out_dir / f'{spec.name}_analysis_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    print(f'Stats : {stats_path}')


if __name__ == '__main__':
    main()
