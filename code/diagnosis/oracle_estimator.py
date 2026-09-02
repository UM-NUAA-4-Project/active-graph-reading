"""Oracle Type II 性能上界估计 + BFS/random 基线对比（任意注册数据集）。

自 m4/oracle_estimator.py 泛化迁移：
- 数据走 datasets 注册表（--dataset/--root 任意切换）。有 load_graph_subset
  的数据集（PPA）：分层采样读 split/标签 CSV（不加载全量图），仅加载被采样
  图；其它（cifar10）：候选 = 已加载 train split 的本地位置；
- 分类器经 train_reader.core.load_classifier_frozen（requires_grad=True：
  GradientTeacher 需要反向传播，冻结仅指 eval 模式 + 不做 optimizer 更新）；
- BFS/random 基线走 active_graph_reader.reader.fixed.create_fixed_reader +
  train_reader.core.rollout_strategy（替代旧 FixedStrategyReader/run_reveal）；
- gradient/structural teacher 自 trajectories.teachers 导入（原 m4 实现）。

在训练集上，由 gradient / structural-degree teacher 分别选出最优 seed，
从每个 seed 出发运行 4 种 reader（gradient, structural-degree, BFS, random），
预算花光后分类器评估准确率。

产出: oracle_report.json / oracle_accuracy.png / oracle_variance.png

Usage:
  python code/diagnosis/oracle_estimator.py \
    --pretrained m3/best_model_ft.pt \
    --dataset ppa --root D:\\ogb-master\\dataset \
    --device cuda:0 --k-per-class 10 --budgets-rel 0.05 0.10 0.15
"""

import argparse
import gc
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import dgl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from active_graph_reader.reader.fixed import create_fixed_reader  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools import cli  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from train_reader.core import (load_classifier_frozen,  # noqa: E402
                               rollout_strategy)
from trajectories.teachers import GradientTeacher, StructuralTeacher  # noqa: E402

SEED_SOURCES = ['gradient', 'structural-degree']
READER_NAMES = ['gradient', 'structural-degree', 'bfs', 'random']


def resolve(p) -> Path:
    """相对路径按 repo root 解析（绝对路径原样）。"""
    path = Path(p)
    return path if path.is_absolute() else CODE_DIR / path


# ---------------------------------------------------------------------------
# 候选池（与 diagnose_checkpoint 同口径：PPA 走 CSV，不加载图数据）
# ---------------------------------------------------------------------------

def _ppa_csv_pool(spec):
    """PPA 类数据集：split CSV + 标签 CSV 读全量 train 候选。"""
    root = Path(spec.root)
    split_csv = (root / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz')
    label_csv = (root / 'ogbg_ppa' / 'raw' / 'graph-label.csv.gz')
    candidates = pd.read_csv(split_csv, compression='gzip', header=None
                             ).values.flatten().tolist()
    label_arr = pd.read_csv(label_csv, compression='gzip', header=None
                            ).values.flatten()
    lookup = {int(i): int(label_arr[i]) for i in candidates}
    return candidates, lookup


def build_train_pool(spec, ds):
    """返回 (candidates, labels_lookup)。

    PPA（load_graph_subset）→ 完整 train split 的 OGB 索引（标签读 CSV，
    不加载图）；其它 → 已加载 train_graphs 的本地位置。
    """
    if spec.load_graph_subset is not None:
        return _ppa_csv_pool(spec)
    labels = ds.train_labels
    candidates = list(range(len(labels)))
    lookup = {i: int(labels[i]) for i in candidates}
    return candidates, lookup


# ---------------------------------------------------------------------------
# Stratified sampling（与旧脚本同口径）
# ---------------------------------------------------------------------------

def sample_stratified(candidates: List[int], labels_lookup: Dict[int, int],
                      k_per_class: int, n_classes: int, seed: int = 42
                      ) -> List[int]:
    """按类各取 k_per_class 张图（不足则全取），返回采样的 graph_idx 列表。"""
    rng = random.Random(seed)
    class_buckets: Dict[int, List[int]] = defaultdict(list)
    for idx in candidates:
        class_buckets[int(labels_lookup[int(idx)])].append(idx)

    selected = []
    for lbl in range(n_classes):
        bucket = class_buckets.get(lbl, [])
        if not bucket:
            continue
        n_pick = min(k_per_class, len(bucket))
        selected.extend(rng.sample(bucket, n_pick))
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Full-graph accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full_graph(classifier, graphs, labels, device,
                        batch_size: int = 64) -> float:
    """全图（不揭示）直接分类的 test 准确率。"""
    correct, total = 0, 0
    for i in tqdm(range(0, len(graphs), batch_size),
                  desc="full-graph eval", leave=False):
        batch_graphs = graphs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_g = dgl.batch(batch_graphs).to(device)
        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long,
                               device=device)
        logits = classifier(batch_g, node_ids)
        preds = logits.argmax(dim=1).cpu().tolist()
        for pred, label in zip(preds, batch_labels):
            total += 1
            if pred == int(label):
                correct += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Core: Oracle Type II estimation
# ---------------------------------------------------------------------------

def estimate_oracle(
    graphs: List,
    graph_labels: List[int],
    classifier: torch.nn.Module,
    budgets_rel: List[float],
    device: torch.device,
    graph_batch_size: int,
) -> Dict[str, Dict[str, Dict[float, List[float]]]]:
    """Run Oracle Type II estimation.

    graphs/graph_labels: 已加载的被采样训练图与标签（对齐）。
    Returns: results[seed_source][reader][budget_rel] = [0/1, ...]
    """
    grad_teacher = GradientTeacher()
    struct_teacher = StructuralTeacher(method='degree')
    builder = SubgraphBuilder()

    bfs_agent = create_fixed_reader('bfs')
    random_agent = create_fixed_reader('random')

    br_sorted = sorted(budgets_rel)

    results: Dict[str, Dict[str, Dict[float, List[float]]]] = {
        src: {rdr: defaultdict(list) for rdr in READER_NAMES}
        for src in SEED_SOURCES
    }

    n_total = len(graphs)

    for batch_start in tqdm(range(0, n_total, graph_batch_size),
                            desc="oracle estimation"):
        batch_positions = list(range(batch_start,
                                     min(batch_start + graph_batch_size,
                                         n_total)))

        # ---- Step 1: per-graph scores + seeds ----
        batch_data = {}
        for pos in batch_positions:
            g_raw = graphs[pos]
            label = int(graph_labels[pos])
            n_nodes = g_raw.num_nodes()
            budgets_abs = [max(1, int(r * n_nodes)) for r in br_sorted]

            # Precompute edge_key_to_id
            edge_key_to_id = {}
            src_edge = g_raw.edges()[0]
            dst_edge = g_raw.edges()[1]
            for eid in range(g_raw.num_edges()):
                u, v = int(src_edge[eid]), int(dst_edge[eid])
                key = (min(u, v), max(u, v))
                if key not in edge_key_to_id:
                    edge_key_to_id[key] = eid

            grad_scores = grad_teacher.score_nodes(g_raw, classifier, label)
            struct_scores = struct_teacher.score_nodes(g_raw)
            grad_seed = int(grad_scores.argmax().item())
            struct_seed = int(struct_scores.argmax().item())

            batch_data[pos] = {
                'graph': g_raw,
                'label': label,
                'grad_scores': grad_scores,
                'struct_scores': struct_scores,
                'grad_seed': grad_seed,
                'struct_seed': struct_seed,
                'budgets_abs': budgets_abs,
                'edge_key_to_id': edge_key_to_id,
            }

        # ---- Step 2: run all reveals, collect (state, meta) ----
        all_states = []
        all_meta = []  # (pos, seed_src, reader, budget_rel, label)

        for pos in batch_positions:
            bd = batch_data[pos]
            g_raw = bd['graph']
            label = bd['label']
            budgets_abs = bd['budgets_abs']
            grad_scores = bd['grad_scores']
            struct_scores = bd['struct_scores']
            ek2id = bd['edge_key_to_id']

            for seed_src in SEED_SOURCES:
                seed = bd['grad_seed'] if seed_src == 'gradient' \
                    else bd['struct_seed']

                for br, abs_b in zip(br_sorted, budgets_abs):
                    # Gradient reader
                    _, st = grad_teacher.generate_trajectory(
                        g_raw, grad_scores, seed, abs_b, edge_key_to_id=ek2id,
                    )
                    all_states.append(st)
                    all_meta.append((pos, seed_src, 'gradient', br, label))

                    # Structural-degree reader
                    _, st = struct_teacher.generate_trajectory(
                        g_raw, struct_scores, seed, abs_b, edge_key_to_id=ek2id,
                    )
                    all_states.append(st)
                    all_meta.append(
                        (pos, seed_src, 'structural-degree', br, label))

                    # BFS reader（create_fixed_reader + rollout_strategy）
                    state = GraphState(g_raw, seed_node=seed,
                                       edge_key_to_id=ek2id)
                    state._budget_limit = abs_b
                    rollout_strategy(bfs_agent, state)
                    all_states.append(state)
                    all_meta.append((pos, seed_src, 'bfs', br, label))

                    # Random reader
                    state = GraphState(g_raw, seed_node=seed,
                                       edge_key_to_id=ek2id)
                    state._budget_limit = abs_b
                    rollout_strategy(random_agent, state)
                    all_states.append(state)
                    all_meta.append((pos, seed_src, 'random', br, label))

        # ---- Step 3: batch build + classify ----
        batched_g, _ = builder.build_batch(all_states)
        batched_g = batched_g.to(device)
        node_ids = torch.zeros(batched_g.num_nodes(), dtype=torch.long,
                               device=device)

        with torch.no_grad():
            logits = classifier(batched_g, node_ids)
            preds = logits.argmax(dim=1).cpu().tolist()

        # ---- Step 4: aggregate ----
        for flat_i, (_pos, seed_src, rdr, br, label) in enumerate(all_meta):
            correct = 1.0 if preds[flat_i] == label else 0.0
            results[seed_src][rdr][br].append(correct)

        # Cleanup
        del all_states, all_meta, batched_g, batch_data
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting: grouped bar charts
# ---------------------------------------------------------------------------

def plot_oracle_charts(
    results: Dict,
    budgets_rel: List[float],
    full_graph_acc: float,
    output_dir: Path,
):
    """Generate accuracy and variance grouped bar charts."""
    reader_labels = {
        'gradient': 'Gradient',
        'structural-degree': 'Struct-Deg',
        'bfs': 'BFS',
        'random': 'Random',
    }
    colors = {
        'gradient': '#2ecc71',
        'structural-degree': '#3498db',
        'bfs': '#f39c12',
        'random': '#e74c3c',
    }
    br_sorted = sorted(budgets_rel)
    n_budgets = len(br_sorted)
    n_readers = len(READER_NAMES)
    bar_width = 0.18
    x = np.arange(n_budgets)

    # ---- Figure 1: Accuracy ----
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig1.suptitle('Oracle Type II — Accuracy (Training Set)',
                  fontsize=14, fontweight='bold')

    for sub_i, seed_src in enumerate(SEED_SOURCES):
        ax = axes1[sub_i]
        title = ('Seed: Gradient Best' if seed_src == 'gradient'
                 else 'Seed: Structural-Deg Best')
        ax.set_title(title, fontsize=13)

        for r_i, rdr in enumerate(READER_NAMES):
            means = [np.mean(results[seed_src][rdr].get(br, [0.0]))
                     for br in br_sorted]
            offset = (r_i - n_readers / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, means, bar_width,
                          color=colors[rdr], label=reader_labels[rdr],
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                        rotation=90)

        ax.axhline(y=full_graph_acc, color='black', linestyle='--',
                   linewidth=1.5, label=f'Full-graph ({full_graph_acc:.3f})')
        ax.set_xlabel('Budget (ratio of |V|)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b:.0%}' for b in br_sorted])
        if sub_i == 0:
            ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(full_graph_acc * 1.15, 0.15))

    fig1.tight_layout()
    acc_path = output_dir / 'oracle_accuracy.png'
    fig1.savefig(acc_path, dpi=150)
    plt.close(fig1)
    print(f"Accuracy chart saved to {acc_path}")

    # ---- Figure 2: Standard deviation ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig2.suptitle('Oracle Type II — Standard Deviation (Training Set)',
                  fontsize=14, fontweight='bold')

    for sub_i, seed_src in enumerate(SEED_SOURCES):
        ax = axes2[sub_i]
        title = ('Seed: Gradient Best' if seed_src == 'gradient'
                 else 'Seed: Structural-Deg Best')
        ax.set_title(title, fontsize=13)

        for r_i, rdr in enumerate(READER_NAMES):
            stds = []
            for br in br_sorted:
                accs = results[seed_src][rdr].get(br, [0.0])
                stds.append(np.std(accs) if len(accs) > 1 else 0.0)
            offset = (r_i - n_readers / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, stds, bar_width,
                          color=colors[rdr], label=reader_labels[rdr],
                          edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, stds):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7,
                        rotation=90)

        ax.set_xlabel('Budget (ratio of |V|)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b:.0%}' for b in br_sorted])
        if sub_i == 0:
            ax.set_ylabel('Standard Deviation', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    fig2.tight_layout()
    var_path = output_dir / 'oracle_variance.png'
    fig2.savefig(var_path, dpi=150)
    plt.close(fig2)
    print(f"Variance chart saved to {var_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(results, budgets_rel, full_graph_acc, config) -> Dict:
    br_sorted = sorted(budgets_rel)

    report_results = {}
    for src in SEED_SOURCES:
        report_results[src] = {}
        for rdr in READER_NAMES:
            report_results[src][rdr] = {}
            for br in br_sorted:
                accs = results[src][rdr].get(br, [])
                if accs:
                    report_results[src][rdr][f'{br:.2f}'] = {
                        'mean': round(float(np.mean(accs)), 4),
                        'std': round(float(np.std(accs)), 4),
                        'min': round(float(np.min(accs)), 4),
                        'max': round(float(np.max(accs)), 4),
                        'n': len(accs),
                    }

    headroom = {}
    for src in SEED_SOURCES:
        headroom[src] = {}
        for br in br_sorted:
            oracle_mean = np.mean(results[src]['gradient'].get(br, [0]))
            bfs_mean = np.mean(results[src]['bfs'].get(br, [0]))
            rand_mean = np.mean(results[src]['random'].get(br, [0]))
            headroom[src][f'{br:.2f}'] = {
                'oracle_vs_bfs': round(oracle_mean - bfs_mean, 4),
                'oracle_vs_random': round(oracle_mean - rand_mean, 4),
                'bfs_vs_random': round(bfs_mean - rand_mean, 4),
            }

    return {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'full_graph_accuracy': round(full_graph_acc, 4),
        'results': report_results,
        'headroom': headroom,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Oracle Type II — Performance Upper Bound Estimation')
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to frozen classifier checkpoint')
    cli.add_dataset_args(p, default_dataset='ppa')
    p.add_argument('--budgets-rel', type=float, nargs='+',
                   default=[0.05, 0.10, 0.15],
                   help='Budget as fraction of |V|')
    p.add_argument('--k-per-class', type=int, default=10,
                   help='Graphs per class（--max-graphs > 0 时被覆盖）')
    p.add_argument('--output-dir', type=str, default='',
                   help='输出目录（默认 result/{dataset}/oracle）')
    p.add_argument('--graph-batch-size', type=int, default=16,
                   help='Graphs per mini-batch')
    p.add_argument('--full-graph-acc', type=float, default=None,
                   help='Pre-computed full-graph accuracy (skip eval)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.k_per_class <= 0 and args.max_graphs <= 0:
        raise ValueError("--k-per-class or --max-graphs required")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = (resolve(args.output_dir) if args.output_dir else
                  CODE_DIR / 'result' / args.dataset / 'oracle')
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Oracle Type II — Performance Upper Bound Estimation")
    print("=" * 60)
    print(f"Budgets (rel): {args.budgets_rel}")
    print(f"Device:        {device}")
    print()

    # ---- 1. Dataset ----
    need_test = args.full_graph_acc is None
    ds = get_dataset(args.dataset, root=args.root,
                     splits=('train', 'test') if need_test else ('train',),
                     max_graphs=args.max_graphs)
    spec = ds.spec

    # 候选池：PPA（load_graph_subset）= CSV 全量 train 索引（不加载图）；
    # 其它 = 已加载 train_graphs 的本地位置
    candidates, labels_lookup = build_train_pool(spec, ds)
    print(f"[1/5] Dataset '{spec.name}' train split: "
          f"{len(candidates)} candidate graphs")

    if args.max_graphs > 0:
        # 旧语义：--max-graphs 覆盖 --k-per-class，从候选池随机抽 N 张
        sampled_idx = random.sample(candidates,
                                    min(args.max_graphs, len(candidates)))
    else:
        sampled_idx = sample_stratified(candidates, labels_lookup,
                                        args.k_per_class, spec.n_classes,
                                        seed=args.seed)

    # 图对象解析：PPA 经 load_graph_subset 精确加载；其它本地位置直取
    if spec.load_graph_subset is not None:
        pool = spec.load_graph_subset(spec, sampled_idx)
        missing = [i for i in sampled_idx if i not in pool]
        if missing:
            print(f"  WARNING: {len(missing)} sampled idx 无法加载（不在 train "
                  f"split 内？），已跳过")
            sampled_idx = [i for i in sampled_idx if i in pool]
        train_graphs = [pool[i][0] for i in sampled_idx]
        train_labels = [pool[i][1] for i in sampled_idx]
    else:
        train_graphs = [ds.train_graphs[i] for i in sampled_idx]
        train_labels = [labels_lookup[i] for i in sampled_idx]

    print(f"  Sampled {len(train_graphs)} training graphs")

    # ---- 2. Classifier ----
    print("\n[2/5] Loading frozen classifier...")
    pretrained_path = resolve(args.pretrained)
    pretrained_ckpt = torch.load(str(pretrained_path), map_location='cpu',
                                 weights_only=False)
    # NOTE: requires_grad=True — GradientTeacher 需要反向传播（冻结仅指
    # eval 模式；与旧版"不设 requires_grad=False"的注释一致）
    classifier = load_classifier_frozen(pretrained_ckpt, device,
                                        requires_grad=True)
    print(f"  Loaded from {pretrained_path}  "
          f"(val_acc={pretrained_ckpt.get('val_acc', 'N/A')})")

    # ---- 3. Full-graph accuracy ----
    if args.full_graph_acc is not None:
        full_graph_acc = args.full_graph_acc
        print(f"\n[3/5] Full-graph accuracy: {full_graph_acc:.4f} (provided)")
    else:
        test_graphs = ds.test_graphs
        test_labels = [int(v) for v in ds.test_labels]
        print(f"\n[3/5] Full-graph accuracy on test set "
              f"({len(test_graphs)} graphs)...")
        full_graph_acc = evaluate_full_graph(classifier, test_graphs,
                                             test_labels, device)
        print(f"  Full-graph test-acc = {full_graph_acc:.4f}")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- 4. Oracle estimation ----
    print(f"\n[4/5] Oracle Type II estimation ({len(train_graphs)} graphs × "
          f"2 seeds × 4 readers × {len(args.budgets_rel)} budgets)...")
    results = estimate_oracle(
        graphs=train_graphs,
        graph_labels=train_labels,
        classifier=classifier,
        budgets_rel=args.budgets_rel,
        device=device,
        graph_batch_size=args.graph_batch_size,
    )

    # Summary table
    print("\n  Results:")
    header = f"  {'Seed Source':<20} {'Reader':<20}"
    for br in sorted(args.budgets_rel):
        header += f"  B={br:.0%}   "
    print(header)
    print("  " + "-" * len(header))
    for src in SEED_SOURCES:
        for rdr in READER_NAMES:
            row = f"  {src:<20} {rdr:<20}"
            for br in sorted(args.budgets_rel):
                accs = results[src][rdr].get(br, [])
                m = np.mean(accs) if accs else 0.0
                row += f"  {m:.4f}"
            print(row)

    # Headroom
    print("\n  Headroom (oracle — baseline):")
    for src in SEED_SOURCES:
        print(f"    Seed: {src}")
        for br in sorted(args.budgets_rel):
            o = np.mean(results[src]['gradient'].get(br, [0]))
            bf = np.mean(results[src]['bfs'].get(br, [0]))
            rn = np.mean(results[src]['random'].get(br, [0]))
            print(f"      B={br:.0%}: oracle={o:.4f}, bfs={bf:.4f} "
                  f"(Δ={o - bf:+.4f}), random={rn:.4f} (Δ={o - rn:+.4f})")

    # ---- 5. Report + plots ----
    print(f"\n[5/5] Saving report and charts...")
    config = {
        'pretrained': str(pretrained_path),
        'dataset': spec.name,
        'root': args.root,
        'max_graphs': args.max_graphs if args.max_graphs > 0 else None,
        'k_per_class': args.k_per_class if args.max_graphs <= 0 else None,
        'n_graphs': len(train_graphs),
        'budgets_rel': args.budgets_rel,
        'seed': args.seed,
        'seed_sources': list(SEED_SOURCES),
        'readers': list(READER_NAMES),
    }
    report = build_report(results, args.budgets_rel, full_graph_acc, config)

    report_path = output_dir / 'oracle_report.json'
    save_json(report, report_path)
    print(f"  Report:   {report_path}")

    plot_oracle_charts(results, args.budgets_rel, full_graph_acc, output_dir)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
