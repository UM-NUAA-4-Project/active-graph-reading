"""M4 轨迹数据分析与可视化脚本。

加载合并后的轨迹文件，输出:
  1. 整体类别标签分布 (bar chart)
  2. 每种 teacher 的标签分布 (grouped subplots)
  3. 不同 teacher 各标签平均 CE (heatmap)
同时保存统计数据为 JSON。

用法:
  python ./code/m4/analyze_trajectories.py --input-dir ./code/m4/trimmed --output-dir ./code/m4/analysis
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent.parent  # process-data/ → m4/ → code/
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

N_CLASSES = 37


def parse_args():
    p = argparse.ArgumentParser(description='M4 Trajectory Analysis')
    p.add_argument('--input-dir', type=str, default=None,
                   help='Directory containing merged .pt files (default: m4/output)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory for charts and JSON (default: same as input-dir)')
    return p.parse_args()


def load_merged_files(input_dir: Path) -> dict:
    """加载所有.pt 文件, 返回 {ratio_str: data}."""
    files = sorted(input_dir.glob('trajectories_balanced_r*.pt'))
    if not files:
        logger.error("No balanced files found in %s", input_dir)
        sys.exit(1)

    result = {}
    for f in files:
        ratio_str = f.stem.replace('trajectories_balanced_r', '')
        data = torch.load(f, map_location='cpu', weights_only=False)
        result[ratio_str] = data
        logger.info("Loaded %s: %d graphs", f.name, data['config']['n_graphs'])
    return result


def load_all_ratio_files(input_dir: Path, glob_pattern: str = 'trajectories_balanced_r*.pt') -> dict:
    """加载所有 budget ratio 的文件, 返回 {ratio_float: [(teacher_name, traj), ...]}."""
    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        logger.error("No files matching '%s' found in %s", glob_pattern, input_dir)
        sys.exit(1)

    result = {}
    for f in files:
        stem = f.stem
        for prefix in ['trajectories_filtered_r', 'trajectories_merged_r', 'trajectories_balanced_r']:
            if stem.startswith(prefix):
                ratio_str = stem[len(prefix):]
                break
        else:
            ratio_str = stem.split('_r')[-1] if '_r' in stem else stem

        ratio_float = float(ratio_str.replace('_', '.'))
        data = torch.load(f, map_location='cpu', weights_only=False)
        flat = flatten_trajectories(data)
        result[ratio_float] = flat
        logger.info("Loaded %s: %d trajectories (ratio=%.3f)", f.name, len(flat), ratio_float)
    return result


def flatten_trajectories(data: dict):
    """将嵌套的 trajectories dict 展平为 [(teacher_name, traj), ...]."""
    trajs = data['trajectories']
    flat = []
    for teacher_name, per_graph in trajs.items():
        for graph_trajs in per_graph:
            for traj in graph_trajs:
                flat.append((teacher_name, traj))
    return flat


def compute_stats(all_flat: list, teacher_names: list):
    """计算统计量。

    all_flat: [(teacher_name, traj), ...]
    返回 stats dict。
    """
    # 1. 整体 label 分布
    label_counts = defaultdict(int)
    for _, traj in all_flat:
        label_counts[int(traj.final_pred)] += 1

    # 2. 每种 teacher 的 label 分布
    teacher_label_counts = {tn: defaultdict(int) for tn in teacher_names}
    # 3. 每种 teacher 各 label 的 CE 累加和计数（用于算均值）
    teacher_label_ce_sum = {tn: defaultdict(float) for tn in teacher_names}
    teacher_label_ce_cnt = {tn: defaultdict(int) for tn in teacher_names}

    for teacher_name, traj in all_flat:
        label = int(traj.final_pred)
        teacher_label_counts[teacher_name][label] += 1
        teacher_label_ce_sum[teacher_name][label] += traj.ce
        teacher_label_ce_cnt[teacher_name][label] += 1

    # 算平均 CE
    teacher_label_avg_ce = {}
    for tn in teacher_names:
        teacher_label_avg_ce[tn] = {}
        for label in range(N_CLASSES):
            s = teacher_label_ce_sum[tn].get(label, 0.0)
            c = teacher_label_ce_cnt[tn].get(label, 0)
            teacher_label_avg_ce[tn][label] = s / c if c > 0 else float('NaN')

    stats = {
        'total_trajectories': len(all_flat),
        'label_distribution': {str(k): label_counts.get(k, 0) for k in range(N_CLASSES)},
        'per_teacher': {},
    }
    for tn in teacher_names:
        stats['per_teacher'][tn] = {
            'total': sum(teacher_label_counts[tn].values()),
            'label_distribution': {str(k): teacher_label_counts[tn].get(k, 0)
                                   for k in range(N_CLASSES)},
            'per_label_avg_ce': {str(k): teacher_label_avg_ce[tn][k]
                                 for k in range(N_CLASSES)},
        }
    return stats


def plot_label_distribution(label_counts: dict, output_path: Path, title_suffix: str = ''):
    """柱状图: 整体类别标签分布."""
    labels = list(range(N_CLASSES))
    counts = [label_counts.get(k, 0) for k in labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, counts, color='steelblue', edgecolor='white')
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Trajectory Count')
    ax.set_title(f'Overall Label Distribution{title_suffix}')
    ax.set_xticks(labels)
    ax.set_xticklabels(labels, fontsize=7)

    # 标注高于均值的 bar
    mean_c = np.mean(counts)
    ax.axhline(y=mean_c, color='red', linestyle='--', linewidth=0.8,
               label=f'Mean ({mean_c:.1f})')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_teacher_label_distribution(teacher_label_counts: dict, output_path: Path,
                                     title_suffix: str = ''):
    """2×3 子图: 每种 teacher 的标签分布."""
    teacher_names = list(teacher_label_counts.keys())
    n_teachers = len(teacher_names)
    n_cols = 3
    n_rows = (n_teachers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_teachers > 1 else [axes]

    labels = list(range(N_CLASSES))
    for i, tn in enumerate(teacher_names):
        ax = axes[i]
        counts = [teacher_label_counts[tn].get(k, 0) for k in labels]
        ax.bar(labels, counts, color='coral', edgecolor='white', width=0.8)
        ax.set_title(tn, fontsize=9)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(labels[::4])
        ax.set_xticklabels(labels[::4], fontsize=6)

    # 隐藏多余的子图
    for j in range(n_teachers, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Per-Teacher Label Distribution{title_suffix}', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_ce_heatmap(teacher_label_avg_ce: dict, output_path: Path,
                     title_suffix: str = ''):
    """热力图: teacher × label 的平均 CE."""
    teacher_names = list(teacher_label_avg_ce.keys())
    labels = list(range(N_CLASSES))

    data = np.full((len(teacher_names), N_CLASSES), np.NaN)
    for i, tn in enumerate(teacher_names):
        for j, label in enumerate(labels):
            data[i, j] = teacher_label_avg_ce[tn].get(label, np.NaN)

    fig, ax = plt.subplots(figsize=(16, 1.2 + 0.5 * len(teacher_names)))

    # 用合适的 colormap, NaN 显示为灰色
    cmap = plt.cm.YlOrRd
    cmap.set_bad('lightgray')

    im = ax.imshow(data, aspect='auto', cmap=cmap)

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(len(teacher_names)))
    ax.set_yticklabels(teacher_names, fontsize=9)
    ax.set_xlabel('Class Label')
    ax.set_title(f'Average Cross-Entropy by Teacher and Label{title_suffix}')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Avg CE')

    # 在非 NaN 格子标注数值
    for i in range(len(teacher_names)):
        for j in range(N_CLASSES):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_per_label_ce_by_budget(all_ratio_data: dict, output_path: Path):
    """分组柱状图: 不同预算下每类标签的平均 CE. x=类别, 分组=budget ratio."""
    ratios = sorted(all_ratio_data.keys())

    ce_matrix = {}  # ce_matrix[label][ratio] = avg_ce
    for ratio in ratios:
        flat = all_ratio_data[ratio]
        label_ce_sum = defaultdict(float)
        label_ce_cnt = defaultdict(int)
        for _, traj in flat:
            label = int(traj.final_pred)
            label_ce_sum[label] += traj.ce
            label_ce_cnt[label] += 1
        for label in range(N_CLASSES):
            if label not in ce_matrix:
                ce_matrix[label] = {}
            s = label_ce_sum.get(label, 0.0)
            c = label_ce_cnt.get(label, 0)
            ce_matrix[label][ratio] = s / c if c > 0 else 0.0

    labels = list(range(N_CLASSES))
    n_ratios = len(ratios)
    x = np.arange(len(labels))
    width = 0.8 / n_ratios

    fig, ax = plt.subplots(figsize=(20, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_ratios))

    for i, ratio in enumerate(ratios):
        values = [ce_matrix[l].get(ratio, 0) for l in labels]
        offset = (i - n_ratios / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=f'ratio={ratio:.3f}',
               color=colors[i], edgecolor='white', linewidth=0.3)

    ax.set_xlabel('Class Label')
    ax.set_ylabel('Average Cross-Entropy')
    ax.set_title('Per-Class Average CE by Budget Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8, ncol=n_ratios)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_ce_trend_across_budgets(all_ratio_data: dict, output_path: Path):
    """折线图: 每类 CE 随 budget 变化趋势. x=揭示比例, y=平均 CE, 每条线一个类."""
    ratios = sorted(all_ratio_data.keys())

    class_ce_trend = {}
    for ratio in ratios:
        flat = all_ratio_data[ratio]
        label_ce_sum = defaultdict(float)
        label_ce_cnt = defaultdict(int)
        for _, traj in flat:
            label = int(traj.final_pred)
            label_ce_sum[label] += traj.ce
            label_ce_cnt[label] += 1
        for label in range(N_CLASSES):
            if label not in class_ce_trend:
                class_ce_trend[label] = {'ratios': [], 'ces': [], 'counts': []}
            s = label_ce_sum.get(label, 0.0)
            c = label_ce_cnt.get(label, 0)
            class_ce_trend[label]['ratios'].append(ratio)
            class_ce_trend[label]['ces'].append(s / c if c > 0 else float('NaN'))
            class_ce_trend[label]['counts'].append(c)

    class_total = {l: sum(class_ce_trend[l]['counts']) for l in range(N_CLASSES)}
    max_count = max(class_total.values()) if any(class_total.values()) else 1

    highlight_classes = [l for l in range(N_CLASSES) if class_total[l] > 0] 
    palette = plt.cm.tab10(np.linspace(0, 1, len(highlight_classes)))

    fig, ax = plt.subplots(figsize=(14, 7))

    for label in range(N_CLASSES):
        trend = class_ce_trend[label]
        r = trend['ratios']
        c = trend['ces']
        total = class_total[label]
        if total == 0:
            continue
        if label in highlight_classes:
            color = palette[list(highlight_classes).index(label)]
            ax.plot(r, c, 'o-', linewidth=2.0, markersize=6, color=color,
                    label=f'class {label} (n={total})')
        else:
            alpha = 0.12 + 0.25 * (total / max_count)
            ax.plot(r, c, '-', linewidth=0.5, color='gray', alpha=min(alpha, 0.6))

    ax.set_xlabel('Reveal Ratio')
    ax.set_ylabel('Average Cross-Entropy')
    ax.set_title('CE Trend Across Budgets — Highlighted: Key Classes')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_zero_traj_heatmap(all_ratio_data: dict, output_path: Path):
    """热力图: 不同预算下哪些类别没有轨迹. 行=类别, 列=budget, 绿=有数据, 红=无数据."""
    ratios = sorted(all_ratio_data.keys())

    has_data = np.ones((N_CLASSES, len(ratios)))

    for j, ratio in enumerate(ratios):
        flat = all_ratio_data[ratio]
        label_set = set()
        for _, traj in flat:
            label_set.add(int(traj.final_pred))
        for i in range(N_CLASSES):
            if i not in label_set:
                has_data[i, j] = 0

    fig, ax = plt.subplots(figsize=(8, max(8, N_CLASSES * 0.32)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(has_data, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f'{r:.3f}' for r in ratios], fontsize=9)
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(range(N_CLASSES), fontsize=7)
    ax.set_xlabel('Reveal Ratio')
    ax.set_ylabel('Class Label')
    ax.set_title('Classes with Trajectories (Green) vs Missing (Red) at Each Budget')

    missing_counts = np.sum(has_data == 0, axis=0)
    for j in range(len(ratios)):
        ax.text(j, N_CLASSES, f'miss\n{int(missing_counts[j])}', ha='center', fontsize=7,
                color='darkred', fontweight='bold', va='top')

    for i in range(N_CLASSES):
        for j in range(len(ratios)):
            if has_data[i, j] == 0:
                ax.text(j, i, '×', ha='center', va='center', fontsize=7, color='darkred',
                        fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _script_dir / 'data' / 'stage3_filtered'
    output_dir = Path(args.output_dir) if args.output_dir else _script_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_data = load_merged_files(input_dir)

    all_stats = {}
    for ratio_str, data in sorted(merged_data.items()):
        logger.info("Analyzing ratio=%s...", ratio_str)
        all_flat = flatten_trajectories(data)
        teacher_names = data['config']['teacher_names']
        stats = compute_stats(all_flat, teacher_names)
        all_stats[ratio_str] = stats

        title_suf = f' (ratio={ratio_str.replace("_", ".")})'
        prefix = f'analysis_r{ratio_str}'

        # 1. 整体 label 分布
        label_counts_raw = {}
        for k in range(N_CLASSES):
            label_counts_raw[k] = stats['label_distribution'].get(str(k), 0)
        plot_label_distribution(label_counts_raw, output_dir / f'{prefix}_label_dist.png',
                                title_suffix=title_suf)

        # 2. 每种 teacher 的 label 分布
        teacher_label_counts = {}
        for tn in teacher_names:
            teacher_label_counts[tn] = {}
            for k in range(N_CLASSES):
                teacher_label_counts[tn][k] = stats['per_teacher'][tn][
                    'label_distribution'].get(str(k), 0)
        plot_teacher_label_distribution(teacher_label_counts,
                                        output_dir / f'{prefix}_teacher_label_dist.png',
                                        title_suffix=title_suf)

        # 3. CE heatmap
        teacher_label_avg_ce = {}
        for tn in teacher_names:
            teacher_label_avg_ce[tn] = {}
            for k in range(N_CLASSES):
                teacher_label_avg_ce[tn][k] = stats['per_teacher'][tn][ 
                    'per_label_avg_ce'].get(str(k), float('NaN'))
        plot_ce_heatmap(teacher_label_avg_ce,
                        output_dir / f'{prefix}_ce_heatmap.png',
                        title_suffix=title_suf)

        logger.info("  Total trajectories: %d", stats['total_trajectories'])
        for tn in teacher_names:
            logger.info("    %s: %d trajectories", tn, stats['per_teacher'][tn]['total'])

    # --- 跨 budget 分析 ---
    logger.info("=== Cross-Budget Analysis ===")
    all_ratio_data = load_all_ratio_files(input_dir)

    plot_per_label_ce_by_budget(all_ratio_data,
                                output_dir / 'analysis_cross_budget_ce.png')
    plot_ce_trend_across_budgets(all_ratio_data,
                                 output_dir / 'analysis_ce_trend.png')
    plot_zero_traj_heatmap(all_ratio_data,
                           output_dir / 'analysis_zero_traj_heatmap.png')

    # 补充 JSON: 跨 budget 统计
    cross_stats = {}
    for ratio, flat in sorted(all_ratio_data.items()):
        missing = sorted([l for l in range(N_CLASSES)
                          if not any(int(t.final_pred) == l for _, t in flat)])
        cross_stats[str(ratio)] = {
            'total_trajectories': len(flat),
            'missing_classes': missing,
            'n_missing': len(missing),
        }
    all_stats['cross_budget'] = cross_stats

    # 保存 JSON
    json_path = output_dir / 'analysis_stats.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info("Saved stats to %s", json_path)


if __name__ == '__main__':
    main()
