"""M4 预算裁剪脚本 — 按类别丢弃 0.20 budget 的轨迹。

策略（来自 analysis doc）：
  - 总轨迹数 < 1000 的类别：保留全部 4 个 budget
  - 总轨迹数 >= 1000 的类别（除 12,21,24,4 外）：丢弃 0.20 budget
  - 例外：12, 21, 24, 4 虽 >= 1000 但仍保留 0.20

用法:
  python ./code/m4/trim_budget.py --input-dir ./code/m4/data/stage1_merged --output-dir ./code/m4/data/stage2_trimmed
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

N_CLASSES = 37

# 丢弃 0.20 的类别：总轨迹 >= 1000 且不在例外列表 {12, 21, 24, 4}
DROP_R020_LABELS = {1, 2, 3, 6, 17, 18, 23, 33, 36}


def parse_args():
    p = argparse.ArgumentParser(description='M4 Budget Trimming')
    p.add_argument('--input-dir', type=str, default=None,
                   help='Directory containing trajectories_merged_r*.pt')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory for trimmed files')
    return p.parse_args()


def load_all_merged(input_dir: Path) -> dict:
    """加载所有 merged .pt 文件, 返回 {ratio_float: (data_dict, file_path)}."""
    files = sorted(input_dir.glob('trajectories_merged_r*.pt'))
    if not files:
        logger.error("No merged files found in %s", input_dir)
        sys.exit(1)

    result = {}
    for f in files:
        stem = f.stem
        ratio_str = stem.replace('trajectories_merged_r', '')
        ratio_float = float(ratio_str.replace('_', '.'))
        data = torch.load(f, map_location='cpu', weights_only=False)
        result[ratio_float] = data
        logger.info("Loaded %s: %d graphs, %d trajectories",
                     f.name, data['config']['n_graphs'],
                     sum(sum(len(trajs) for trajs in per_graph)
                         for per_graph in data['trajectories'].values()))
    return result


def count_per_class(data: dict) -> dict:
    """统计一个 data dict 中各类别的轨迹数。"""
    counts = defaultdict(int)
    for per_graph in data['trajectories'].values():
        for graph_trajs in per_graph:
            for traj in graph_trajs:
                counts[int(traj.final_pred)] += 1
    return dict(counts)


def filter_r020(data: dict) -> tuple:
    """从 r0_200 数据中移除丢弃类别的轨迹。

    返回 (filtered_data, removed_count).
    """
    trajs_dict = data['trajectories']
    teacher_names = list(trajs_dict.keys())
    n_graphs = len(trajs_dict[teacher_names[0]]) if teacher_names else 0

    filtered = {tn: [[] for _ in range(n_graphs)] for tn in teacher_names}
    kept_total = 0
    removed_total = 0

    for g in range(n_graphs):
        for tn in teacher_names:
            for traj in trajs_dict[tn][g]:
                if int(traj.final_pred) in DROP_R020_LABELS:
                    removed_total += 1
                else:
                    filtered[tn][g].append(traj)
                    kept_total += 1

    result = {
        'trajectories': filtered,
        'config': data['config'],
    }
    return result, removed_total


def main():
    args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _script_dir / 'data' / 'stage1_merged'
    output_dir = Path(args.output_dir) if args.output_dir else _script_dir / 'data' / 'stage2_trimmed'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_all_merged(input_dir)

    # 统计裁剪前各类别轨迹数（跨所有 budget）
    before_counts = defaultdict(int)
    for ratio, data in all_data.items():
        for label, c in count_per_class(data).items():
            before_counts[label] += c

    # 确定哪些类别需要丢弃 0.20
    keep_r020 = set()
    drop_r020 = set()
    for label in range(N_CLASSES):
        if label in DROP_R020_LABELS:
            drop_r020.add(label)
        else:
            keep_r020.add(label)

    logger.info("Classes keeping 0.20: %s", sorted(keep_r020))
    logger.info("Classes dropping 0.20: %s", sorted(drop_r020))

    grand_before = 0
    grand_after = 0

    for ratio, data in sorted(all_data.items()):
        ratio_str = f"{ratio:.3f}".replace('.', '_')
        before_count = sum(count_per_class(data).values())

        if abs(ratio - 0.20) < 0.001:
            filtered_data, removed = filter_r020(data)
            after_count = before_count - removed
            logger.info("ratio=%.3f: %d → %d (removed %d)",
                         ratio, before_count, after_count, removed)
        else:
            filtered_data = data
            after_count = before_count
            logger.info("ratio=%.3f: %d (unchanged)", ratio, before_count)

        grand_before += before_count
        grand_after += after_count

        out_path = output_dir / f'trajectories_merged_r{ratio_str}.pt'
        torch.save(filtered_data, out_path)
        logger.info("  Saved %s", out_path)

    logger.info("=== Total: %d → %d (removed %d, %.1f%%) ===",
                 grand_before, grand_after,
                 grand_before - grand_after,
                 (grand_before - grand_after) / grand_before * 100 if grand_before > 0 else 0)

    # 裁剪后各类别轨迹数
    after_counts = defaultdict(int)
    for ratio, data in sorted(all_data.items()):
        ratio_str = f"{ratio:.3f}".replace('.', '_')
        out_path = output_dir / f'trajectories_merged_r{ratio_str}.pt'
        reloaded = torch.load(out_path, map_location='cpu', weights_only=False)
        for label, c in count_per_class(reloaded).items():
            after_counts[label] += c

    logger.info("=== Per-Class Before → After ===")
    for label in range(N_CLASSES):
        b = before_counts.get(label, 0)
        a = after_counts.get(label, 0)
        if b != a:
            logger.info("  class %2d: %5d → %5d  (removed %d)", label, b, a, b - a)
        elif b > 0:
            logger.info("  class %2d: %5d (unchanged)", label, b)


if __name__ == '__main__':
    main()
