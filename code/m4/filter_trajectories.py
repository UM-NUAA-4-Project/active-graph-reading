"""M4 轨迹数据筛选脚本 — 向量归一化去重。

将轨迹视为 binary vector (节点揭示与否) → L2 归一化 → 余弦相似度去重。
同一 (graph_idx, reveal_ratio) 下，高度相似的轨迹只保留 CE 最低的。

用法:
  python code/m4/filter_trajectories.py [--input-dir m4/output] [--output-dir m4/output]
                                        [--threshold 0.85]
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import List, Set, Tuple

import torch

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from m4.teachers.base import Trajectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='M4 Trajectory Filter (vector-based dedup)')
    p.add_argument('--input-dir', type=str, default=None,
                   help='Directory containing merged .pt files (default: m4/output)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory for filtered files (default: same as input-dir)')
    p.add_argument('--threshold', type=float, default=0.85,
                   help='Cosine similarity threshold for dedup (default: 0.85)')
    return p.parse_args()


def cosine_similarity(set1: Set[int], set2: Set[int]) -> float:
    """计算两个轨迹集表示的余弦相似度。

    cos_sim = |A ∩ B| / sqrt(|A| * |B|)
    """
    n_intersection = len(set1 & set2)
    if n_intersection == 0:
        return 0.0
    return n_intersection / math.sqrt(len(set1) * len(set2))


def dedup_group(trajs: List[Trajectory], threshold: float) -> List[Trajectory]:
    """组内贪心去重: 按 CE 升序, 优先保留低 CE 轨迹。

    trajs: 同一 (graph_idx, reveal_ratio) 的所有轨迹
    返回保留的轨迹列表。
    """
    if len(trajs) <= 1:
        return trajs

    # 按 CE 升序排列
    sorted_trajs = sorted(trajs, key=lambda t: t.ce)

    kept: List[Trajectory] = []
    kept_sets: List[Set[int]] = []

    for traj in sorted_trajs:
        traj_set = set(traj.reveal_order)
        is_dup = False
        for ks in kept_sets:
            if cosine_similarity(traj_set, ks) > threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(traj)
            kept_sets.append(traj_set)

    return kept


def filter_trajectories(trajs_dict: dict, threshold: float) -> Tuple[dict, int, int]:
    """对嵌套 trajectories dict 进行去重。

    trajs_dict: {teacher_name: [[Trajectory for graph_i], ...]}
    返回 (filtered_dict, before_count, after_count).
    """
    teacher_names = list(trajs_dict.keys())
    n_graphs = len(trajs_dict[teacher_names[0]])

    # 初始化 filtered dict: 空列表占位
    filtered = {tn: [[] for _ in range(n_graphs)] for tn in teacher_names}

    before_total = 0
    after_total = 0

    for g in range(n_graphs):
        # 收集该图该 ratio 下所有 teacher 的轨迹
        group: List[Tuple[str, Trajectory]] = []
        for tn in teacher_names:
            for traj in trajs_dict[tn][g]:
                group.append((tn, traj))
                before_total += 1

        if not group:
            continue

        # 去重
        all_trajs = [t for _, t in group]
        kept_trajs = dedup_group(all_trajs, threshold)

        # 按 teacher 归类放回
        kept_set = set(id(t) for t in kept_trajs)
        for tn, traj in group:
            if id(traj) in kept_set:
                filtered[tn][g].append(traj)
                after_total += 1

    return filtered, before_total, after_total


def main():
    args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _script_dir / 'output'
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('trajectories_merged_r*.pt'))
    if not files:
        logger.error("No merged files found in %s", input_dir)
        sys.exit(1)

    logger.info("Threshold: %.2f", args.threshold)
    grand_before = 0
    grand_after = 0

    for f in files:
        ratio_str = f.stem.replace('trajectories_merged_r', '')
        logger.info("Processing ratio=%s...", ratio_str)

        data = torch.load(f, map_location='cpu', weights_only=False)
        config = data['config']
        logger.info("  %d graphs, %d teachers", config['n_graphs'],
                     len(config['teacher_names']))

        filtered_trajs, before, after = filter_trajectories(
            data['trajectories'], args.threshold)

        removed = before - after
        rate = removed / before * 100 if before > 0 else 0
        logger.info("  Before: %d  After: %d  Removed: %d (%.1f%%)",
                     before, after, removed, rate)
        grand_before += before
        grand_after += after

        # 统计各 teacher 剩余轨迹
        for tn in config['teacher_names']:
            t_count = sum(len(trajs) for trajs in filtered_trajs[tn])
            logger.info("    %25s: %d trajectories", tn, t_count)

        output_path = output_dir / f'trajectories_filtered_r{ratio_str}.pt'
        torch.save({
            'trajectories': filtered_trajs,
            'config': config,
        }, output_path)
        logger.info("  Saved to %s", output_path)

    grand_removed = grand_before - grand_after
    grand_rate = grand_removed / grand_before * 100 if grand_before > 0 else 0
    logger.info("=== Grand total: %d → %d (removed %d, %.1f%%) ===",
                 grand_before, grand_after, grand_removed, grand_rate)


if __name__ == '__main__':
    main()
