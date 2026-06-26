"""M4 类别平衡脚本 — 对过代表类别进行去重 + 上限截断 + 预算比例降采样。

策略:
  1. 对每个 (graph_position, budget_ratio) 组内，对目标类轨迹做余弦相似度去重
  2. 去重后若某类总数 > max_per_class，按各预算占比分配配额，各预算内按 CE 升序保留

用法:
  python code/m4/process-data/balance_classes.py \
      --input-dir code/m4/data/stage3_trimmed \
      --output-dir code/m4/data/stage4_balanced \
      --labels 2 6 17 18 \
      --max-per-class 8000 \
      --sim-threshold 0.85
"""

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent.parent  # process-data/ → m4/ → code/
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from m4.teachers.base import Trajectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='M4 Class Balance Filter')
    p.add_argument('--input-dir', type=str, default=None,
                   help='Input dir with merged .pt files (default: data/stage3_trimmed)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output dir (default: data/stage4_balanced)')
    p.add_argument('--labels', type=int, nargs='+', default=[2, 6, 17, 18],
                   help='Target classes to downsample (default: 2 6 17 18)')
    p.add_argument('--max-per-class', type=int, default=8000,
                   help='Per-class trajectory cap (default: 8000)')
    p.add_argument('--sim-threshold', type=float, default=0.85,
                   help='Cosine similarity threshold for dedup (default: 0.85)')
    return p.parse_args()


def similarity(set1: set, set2: set) -> float:
    """余弦相似度: |A ∩ B| / sqrt(|A| * |B|)"""
    n_intersection = len(set1 & set2)
    if n_intersection == 0:
        return 0.0
    return n_intersection / math.sqrt(len(set1) * len(set2))


def dedup_group(trajs: list, threshold: float) -> list:
    """贪心去重: 按 CE 升序，优先保留低 CE 轨迹。"""
    if len(trajs) <= 1:
        return trajs

    sorted_trajs = sorted(trajs, key=lambda t: t.ce)
    kept = []
    kept_sets = []

    for traj in sorted_trajs:
        traj_set = set(traj.reveal_order)
        is_dup = False
        for ks in kept_sets:
            if similarity(traj_set, ks) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(traj)
            kept_sets.append(traj_set)

    return kept


def main():
    args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _script_dir / 'data' / 'stage3_trimmed'
    output_dir = Path(args.output_dir) if args.output_dir else _script_dir / 'data' / 'stage4_balanced'
    output_dir.mkdir(parents=True, exist_ok=True)

    target_labels = set(args.labels)
    logger.info("Target classes: %s", sorted(target_labels))
    logger.info("Max per class: %d", args.max_per_class)
    logger.info("Sim threshold: %.2f", args.sim_threshold)

    files = sorted(input_dir.glob('trajectories_trimmed_r*.pt'))
    if not files:
        logger.error("No trimmed files found in %s", input_dir)
        sys.exit(1)

    # ---- Phase 1: dedup per (graph_pos, ratio) for target classes ----
    # keep_ids: set of id(traj) that survive dedup
    # Also track per-class per-budget counts for later cap enforcement
    keep_ids: set[int] = set()
    # class_budget_kept[label][ratio_str] = list of traj surviving dedup
    class_budget_kept: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # Store loaded data for Phase 3 rebuild
    all_data: dict[str, dict] = {}

    grand_before = 0
    grand_after_dedup = 0

    for f in files:
        ratio_str = f.stem.replace('trajectories_trimmed_r', '')
        logger.info("Loading ratio=%s for dedup pass...", ratio_str)

        data = torch.load(f, map_location='cpu', weights_only=False)
        all_data[ratio_str] = data
        trajs_dict = data['trajectories']
        teacher_names = list(trajs_dict.keys())
        n_graphs = len(trajs_dict[teacher_names[0]])

        # Maps: (class_label, graph_pos) → list of trajectories
        # We collect per graph position to apply dedup
        for g in range(n_graphs):
            # Collect all trajs at graph position g
            class_groups: dict[int, list] = defaultdict(list)
            for tn in teacher_names:
                for traj in trajs_dict[tn][g]:
                    label = traj.final_pred
                    class_groups[label].append(traj)
                    grand_before += 1

            for label, trajs in class_groups.items():
                if label in target_labels:
                    kept = dedup_group(trajs, args.sim_threshold)
                    class_budget_kept[label][ratio_str].extend(kept)
                    for t in kept:
                        keep_ids.add(id(t))
                        grand_after_dedup += 1
                else:
                    for t in trajs:
                        keep_ids.add(id(t))
                        grand_after_dedup += 1

    dedup_removed = grand_before - grand_after_dedup
    logger.info("Phase 1 (dedup): %d → %d (removed %d, %.1f%%)",
                 grand_before, grand_after_dedup, dedup_removed,
                 dedup_removed / grand_before * 100 if grand_before > 0 else 0)

    # ---- Phase 2: per-class cap with budget-proportional truncation ----
    for label in sorted(target_labels):
        budget_trajs = class_budget_kept[label]
        total = sum(len(ts) for ts in budget_trajs.values())
        if total == 0:
            continue

        logger.info("Class %d after dedup: %d trajectories", label, total)

        if total <= args.max_per_class:
            logger.info("  Under cap (%d), no truncation", args.max_per_class)
            continue

        # Compute proportional quota per budget
        budget_quota: dict[str, int] = {}
        allocated = 0
        sorted_budgets = sorted(budget_trajs.keys())
        for i, rk in enumerate(sorted_budgets):
            count = len(budget_trajs[rk])
            if i == len(sorted_budgets) - 1:
                quota = args.max_per_class - allocated
            else:
                quota = max(1, int(args.max_per_class * count / total))
            budget_quota[rk] = quota
            allocated += quota

        # Truncate each budget: sort by CE, keep top quota
        for rk in sorted_budgets:
            trajs = budget_trajs[rk]
            quota = budget_quota[rk]
            if len(trajs) <= quota:
                continue

            sorted_trajs = sorted(trajs, key=lambda t: t.ce)
            removed = [t for t in sorted_trajs[quota:]]
            for t in removed:
                keep_ids.discard(id(t))

            before = len(trajs)
            logger.info("  budget %s: %d → %d", rk, before, quota)

    # ---- Phase 3: rebuild and save ----
    grand_after = 0

    for ratio_str, data in all_data.items():
        trajs_dict = data['trajectories']
        config = data['config']
        teacher_names = list(trajs_dict.keys())
        n_graphs = len(trajs_dict[teacher_names[0]])

        filtered = {tn: [[] for _ in range(n_graphs)] for tn in teacher_names}
        before = 0
        after = 0

        for tn in teacher_names:
            for g in range(n_graphs):
                for traj in trajs_dict[tn][g]:
                    before += 1
                    if id(traj) in keep_ids:
                        filtered[tn][g].append(traj)
                        after += 1

        removed = before - after
        logger.info("Ratio %s: %d → %d (removed %d, %.1f%%)",
                     ratio_str, before, after, removed,
                     removed / before * 100 if before > 0 else 0)

        grand_after += after

        output_path = output_dir / f'trajectories_balanced_r{ratio_str}.pt'
        torch.save({
            'trajectories': filtered,
            'config': config,
        }, output_path)
        logger.info("  Saved to %s", output_path)

    grand_removed = grand_before - grand_after
    logger.info("=== Grand total: %d → %d (removed %d, %.1f%%) ===",
                 grand_before, grand_after, grand_removed,
                 grand_removed / grand_before * 100 if grand_before > 0 else 0)

    # Per-class summary
    class_counts_after: dict[int, int] = defaultdict(int)
    for ratio_str, data in all_data.items():
        trajs_dict = data['trajectories']
        teacher_names = list(trajs_dict.keys())
        n_graphs = len(trajs_dict[teacher_names[0]])
        for tn in teacher_names:
            for g in range(n_graphs):
                for traj in trajs_dict[tn][g]:
                    if id(traj) in keep_ids:
                        class_counts_after[traj.final_pred] += 1

    for label in sorted(target_labels):
        logger.info("  class %2d: %d", label, class_counts_after.get(label, 0))


if __name__ == '__main__':
    main()
