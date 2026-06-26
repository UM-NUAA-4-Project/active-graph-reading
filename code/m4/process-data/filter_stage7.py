"""Stage7: 每 (class, budget) ≤ 2000 条，超出按 CE 升序截断。

用法:
  python code/m4/process-data/filter_stage7.py
    --input-dir code/m4/data/stage6_merged
    --output-dir code/m4/data/stage7_filtered
    --max-per-class-budget 2000
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent.parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

BUDGET_KEYS = ["0_050", "0_100", "0_150", "0_200"]


def parse_args():
    p = argparse.ArgumentParser(description='Stage7: Per-class-budget cap')
    p.add_argument('--input-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'stage6_merged'))
    p.add_argument('--output-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'stage7_filtered'))
    p.add_argument('--max-per-class-budget', type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_n = args.max_per_class_budget
    print(f"Max per class-budget: {max_n}")

    grand_before = 0
    grand_after = 0
    class_budget_before = defaultdict(int)
    class_budget_after = defaultdict(int)

    for bk in BUDGET_KEYS:
        fpath = input_dir / f'trajectories_merged_r{bk}.pt'
        if not fpath.exists():
            print(f"  {bk}: file not found, skipping")
            continue

        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        trajs_dict = data['trajectories']
        config = data['config']
        teacher_names = config['teacher_names']
        n_graphs = config['n_graphs']

        # Collect all trajectories with their graph position and teacher
        # so we can rebuild after pruning
        all_entries = []  # (class, teacher_name, graph_pos, traj)
        before_total = 0
        for tn in teacher_names:
            for g in range(n_graphs):
                for traj in trajs_dict[tn][g]:
                    all_entries.append((traj.final_pred, tn, g, traj))
                    before_total += 1
                    class_budget_before[(traj.final_pred, bk)] += 1

        # Group by class, sort by CE, keep top max_n
        class_groups = defaultdict(list)
        for lbl, tn, g, traj in all_entries:
            class_groups[lbl].append((tn, g, traj))

        keep_ids = set()
        for lbl, entries in class_groups.items():
            # 按 CE 升序排列，优先保留低 CE
            sorted_entries = sorted(entries, key=lambda x: x[2].ce)
            kept = sorted_entries[:max_n]
            for _tn, _g, traj in kept:
                keep_ids.add(id(traj))
            class_budget_after[(lbl, bk)] += len(kept)

        # Rebuild
        filtered = {tn: [[] for _ in range(n_graphs)] for tn in teacher_names}
        after_total = 0
        for lbl, tn, g, traj in all_entries:
            if id(traj) in keep_ids:
                filtered[tn][g].append(traj)
                after_total += 1

        removed = before_total - after_total
        print(f"  {bk}: {before_total} → {after_total} (removed {removed})")
        grand_before += before_total
        grand_after += after_total

        out_path = output_dir / f'trajectories_filtered_r{bk}.pt'
        torch.save({
            'trajectories': filtered,
            'config': {**config, 'stage7_cap': max_n},
        }, str(out_path))
        print(f"    Saved {out_path}")

    print(f"\nTotal: {grand_before} → {grand_after} (removed {grand_before - grand_after})")

    # Per-class summary of capped classes
    print("\nClasses capped (top-2000 per budget):")
    for (lbl, bk), before in sorted(class_budget_before.items(), key=lambda x: x[1], reverse=True):
        after = class_budget_after.get((lbl, bk), 0)
        if before > max_n:
            print(f"  class {lbl} @ {bk}: {before} → {after}")


if __name__ == '__main__':
    main()
