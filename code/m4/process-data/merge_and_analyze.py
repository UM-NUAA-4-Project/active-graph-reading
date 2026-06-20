"""Stage6: Merge teacher (stage5) and reader trajectories, dedup, cap, analyze.

Usage:
  python code/m4/process-data/merge_and_analyze.py
    --teacher-dir code/m4/data/stage5_filtered
    --reader-dir code/m4/data/reader_trajectories
    --output-dir code/m4/data/stage6_merged
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent.parent  # process-data/ -> m4/ -> code/
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

from m4.teachers.base import Trajectory

# ---- Constants ----
REGULAR_CLASSES = {1, 2, 3, 4, 6, 12, 17, 18, 21, 23, 24, 33, 36}
SCARCE_CLASSES = {0, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 22,
                  25, 26, 27, 28, 29, 30, 31, 32, 34, 35}
N_CLASSES = 37
SIM_THRESHOLD = 0.85
MAX_PER_GRAPH_BUDGET = 10  # for regular classes only
TARGET_PER_CLASS_BUDGET = 200

# Budget canonical keys and filename mappings
BUDGET_KEYS = ["0_050", "0_100", "0_150", "0_200"]
BUDGET_VALUES = [0.05, 0.10, 0.15, 0.20]
BUDGET_TO_KEY = dict(zip(BUDGET_VALUES, BUDGET_KEYS))
KEY_TO_BUDGET = dict(zip(BUDGET_KEYS, BUDGET_VALUES))

# Round reveal_ratio to canonical budget (snap to nearest)
BUDGET_SNAP = {0.05: 0.05, 0.10: 0.10, 0.15: 0.15, 0.20: 0.20}


def snap_budget(ratio: float) -> float:
    """Snap a reveal_ratio to the nearest canonical budget value."""
    val = round(ratio, 2)
    if val in BUDGET_SNAP:
        return BUDGET_SNAP[val]
    # fallback: nearest
    budgets = sorted(BUDGET_SNAP.keys())
    return min(budgets, key=lambda b: abs(b - val))


def load_teacher_data(teacher_dir: Path) -> Dict[str, dict]:
    """Load all trajectories_balanced_r*.pt files from stage5 directory.
    Returns {budget_key: raw_data}.
    """
    result = {}
    for bk in BUDGET_KEYS:
        fpath = teacher_dir / f'trajectories_balanced_r{bk}.pt'
        if fpath.exists():
            data = torch.load(str(fpath), map_location='cpu', weights_only=False)
            result[bk] = data
            total = sum(
                sum(len(tl) for tl in per_graph)
                for per_graph in data['trajectories'].values()
            )
            print(f"  Teacher {bk}: {total} trajectories, "
                  f"{data['config']['n_graphs']} graph slots, "
                  f"{len(data['config']['teacher_names'])} teachers")
        else:
            print(f"  Teacher {bk}: file not found, skipping")
            result[bk] = None
    return result


def load_reader_data(reader_dir: Path) -> List[Trajectory]:
    """Load all reader_trajectories-*.pt files. Returns flat list of all trajectories."""
    all_trajs = []
    files = sorted(reader_dir.glob('reader_trajectories-*.pt'))
    if not files:
        print("  Reader: no files found")
        return all_trajs

    for fpath in files:
        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        trajs_wrapper = data['trajectories']  # List[List[Trajectory]]
        for graph_entry in trajs_wrapper:
            for traj in graph_entry:
                all_trajs.append(traj)

    print(f"  Reader: {len(all_trajs)} total trajectories from {len(files)} file(s)")
    return all_trajs


def flatten_and_group(
    teacher_data: Dict[str, dict],
    reader_trajs: List[Trajectory],
) -> Dict[Tuple[int, str], List[Tuple[str, Trajectory]]]:
    """Flatten teacher + reader into per-(graph_idx, budget_key) groups.

    Returns: {(graph_idx, budget_key): [(source, traj), ...]}
    source is "teacher" or "reader".
    """
    groups: Dict[Tuple[int, str], List[Tuple[str, Trajectory]]] = defaultdict(list)

    # Teacher data: nested by teacher → graph_slot → trajectory list
    for bk, data in teacher_data.items():
        if data is None:
            continue
        trajs_dict = data['trajectories']  # {teacher_name: [[Traj], ...]}
        for teacher_name, per_graph in trajs_dict.items():
            for graph_trajs in per_graph:
                for traj in graph_trajs:
                    key = (traj.graph_idx, bk)
                    groups[key].append(("teacher", traj))

    # Reader data: flat list, snap budget
    for traj in reader_trajs:
        budget_val = snap_budget(traj.reveal_ratio)
        bk = BUDGET_TO_KEY.get(budget_val)
        if bk is None:
            continue  # skip if not a canonical budget
        key = (traj.graph_idx, bk)
        groups[key].append(("reader", traj))

    # Sort each group: teacher first, then by CE ascending
    for key in groups:
        groups[key].sort(key=lambda x: (0 if x[0] == "teacher" else 1, x[1].ce))

    total_trajs = sum(len(v) for v in groups.values())
    n_graphs = len(set(k[0] for k in groups))
    print(f"  Flattened: {total_trajs} trajectories across {n_graphs} unique graphs, "
          f"{len(groups)} (graph, budget) groups")
    return groups


def process_group(
    trajs: List[Tuple[str, Trajectory]],
    is_regular: bool,
) -> List[Trajectory]:
    """Dedup within a (graph, budget) group. If regular class, cap at MAX_PER_GRAPH_BUDGET.

    trajs is already sorted: teacher-first, then CE ascending.
    """
    if len(trajs) <= 1:
        return [t for _, t in trajs]

    kept: List[Trajectory] = []
    kept_sets: List[Set[int]] = []

    for _source, traj in trajs:
        traj_set = set(traj.reveal_order)
        is_dup = False
        for ks in kept_sets:
            if similarity(traj_set, ks) > SIM_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(traj)
            kept_sets.append(traj_set)
            if is_regular and len(kept) >= MAX_PER_GRAPH_BUDGET:
                break

    return kept


def classify_group(trajs: List[Tuple[str, Trajectory]]) -> bool:
    """Determine if a group is 'regular' (capped) or 'scarce' (no cap).

    A group is regular if ALL trajectories belong to regular classes.
    If any trajectory is from a scarce class, the whole group is scarce (uncapped).
    """
    for _source, traj in trajs:
        if traj.final_pred in SCARCE_CLASSES:
            return False  # scarce — no cap
    return True  # regular — apply cap


def process_all_groups(
    groups: Dict[Tuple[int, str], List[Tuple[str, Trajectory]]],
) -> Dict[Tuple[int, str], List[Trajectory]]:
    """Dedup and cap all groups. Return {key: [kept trajectories]}."""
    result = {}
    stats_before = 0
    stats_after = 0

    for key, trajs in groups.items():
        is_regular = classify_group(trajs)
        stats_before += len(trajs)
        kept = process_group(trajs, is_regular)
        stats_after += len(kept)
        if kept:
            result[key] = kept

    removed = stats_before - stats_after
    print(f"  Dedup/cap: {stats_before} -> {stats_after} "
          f"(removed {removed}, {removed/stats_before*100:.1f}%)")
    print(f"  Output groups: {len(result)}")
    return result


def rebuild_and_save(
    processed: Dict[Tuple[int, str], List[Trajectory]],
    output_dir: Path,
) -> Dict[str, dict]:
    """Rebuild nested teacher-format dicts and save per-budget .pt files.

    Returns the per-budget dicts for subsequent analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all graph indices per budget
    budget_graphs: Dict[str, Set[int]] = defaultdict(set)
    for (gid, bk), trajs in processed.items():
        budget_graphs[bk].add(gid)

    # Build sorted graph index -> slot position mapping per budget
    budget_slots: Dict[str, Dict[int, int]] = {}
    for bk in BUDGET_KEYS:
        sorted_gids = sorted(budget_graphs.get(bk, set()))
        budget_slots[bk] = {gid: pos for pos, gid in enumerate(sorted_gids)}

    # Determine teacher names: original 5 teachers + "v0-reader"
    all_teacher_names = ['gradient', 'structural-degree', 'structural-pagerank',
                         'hybrid-a0.30', 'hybrid-a0.70', 'v0-reader']

    result = {}
    for bk in BUDGET_KEYS:
        slots = budget_slots[bk]
        n_graphs = len(slots)
        if n_graphs == 0:
            print(f"  {bk}: no trajectories, skipping")
            result[bk] = None
            continue

        # Initialize empty slots per teacher
        trajs_dict: Dict[str, List[List[Trajectory]]] = {
            tn: [[] for _ in range(n_graphs)] for tn in all_teacher_names
        }

        for (gid, key_bk), traj_list in processed.items():
            if key_bk != bk:
                continue
            pos = slots[gid]
            for traj in traj_list:
                teacher_name = traj.teacher_type if traj.teacher_type in all_teacher_names else 'v0-reader'
                if teacher_name not in trajs_dict:
                    trajs_dict[teacher_name] = [[] for _ in range(n_graphs)]
                trajs_dict[teacher_name][pos].append(traj)

        total = sum(
            sum(len(tl) for tl in per_graph)
            for per_graph in trajs_dict.values()
        )
        print(f"  {bk}: {n_graphs} graphs, {total} trajectories")

        config = {
            'reveal_ratio': KEY_TO_BUDGET[bk],
            'n_graphs': n_graphs,
            'teacher_names': [tn for tn in all_teacher_names
                              if any(len(tl) > 0 for tl in trajs_dict[tn])],
        }

        out_path = output_dir / f'trajectories_merged_r{bk}.pt'
        torch.save({'trajectories': trajs_dict, 'config': config}, str(out_path))
        print(f"    Saved {out_path}")
        result[bk] = {'trajectories': trajs_dict, 'config': config}

    return result


def similarity(set1: Set[int], set2: Set[int]) -> float:
    """Cosine similarity: |A ∩ B| / sqrt(|A| * |B|)"""
    n_intersection = len(set1 & set2)
    if n_intersection == 0:
        return 0.0
    return n_intersection / math.sqrt(len(set1) * len(set2))


def analyze_and_report(
    processed: Dict[Tuple[int, str], List[Trajectory]],
    rebuilt: Dict[str, dict],
    output_dir: Path,
):
    """Compute per-(class, budget) statistics and write merge_report.json."""
    report = {
        'config': {
            'regular_classes': sorted(REGULAR_CLASSES),
            'scarce_classes': sorted(SCARCE_CLASSES),
            'sim_threshold': SIM_THRESHOLD,
            'max_per_graph_budget': MAX_PER_GRAPH_BUDGET,
            'target_per_class_budget': TARGET_PER_CLASS_BUDGET,
        },
        'summary': {},
        'per_budget': {},
        'per_class': {},
        'gaps': [],
    }

    # Per-budget per-class stats
    budget_class_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    budget_class_graphs: Dict[str, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    budget_class_source: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))

    # Per-class CE tracking
    budget_class_ce_sum: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    budget_class_ce_cnt: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for (gid, bk), trajs in processed.items():
        for traj in trajs:
            c = traj.final_pred
            budget_class_counts[bk][c] += 1
            budget_class_graphs[bk][c].add(gid)
            src = 'teacher' if traj.teacher_type != 'v0-reader' else 'reader'
            budget_class_source[bk][c][src] += 1
            budget_class_ce_sum[bk][c] += traj.ce
            budget_class_ce_cnt[bk][c] += 1

    # Build per-budget report
    for bk in BUDGET_KEYS:
        counts = budget_class_counts.get(bk, {})
        total = sum(counts.values())
        report['per_budget'][bk] = {
            'total_trajectories': total,
            'n_classes_represented': len([c for c in range(N_CLASSES) if counts.get(c, 0) > 0]),
            'label_distribution': {str(c): counts.get(c, 0) for c in range(N_CLASSES)},
        }

    # Build per-class report
    for c in sorted(REGULAR_CLASSES | SCARCE_CLASSES):
        class_info = {'type': 'regular' if c in REGULAR_CLASSES else 'scarce', 'budgets': {}}
        total_for_class = 0
        for bk in BUDGET_KEYS:
            n = budget_class_counts.get(bk, {}).get(c, 0)
            n_graphs = len(budget_class_graphs.get(bk, {}).get(c, set()))
            ce_sum = budget_class_ce_sum.get(bk, {}).get(c, 0.0)
            ce_cnt = budget_class_ce_cnt.get(bk, {}).get(c, 0)
            avg_ce = ce_sum / ce_cnt if ce_cnt > 0 else None
            sources = budget_class_source.get(bk, {}).get(c, {})
            gap = max(0, TARGET_PER_CLASS_BUDGET - n) if c in REGULAR_CLASSES else None

            class_info['budgets'][bk] = {
                'n_trajectories': n,
                'n_graphs_used': n_graphs,
                'n_teacher': sources.get('teacher', 0),
                'n_reader': sources.get('reader', 0),
                'avg_ce': avg_ce,
                'gap_to_200': gap,
            }
            total_for_class += n

        class_info['total_across_budgets'] = total_for_class
        report['per_class'][str(c)] = class_info

        # Gap tracking
        if c in REGULAR_CLASSES:
            for bk in BUDGET_KEYS:
                n = budget_class_counts.get(bk, {}).get(c, 0)
                gap = TARGET_PER_CLASS_BUDGET - n
                if gap > 0:
                    report['gaps'].append({
                        'class': c, 'budget': bk,
                        'current': n, 'target': TARGET_PER_CLASS_BUDGET, 'gap': gap,
                    })

    # Summary
    total_all = sum(
        sum(dc.values()) for dc in budget_class_counts.values()
    )
    missing_per_budget = {}
    for bk in BUDGET_KEYS:
        missing = [c for c in range(N_CLASSES) if budget_class_counts.get(bk, {}).get(c, 0) == 0]
        missing_per_budget[bk] = {'missing_classes': missing, 'n_missing': len(missing)}

    report['summary'] = {
        'total_trajectories': total_all,
        'n_gaps': len(report['gaps']),
        'missing_per_budget': missing_per_budget,
    }

    # Write JSON
    json_path = output_dir / 'merge_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {json_path}")

    # Print summary
    print(f"\n=== Merge Summary ===")
    print(f"Total trajectories: {total_all}")
    print(f"Regular-class gaps remaining: {len(report['gaps'])}")
    for bk in BUDGET_KEYS:
        mi = missing_per_budget.get(bk, {})
        print(f"  {bk}: missing classes={mi.get('n_missing', '?')} {mi.get('missing_classes', [])}")
    if report['gaps']:
        print("Gaps (regular classes below 200):")
        for g in report['gaps']:
            print(f"  class {g['class']} @ {g['budget']}: {g['current']}/{g['target']} (gap={g['gap']})")

    return report


def parse_args():
    p = argparse.ArgumentParser(description='Stage6: Merge & Analyze Trajectories')
    p.add_argument('--teacher-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'stage5_filtered'))
    p.add_argument('--reader-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'reader_trajectories'))
    p.add_argument('--output-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'stage6_merged'))
    p.add_argument('--sim-threshold', type=float, default=0.85)
    p.add_argument('--max-per-graph', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    teacher_dir = Path(args.teacher_dir)
    reader_dir = Path(args.reader_dir)
    output_dir = Path(args.output_dir)

    global SIM_THRESHOLD, MAX_PER_GRAPH_BUDGET
    SIM_THRESHOLD = args.sim_threshold
    MAX_PER_GRAPH_BUDGET = args.max_per_graph

    print("=== Stage6: Merge & Analyze ===")
    print(f"Teacher dir: {teacher_dir}")
    print(f"Reader dir:  {reader_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Sim threshold: {SIM_THRESHOLD}")
    print(f"Max per graph/budget: {MAX_PER_GRAPH_BUDGET}")

    # 1. Load
    print("\n--- Loading teacher data ---")
    teacher_data = load_teacher_data(teacher_dir)
    print("\n--- Loading reader data ---")
    reader_trajs = load_reader_data(reader_dir)

    # 2. Flatten + group
    print("\n--- Flattening and grouping ---")
    groups = flatten_and_group(teacher_data, reader_trajs)

    # 3. Dedup + cap
    print("\n--- Dedup and cap ---")
    processed = process_all_groups(groups)

    # 4. Rebuild + save
    print("\n--- Rebuilding output format ---")
    rebuilt = rebuild_and_save(processed, output_dir)

    # 5. Analyze + report
    print("\n--- Analysis ---")
    report = analyze_and_report(processed, rebuilt, output_dir)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
