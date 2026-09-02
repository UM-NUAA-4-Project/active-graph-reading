"""filtering：轨迹相似去重 + 每图 top-k + 教师多样性（自 m7/filter_trajectories.py 收编）。

流程（两阶段，与 m4/m5 生产管线对齐）：
  Stage 1 相似去重：同图内轨迹按 CE 升序贪心去重（余弦相似度
    |A∩B|/sqrt(|A|·|B|) > threshold 判为重复）。
  Stage 2 每图 top-k：去重后按 CE 升序取每图 top-k，尽量覆盖各教师各 1 条。
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tools.trajectory_io import load_trajectory_file
from .trajectory import Trajectory


def similarity(set1: Set[int], set2: Set[int]) -> float:
    """余弦相似度: |A ∩ B| / sqrt(|A| * |B|)。"""
    n_inter = len(set1 & set2)
    if n_inter == 0:
        return 0.0
    return n_inter / math.sqrt(len(set1) * len(set2))


def dedup_group(records: List[tuple], threshold: float) -> List[tuple]:
    """贪心去重：按 CE 升序（质量降序）遍历，与已保留集合相似度 > threshold 则丢弃。

    records: (quality, teacher, traj)，quality = -ce。返回值保持质量降序。
    """
    if len(records) <= 1:
        return records

    records = sorted(records, key=lambda r: r[0], reverse=True)
    kept = []
    kept_sets = []
    for rec in records:
        traj_set = set(rec[2].reveal_order)
        if any(similarity(traj_set, ks) > threshold for ks in kept_sets):
            continue
        kept.append(rec)
        kept_sets.append(traj_set)
    return kept


def select_top_k(records: List[tuple], top_k: int, require_diversity: bool) -> List[Trajectory]:
    """每图 top-k 选择（m4 TrajectoryFilter._select_top_k 移植）。

    records: (quality, teacher, traj)，quality 降序。
    策略：恒取最优 1 条 → 尽量纳入每个教师类型各 1 条 → 其余按质量补齐。
    """
    if len(records) <= top_k:
        return [r[2] for r in records]

    selected = []
    used_teachers: Set[str] = set()
    remaining = list(records)

    selected.append(remaining.pop(0))
    used_teachers.add(selected[0][1])

    if require_diversity:
        all_types = sorted({r[1] for r in remaining})
        for ttype in all_types:
            if len(selected) >= top_k:
                break
            if ttype in used_teachers:
                continue
            for i, r in enumerate(remaining):
                if r[1] == ttype:
                    selected.append(remaining.pop(i))
                    used_teachers.add(ttype)
                    break

    while len(selected) < top_k and remaining:
        selected.append(remaining.pop(0))

    return [r[2] for r in selected]


def filter_file(
    fpath: Path, teachers: List[str], top_k: int, require_diversity: bool,
    sim_threshold: float,
) -> Dict:
    """过滤单个 trajectory 文件，返回 {'data': 保存字典, 'report': 统计}。"""
    data = load_trajectory_file(fpath)
    orig_config = data.get('config', {})
    teacher_names = teachers or list(orig_config.get('teachers', [])
                                     or data['trajectories'].keys())

    records: List[tuple] = []  # (quality, teacher, traj)
    for tname in teacher_names:
        for per_graph in data['trajectories'].get(tname, []):
            for traj in per_graph:
                records.append((-traj.ce, tname, traj))

    n_before = len(records)

    # Stage 1: 按图分组 → 相似去重；Stage 2: 去重后每图 top-k + 多样性
    by_graph: Dict[int, List[tuple]] = defaultdict(list)
    for rec in records:
        by_graph[rec[2].graph_idx].append(rec)

    max_g_idx = max(by_graph, default=-1)
    result = {tname: [[] for _ in range(max_g_idx + 1)] for tname in teacher_names}
    n_after_dedup = 0
    diverse = single = 0
    for g_idx, recs in by_graph.items():
        deduped = dedup_group(recs, sim_threshold)
        n_after_dedup += len(deduped)
        kept = select_top_k(deduped, top_k, require_diversity)
        if len({t.teacher_type for t in kept}) > 1:
            diverse += 1
        else:
            single += 1
        for traj in kept:
            result[traj.teacher_type][g_idx].append(traj)

    report = {
        'n_before': n_before,
        'n_after_dedup': n_after_dedup,
        'dedup_removed': n_before - n_after_dedup,
        'n_after': sum(len(l) for lists in result.values() for l in lists),
        'n_graphs_after': len(by_graph),
        'diverse_graphs': diverse,
        'single_teacher_graphs': single,
    }
    save = {
        'trajectories': result,
        'config': {
            **orig_config,
            'top_k': top_k,
            'require_diversity': require_diversity,
            'sim_threshold': sim_threshold,
            'filter_report': report,
        },
    }
    return {'data': save, 'report': report}
