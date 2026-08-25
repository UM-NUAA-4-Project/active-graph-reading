"""M7 轨迹过滤：相似去重 → 每图 top-k + 教师多样性（供 train_bc.py 使用）。

流程（两阶段，与 m4/m5 生产管线对齐）：
  Stage 1 相似去重（迁移自 m4/process-data/balance_classes.py）：
    同图内轨迹按 CE 升序贪心去重——对每个候选取 set(reveal_order) 与已保留
    集合做余弦相似度 |A∩B|/sqrt(|A|·|B|)，> --sim-threshold 判为重复丢弃，
    低 CE（高质量，含教师优先）先保留。
  Stage 2 每图 top-k（迁移自 m4/trajectory_filter.py，M5 BC 前的质量过滤）：
    去重后的轨迹再按 CE 升序取每图 top-k，且尽量保证覆盖各教师类型各 1 条。

正确性：生成器内部已只保留 is_correct=True（m4 generate_all），此处不再校验；
质量分 = -CE——m4 原 TrajectoryScorer 的 BE/IG/Stability 合成依赖旧全量版
Trajectory 的 entropy_per_step/budget_used（现精简版只有 ce），生产管线
（balance_classes.py / filter_stage7.py）排序键同为 CE 升序，此处保持一致。

输出格式与 train_bc.py 的 merge_trajectories 完全一致
（{'trajectories': {teacher: [[traj,...],...]}, 'config': {...}}），
训练时把 train_bc.py 的 --data-dir 指向输出目录即可。

用法:
  python code/m7/filter_trajectories.py
  python code/m7/filter_trajectories.py --sim-threshold 0.90 --top-k 5 --no-diversity
  python code/m7/train_bc.py ... --data-dir m7/data/trajectories_filtered
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
sys.path.insert(0, str(_repo_root))
# 移除 code/m7 自身：dgl→torchdata 的 `import datasets` 不得解析到 code/m7/datasets
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import torch
from tqdm import tqdm

from m4.teachers.base import Trajectory


def ratio_key_from_name(fname: str) -> str:
    """'trajectories_r0_050.pt' -> '0_050'。"""
    return Path(fname).stem.rsplit('_r', 1)[1]


def similarity(set1: Set[int], set2: Set[int]) -> float:
    """余弦相似度: |A ∩ B| / sqrt(|A| * |B|)（同 balance_classes.py）。"""
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
    """m4 TrajectoryFilter._select_top_k 的移植。

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
    data = torch.load(str(fpath), map_location='cpu', weights_only=False)
    orig_config = data.get('config', {})
    teacher_names = teachers or orig_config.get('teachers', [])

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


def main():
    parser = argparse.ArgumentParser(
        description='M7 轨迹过滤（相似去重 → 每图 top-k + 教师多样性）')
    parser.add_argument('--input-dir', default='m7/data/trajectories',
                        help='输入目录（generate_trajectories.py 输出，相对 code/ 或绝对路径）')
    parser.add_argument('--output-dir', default='m7/data/trajectories_filtered',
                        help='输出目录（相对 code/ 或绝对路径）')
    parser.add_argument('--sim-threshold', type=float, default=0.85,
                        help='同图去重的余弦相似度阈值（balance_classes.py 默认 0.85）')
    parser.add_argument('--top-k', type=int, default=3,
                        help='去重后每图保留最大轨迹数（m4 默认 3）')
    parser.add_argument('--no-diversity', action='store_true',
                        help='关闭教师类型多样性保证（默认开启）')
    parser.add_argument('--teachers', nargs='+', default=None,
                        help='教师名列表（默认取各文件 config.teachers）')
    args = parser.parse_args()

    input_dir = _repo_root / args.input_dir
    output_dir = _repo_root / args.output_dir
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}（先运行 m7/generate_trajectories.py）")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('trajectories_r*.pt'))
    if not files:
        raise SystemExit(f"输入目录无 trajectories_r*.pt：{input_dir}")
    require_diversity = not args.no_diversity

    report = {}
    for fpath in tqdm(files, desc="Filtering"):
        key = ratio_key_from_name(fpath.name)
        out = filter_file(fpath, args.teachers, args.top_k, require_diversity,
                          args.sim_threshold)
        report[key] = out['report']
        save_path = output_dir / fpath.name  # 文件名与 train_bc.py 期望的 trajectories_r{ratio}.pt 一致
        torch.save(out['data'], str(save_path))
        r = out['report']
        print(f"  {fpath.name}: {r['n_before']} -> dedup {r['n_after_dedup']} "
              f"(removed {r['dedup_removed']}) -> top-k {r['n_after']} "
              f"(graphs {r['n_graphs_after']}, diverse {r['diverse_graphs']}, "
              f"single {r['single_teacher_graphs']})")
        print(f"  Saved: {save_path}")

    report_path = output_dir / 'filter_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")
    print(f"下一步: python code/m7/train_bc.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    main()
