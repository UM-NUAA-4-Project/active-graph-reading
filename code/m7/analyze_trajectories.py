"""统计轨迹文件夹的标签分布（teacher_type + 图 label），保存 JSON 并可视化。

用法:
  python code/m7/analyze_trajectories.py [--data-dir m7/data/trajectories_filtered]
输出（与 --data-dir 同目录）:
  label_distribution.json — 每 ratio/teacher/label 的轨迹数与图数
  trajectory_teacher_dist.png — teacher 分布（含 per-ratio 分组）
  trajectory_label_dist.png — 图 label 分布（整体 + 分 teacher）
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
_project_root = _repo_root.parent                   # 仓库根
sys.path.insert(0, str(_repo_root))
# 移除 code/m7 自身：dgl→torchdata 的 `import datasets` 不得解析到 code/m7/datasets
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from m4.teachers.base import Trajectory


def load_trajs(data_dir: Path) -> dict:
    """-> {'0_050': {'teacher': [Trajectory...]}, ...} 过滤掉 -1 label 的轨迹。"""
    merged = {}
    for fpath in sorted(data_dir.glob('trajectories_r*.pt')):
        ratio = fpath.stem.replace('trajectories_r', '')
        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        merged[ratio] = data['trajectories']
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='m7/data/trajectories_filtered')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = _repo_root / data_dir
    print(f'Scanning: {data_dir}')

    per_ratio_trajs = load_trajs(data_dir)

    stats = {'per_ratio': {}, 'teacher': {}, 'label': {}}
    label_by_teacher = defaultdict(Counter)

    for ratio, teachers_dict in sorted(per_ratio_trajs.items()):
        stats['per_ratio'][ratio] = {'teacher': {}, 'label': {}, 'n_traj': 0}
        for teacher, per_graph_list in sorted(teachers_dict.items()):
            n_traj = sum(len(g) for g in per_graph_list)
            stats['per_ratio'][ratio]['teacher'][teacher] = n_traj
            stats['per_ratio'][ratio]['n_traj'] += n_traj
            stats['teacher'][teacher] = stats['teacher'].get(teacher, 0) + n_traj
            for traj_list in per_graph_list:
                for tr in traj_list:
                    # 过滤后 is_correct 恒为 True，final_pred 即真标签
                    lbl = str(tr.final_pred)
                    stats['label'][lbl] = stats['label'].get(lbl, 0) + 1
                    stats['per_ratio'][ratio]['label'][lbl] = \
                        stats['per_ratio'][ratio]['label'].get(lbl, 0) + 1
                    label_by_teacher[teacher][tr.final_pred] += 1
        print(f'  {ratio}: {stats["per_ratio"][ratio]["n_traj"]} trajs, '
              f'{stats["per_ratio"][ratio]["teacher"]}')

    stats['label_by_teacher'] = {
        t: {str(k): v for k, v in c.items()} for t, c in sorted(label_by_teacher.items())}
    stats['n_traj'] = sum(stats['teacher'].values())

    out_json = data_dir / 'label_distribution.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {out_json}')

    # ---- 可视化 1: teacher 分布（含 per-ratio 分组条） ----
    teachers = sorted(stats['teacher'])
    ratios = sorted(stats['per_ratio'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(teachers, [stats['teacher'][t] for t in teachers])
    ax1.set_title('Teacher distribution (all ratios)')
    ax1.set_xlabel('teacher')
    ax1.set_ylabel('# trajectories')
    ax1.tick_params(axis='x', rotation=0)
    for i, t in enumerate(teachers):
        ax1.text(i, stats['teacher'][t], str(stats['teacher'][t]),
                 ha='center', va='bottom', fontsize=8)

    width = 0.8 / len(ratios)
    for j, ratio in enumerate(ratios):
        counts = [stats['per_ratio'][ratio]['teacher'].get(t, 0) for t in teachers]
        xs = [i + j * width - 0.4 + width / 2 for i in range(len(teachers))]
        ax2.bar(xs, counts, width=width, label=ratio)
    ax2.set_xticks(range(len(teachers)))
    ax2.set_xticklabels(teachers)
    ax2.set_title('Teacher distribution per ratio')
    ax2.set_xlabel('teacher')
    ax2.set_ylabel('# trajectories')
    ax2.legend()
    fig.tight_layout()
    out_png1 = data_dir / 'trajectory_teacher_dist.png'
    fig.savefig(out_png1, dpi=150)
    print(f'Saved: {out_png1}')

    # ---- 可视化 2: label 分布（整体 + 分 teacher） ----
    labels = sorted(stats['label'], key=int)
    x = range(len(labels))
    n_teacher = len(teachers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x, [stats['label'][l] for l in labels])
    ax1.set_title('Label distribution (all trajectories)')
    ax1.set_xlabel('graph label')
    ax1.set_ylabel('# trajectories')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    width = 0.8 / n_teacher
    for j, t in enumerate(teachers):
        lb = stats['label_by_teacher'].get(t, {})
        counts = [lb.get(l, 0) for l in labels]
        xs = [i + j * width - 0.4 + width / 2 for i in x]
        ax2.bar(xs, counts, width=width, label=t)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title(f'Label distribution per teacher ({n_teacher} teachers)')
    ax2.set_xlabel('graph label')
    ax2.set_ylabel('# trajectories')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out_png2 = data_dir / 'trajectory_label_dist.png'
    fig.savefig(out_png2, dpi=150)
    print(f'Saved: {out_png2}')


if __name__ == '__main__':
    main()
