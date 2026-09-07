"""从轨迹文件恢复教师成功率（预测 4.1 的 teacher 侧基准，免 GPU）。

背景：轨迹在生成端只保存 is_correct=True 的成功轨迹
（trajectories/core/generator.py: `if traj.is_correct: ...`），教师终端
准确率本身从未落盘。但生成时每图每教师恰尝试 config.n_seeds 个种子，
且文件按图保留了成功条数，故

    教师成功率(budget) = Σ_g kept(g) / (n_seeds × n_graphs)

可直接由文件恢复（train split、生成所用冻结分类器口径）。
每图成功数不高于 n_seeds（无重复并入）时估计无偏；n_seeds 缺失或
slots ≠ n_graphs 时按注记降级处理。

输出 eval-schema JSON（供 comparison/plot_bc_gap.py 直接消费）：
  {<split-key>: {teacher: {"mean": {budget: {mean, std, n}}}}, "config": {...}}
其中 mean = 成功率，std = 逐图成功比例的样本标准差（图间异质性），
n = 图数。split-key 默认 'train'（口径是训练分布）；plot_bc_gap 侧用
第 4 段 split 覆盖对齐（如 teacher=x.json:gradient:train）。

用法:
  python code/trajectories/estimate_teacher_success.py \
      --data-dir data/cifar10/trajectories \
      --output result/cifar10/teacher_eval/eval_teacher_train.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np

from tools.trajectory_io import (load_trajectory_file,  # noqa: E402
                                 ratio_key_from_name)


def estimate_file(path: Path):
    """→ {(teacher, budget): dict(rate, std, n, warn)} 单文件估计。"""
    data = load_trajectory_file(str(path))
    cfg = data.get('config', {})
    n_seeds = int(cfg.get('n_seeds', 0))
    n_graphs = int(cfg.get('n_graphs', 0))
    budget = cfg.get('budget_ratio')
    if budget is None:
        budget = ratio_key_from_name(path.name)
    if not n_seeds or not n_graphs:
        raise SystemExit(f"{path.name}: config 缺 n_seeds/n_graphs，无法恢复")
    out = {}
    for teacher, per_graph in data.get('trajectories', {}).items():
        counts = np.array([len(x) for x in per_graph], dtype=float)
        total = float(counts.sum())
        warn = []
        if counts.max() > n_seeds:
            warn.append(f'每图成功数最大值 {counts.max():.0f} > n_seeds'
                        f'（可能多轮合并，估计偏高）')
        if len(counts) != n_graphs:
            warn.append(f'slots={len(counts)} ≠ n_graphs={n_graphs}'
                        f'（空图槽可能被省略或子集生成，以 slots 为分母下界）')
        universe = n_graphs if len(counts) == n_graphs else len(counts)
        rate = total / (n_seeds * universe)
        frac = (counts / n_seeds).tolist()
        out[(teacher, round(float(budget), 4))] = {
            'rate': rate, 'fracs': frac, 'n': int(universe), 'warn': warn,
            'total_kept': int(total), 'n_seeds': n_seeds,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description='轨迹文件 → 教师成功率 eval JSON')
    parser.add_argument('--data-dir', nargs='+', required=True,
                        help='轨迹目录（含 trajectories_*.pt），可多个')
    parser.add_argument('--output', required=True, help='输出 eval-schema JSON')
    parser.add_argument('--split-key', default='train',
                        help="输出 JSON 的 split 键名（默认 'train'：估计口径是训练分布）")
    args = parser.parse_args()

    from tools.paths import resolve
    agg = defaultdict(lambda: {'total': 0.0, 'fracs': [], 'n': 0,
                               'warn': set(), 'n_seeds': set()})
    for dir_arg in args.data_dir:
        ddir = resolve(dir_arg)
        files = sorted(ddir.glob('trajectories_*.pt'))
        if not files:
            raise SystemExit(f'{ddir} 下没有 trajectories_*.pt')
        for f in files:
            for (teacher, budget), est in estimate_file(f).items():
                a = agg[(teacher, budget)]
                a['total'] += est['total_kept']
                a['fracs'].extend(est['fracs'])
                a['n'] += est['n']
                a['warn'].update(est['warn'])
                a['n_seeds'].add(est['n_seeds'])

    result = {args.split_key: {}}
    notes = []
    for (teacher, budget), a in sorted(agg.items()):
        if len(a['n_seeds']) > 1:
            a['warn'].add(f'多文件 n_seeds 不一致: {sorted(a["n_seeds"])}')
        rate = a['total'] / (a['n'] * max(a['n_seeds']))
        std = float(np.std(a['fracs'], ddof=1)) if len(a['fracs']) > 1 else 0.0
        mean_entry = result[args.split_key].setdefault(
            teacher, {'mean': {}})
        mean_entry['mean'][f'{budget:.2f}'] = {
            'mean': round(rate, 6), 'std': round(std, 6), 'n': a['n']}
        for w in sorted(a['warn']):
            notes.append(f'{teacher}@{budget:.2f}: {w}')
        print(f"{teacher:22s} @{budget:.2f}: rate={rate:.4f}  "
              f"(kept={a['total']:.0f} / n_seeds×graphs={a['n'] * max(a['n_seeds'])})")

    result['config'] = {
        'source_dirs': args.data_dir,
        'note': '教师成功率由成功过滤轨迹文件恢复：rate = kept/(n_seeds×graphs)；'
                '口径 = 训练 split + 生成时所用冻结分类器；std = 逐图成功比例的样本标准差。',
        'warnings': notes,
    }
    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f"\nSaved: {out_path}")
    for w in notes:
        print(f'  ⚠ {w}')


if __name__ == '__main__':
    main()
