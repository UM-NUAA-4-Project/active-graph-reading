"""BC 复合误差签名分析（预测 4.1 / 命题 4.1 / 注 4.2，见 docs/ 数学分析文档第 4、8 章）。

读入若干条标注的 eval JSON 曲线（eval.py 输出的 eval_fixed.json / eval_learnable.json，
结构 {split: {strategy: {seed|mean: {budget: {mean,std,n}}}}}），对"基准曲线 − 目标曲线"
的逐预算差（gap）做线性 / 过原点线性 / 二次拟合，判别复合误差签名：
  - 线性（gap ≈ B·ε̄，过原点拟合优）        ⇒ 良性复合，(A1) 的预期；
  - 凸增（二次项显著为正，中点低于弦）      ⇒ 协变量偏移放大，BC 需 DAgger/RL 修正；
  - gap ≤ 0                                 ⇒ 基准不占优，签名不适用。

每条曲线来自一个 eval JSON：--eval 名字=路径[:策略键[:split]]。策略键缺省时
优先取 'learnable'，否则取 split 下第一个策略（fixed JSON 可用第三段指定，
如 fixed=...:bfs）。教师侧基准有两条来路：
  1) 免 GPU 恢复：trajectories/estimate_teacher_success.py 从成功过滤轨迹文件
     按 kept/(n_seeds×graphs) 恢复 train 侧教师成功率（生成端只保存
     is_correct=True，generator.py:129）——产出的 JSON 口径为 train split，
     用第 4 段覆盖对齐（teacher=x.json:gradient:train），跨 split 时脚本会警告；
  2) 干净口径：对教师策略在 held-out 上补跑 eval（test 侧），直接插入。

输出（默认第一个 JSON 所在目录）：
  {name}.png          两联图：准确率曲线 + gap 及拟合线
  {name}_summary.json 曲线数值与拟合参数（入论文/复算）

用法:
  python code/comparison/plot_bc_gap.py \
      --eval fixed=code/result/cifar10/fixed/eval_fixed.json:bfs \
             bc=code/result/cifar10/experiments/ev1/evidence/bc/eval_learnable.json \
             rl=code/result/cifar10/experiments/ev-R-read/evidence/rl/eval_learnable.json \
      --base fixed --targets bc rl
"""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np

_SPLIT_NAMES = ('train', 'valid', 'test')
_CURVE_COLORS = {'fixed': '#999999', 'teacher': '#DD8452', 'bc': '#4C72B0',
                 'rl': '#C44E52'}


def _resolve(p: str) -> Path:
    from tools.paths import resolve
    return resolve(p)


def _parse_eval_arg(arg: str):
    """name=path[:strategy[:split]] → (name, path, strategy or None, split or None)。

    第 4 段 split 覆盖用于教师成功率 JSON（其口径是 train split，见
    trajectories/estimate_teacher_success.py）；覆盖后跨 split 作差会打印警告。
    """
    name, _, rest = arg.partition('=')
    if not name or not rest:
        raise ValueError(f"--eval 项须为 name=path[:strategy[:split]]，收到：{arg}")
    parts = rest.rsplit(':', 2)
    if len(parts) == 3 and parts[1] and parts[2] in _SPLIT_NAMES:
        path, strategy, split = parts
        return name, path, (strategy or None), split
    if ':' in rest:
        path, strategy = rest.rsplit(':', 1)
        return name, path, strategy or None, None
    return name, rest, None, None


def _pick_strategy(data: dict, split: str, strategy: str | None):
    strategies = [s for s in data.get(split, {}) if isinstance(data[split][s], dict)]
    if strategy:
        if strategy not in strategies:
            raise KeyError(f"split '{split}' 下无策略 '{strategy}'（有：{strategies}）")
        return strategy
    if 'learnable' in strategies:
        return 'learnable'
    if not strategies:
        raise KeyError(f"split '{split}' 下没有任何策略曲线")
    return sorted(strategies)[0]


def _pick_split(data: dict, split: str | None):
    if split:
        if split not in data:
            raise KeyError(f"JSON 无 split '{split}'（有：{list(data)}）")
        return split
    for name in _SPLIT_NAMES:
        if name in data:
            return name
    raise KeyError(f"JSON 中找不到 train/valid/test 任一 split（有：{list(data)}）")


def extract_curve(data: dict, split: str | None, strategy: str | None):
    """→ (split, strategy, budgets[], acc[])：优先 'mean'，否则跨 seed 平均。"""
    split = _pick_split(data, split)
    strategy = _pick_strategy(data, split, strategy)
    seed_dict = data[split][strategy]
    if 'mean' in seed_dict:
        stats = seed_dict['mean']
    else:  # 无预聚合则跨 seed 平均
        seeds = [v for k, v in seed_dict.items() if k.startswith('s')]
        budget_keys = sorted(seeds[0])
        stats = {b: {'mean': float(np.mean([s[b]['mean'] for s in seeds]))}
                 for b in budget_keys}
    budgets = sorted(float(b) for b in stats)
    acc = [float(stats.get(f'{b:.2f}', stats.get(str(b), {}))['mean'])
           for b in budgets]
    return split, strategy, budgets, acc


def _fit(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    pred = np.polyval(coeffs, x)
    resid = float(np.sum((y - pred) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - resid / total if total > 0 else float('nan')
    return {'coeffs': [float(c) for c in coeffs], 'resid': resid, 'r2': r2}


def fit_gap(budgets, gaps):
    """返回三种拟合与启发式签名标签。"""
    x = np.asarray(budgets, dtype=float)
    y = np.asarray(gaps, dtype=float)
    out = {'n_points': len(x), 'gaps': [float(g) for g in y]}
    if len(x) < 2:
        out['verdict'] = '点数不足（<2），仅报告 gap'
        return out
    # 过原点：强制截距 0（理论 gap ≤ B·ε 过原点）
    k0 = float(np.sum(x * y) / np.sum(x * x))
    pred0 = k0 * x
    resid0 = float(np.sum((y - pred0) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    out['origin'] = {'slope_per_ratio': k0,
                     'r2': 1.0 - resid0 / total if total > 0 else float('nan'),
                     'resid': resid0}
    out['affine'] = _fit(x, y, 1)
    if len(x) >= 3:
        out['quad'] = _fit(x, y, 2)
    # 启发式签名（3 点只是提示性判别，谨慎外推）
    if np.all(y <= 1e-9):
        out['verdict'] = 'gap ≤ 0：基准不占优，复合误差签名不适用'
        return out
    second_diff = (y[2] - y[1]) - (y[1] - y[0]) if len(x) >= 3 else None
    max_abs = float(np.max(np.abs(y)))
    if second_diff is not None and second_diff > 0.5 * max_abs:
        out['verdict'] = '凸增签名：偏离概率增长快于线性 ⇒ 协变量偏移放大（DAgger/RL 的存在理由）'
    elif second_diff is not None and second_diff < -0.1 * max_abs:
        out['verdict'] = ('次线性/趋平签名（良性）：gap 增长慢于线性、未见发散——'
                          '与终端稀疏奖励下的线性复合界一致（经典 T² 是密集奖励口径），'
                          '未观测到协变量偏移放大')
    elif out['origin']['r2'] is not None and out['origin']['r2'] > 0.9:
        out['verdict'] = (f"线性签名（良性复合）：gap ≈ {k0:.3f}·B，"
                          f"隐含每步错误率 ε̄ ≈ {k0 * float(x[1] - x[0]):.4f} / "
                          f"{(x[1] - x[0]) * 100:.0f}% 预算步")
    else:
        out['verdict'] = '模式不明（次线性或不规则），查看拟合参数'
    return out


def main():
    parser = argparse.ArgumentParser(
        description='BC 复合误差签名：曲线对 gap-vs-B 拟合')
    parser.add_argument('--eval', nargs='+', required=True,
                        help='标注曲线 name=path[:strategy]，可多条')
    parser.add_argument('--base', default='',
                        help='基准（expert 侧）曲线名；默认第一个')
    parser.add_argument('--targets', nargs='*', default=[],
                        help='目标曲线名；默认除基准外全部')
    parser.add_argument('--split', default='',
                        help='train/valid/test；默认自动选第一个可用')
    parser.add_argument('--budgets', nargs='*', type=float, default=[],
                        help='只取这些预算比例；默认取各曲线交集')
    parser.add_argument('--output-dir', default='',
                        help='默认第一个 JSON 所在目录')
    parser.add_argument('--name', default='bc_gap', help='输出文件名前缀')
    parser.add_argument('--ceiling', type=float, default=None,
                        help='全图信息天花板（0-1，由 comparison/info_ceiling.py '
                             '算出），在准确率面板画水平虚线')
    args = parser.parse_args()

    curves = {}
    raw = {}
    first_path = None
    for arg in args.eval:
        name, path, strategy, split_ov = _parse_eval_arg(arg)
        path = _resolve(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        raw[name] = (data, path, strategy, split_ov)
        if first_path is None:
            first_path = path

    # split 解析：有第 4 段覆盖的曲线用其覆盖；其余必须共享一个 split（优先 test）
    no_ov = {n: v for n, v in raw.items() if v[3] is None}
    if args.split:
        common = args.split
        missing = [n for n, (d, _, _, _) in no_ov.items() if common not in d]
        if missing:
            raise SystemExit(f'split {common!r} 在这些 JSON 中缺失: {missing}，'
                             f'请改用 --split')
    elif no_ov:
        common = next((s for s in ('test', 'valid', 'train')
                       if all(s in d for d, _, _, _ in no_ov.values())), None)
        if common is None:
            raise SystemExit('未覆盖 split 的各 JSON 没有公共 split'
                             '（train/valid/test），请用 --split 或第 4 段覆盖')
    else:
        common = None

    for name, (data, _path, strategy, split_ov) in raw.items():
        use = split_ov or common
        _s, strat, budgets, acc = extract_curve(data, use, strategy)
        curves[name] = {'split': _s, 'strategy': strat, 'file': str(_path),
                        'budgets': budgets, 'acc': acc}
        print(f"curve '{name}': {_s}/{strat}  "
              + '  '.join(f'{b:.0%}={a:.4f}' for b, a in zip(budgets, acc)))
    used_splits = sorted({c['split'] for c in curves.values()})
    if len(used_splits) > 1:
        per = {n: c['split'] for n, c in curves.items()}
        print(f"⚠ 跨 split 比较：{per}\n  gap 的两端口径不同"
              f"（典型：教师=train 侧恢复值，读者=test 侧评估值）。"
              f"分布 iid 时可接受，否则先做 test 侧教师 eval。")

    base = args.base or next(iter(curves))
    targets = args.targets or [n for n in curves if n != base]
    for t in targets:
        if t not in curves:
            raise SystemExit(f"--targets 引用了未定义曲线 '{t}'")

    common = None
    for c in curves.values():
        s = set(c['budgets'])
        common = s if common is None else (common & s)
    budgets = sorted(common)
    if args.budgets:
        budgets = [b for b in budgets if any(abs(b - w) < 1e-9
                                             for w in args.budgets)]
    if not budgets:
        raise SystemExit('各曲线没有公共预算点')

    out_dir = (_resolve(args.output_dir) if args.output_dir
               else first_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, all_curves = [], {}
    for name, c in curves.items():
        idx = [i for i, b in enumerate(c['budgets']) if b in budgets]
        all_curves[name] = {'split': c['split'], 'strategy': c['strategy'],
                            'file': c['file'],
                            'budgets': [c['budgets'][i] for i in idx],
                            'acc': [c['acc'][i] for i in idx]}
    for t in targets:
        b_acc = all_curves[base]['acc']
        t_acc = all_curves[t]['acc']
        gaps = [a - b for a, b in zip(b_acc, t_acc)]
        fit = fit_gap(budgets, gaps)
        pairs.append({'base': base, 'target': t, 'budgets': budgets,
                      'gap': gaps, 'fit': fit})
        print(f"\n[{base} − {t}]  gap: "
              + '  '.join(f'{b:.0%}:{g:+.4f}' for b, g in zip(budgets, gaps)))
        print(f"  origin-fit slope={fit['origin']['slope_per_ratio']:.3f} "
              f"(R²={fit['origin']['r2']:.3f})   "
              f"affine={[round(c, 4) for c in fit['affine']['coeffs']]} "
              f"(R²={fit['affine']['r2']:.3f})"
              + (f"   quad={[round(c, 4) for c in fit['quad']['coeffs']]}"
                 if 'quad' in fit else ''))
        print(f"  verdict: {fit['verdict']}")

    _plot(all_curves, pairs, base, budgets, out_dir, args.name, args.ceiling)
    summary = {'base': base, 'targets': targets, 'budgets': budgets,
               'curves': all_curves, 'pairs': pairs}
    summary_path = out_dir / f'{args.name}_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding='utf-8')
    print(f"\nSummary saved to {summary_path}")


def _plot(curves, pairs, base, budgets, out_dir: Path, name: str,
          ceiling: float | None = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for cname, c in curves.items():
        ax1.plot([b * 100 for b in c['budgets']], c['acc'], 'o-',
                 color=_CURVE_COLORS.get(cname, None), label=cname)
    ax1.set_xlabel('Budget Ratio (%)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Budget')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    if ceiling is not None:
        ax1.axhline(ceiling, color='purple', linestyle=':', linewidth=1.4,
                    label=f'info ceiling ({ceiling:.3f})')
        ax1.legend(fontsize=8)

    x = np.linspace(0, max(budgets) * 100, 50)
    for p in pairs:
        gaps = [g * 100 for g in p['gap']]
        ax2.plot([b * 100 for b in p['budgets']], gaps, 'o-',
                 color=_CURVE_COLORS.get(p['target'], None),
                 label=f"{p['base']}−{p['target']}")
        f = p['fit']
        if 'origin' in f:
            k = f['origin']['slope_per_ratio']
            ax2.plot(x, k * (x / 100) * 100, '--', alpha=0.5,
                     color=_CURVE_COLORS.get(p['target'], None))
    ax2.set_xlabel('Budget Ratio (%)')
    ax2.set_ylabel('Gap (accuracy points)')
    ax2.set_title('Gap vs Budget (dashed = through-origin linear fit)')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = out_dir / f'{name}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


if __name__ == '__main__':
    main()
