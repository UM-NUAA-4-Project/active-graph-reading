"""三阶段对比图 QA：数据一致性、结构、文字重叠检查（泛化版，无 CLI 改动需求）。

自 code/m5/base/analysis/qa_three_stage.py 迁移，适配泛化版 plot_three_stage。
用法：先记下 plot_three_stage 的参数，用相同参数调用本脚本，例如：
  python code/comparison/qa_three_stage.py --fixed-json ... --bc-json ... \
      --rl-json ... --splits test --output-dir result/ppa/comparison
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_three_stage import (build_stages, build_figure, load_stats,  # noqa: E402
                              common_splits, resolve, YMIN, YMAX)


def check_data(stats, splits, budgets):
    print('== data spot checks ==')
    raw = {}
    for name in stats:
        with open(stats[name]['path'], encoding='utf-8') as f:
            raw[name] = json.load(f)
        strategy = stats[name]['strategy']
        for split in splits:
            for b in budgets:
                mean, sem = stats[name][split]['mean'][b]
                cell = raw[name][split][strategy]['mean'][f'{b:.2f}']
                assert abs(mean - cell['mean']) < 1e-9, (name, split, b)
                assert abs(sem - cell['std'] / (max(cell['n'], 1) ** 0.5)) < 1e-9
    print(f'  {len(stats)} stages x {len(splits)} splits x {len(budgets)} budgets '
          'all match JSON mean/sem. OK')
    for name in stats:
        line = []
        for split in splits:
            vals = [stats[name][split]['mean'][b][0] for b in budgets]
            line.append(f'{split}=' + ','.join(f'{v:.3f}' for v in vals))
        print(f'    {name:6s} ' + '  '.join(line))


def check_text_overlaps(fig, axes):
    print('== text overlap check ==')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    texts = []
    for ax in axes.flat:
        for t in ax.texts:
            texts.append(('panel', t))
        for lbl in (ax.get_xticklabels() + ax.get_yticklabels()):
            if lbl.get_text():
                texts.append(('tick', lbl))
        if ax.get_title():
            texts.append(('title', ax.title))
        if ax.get_xlabel():
            texts.append(('xlabel', ax.xaxis.label))
        if ax.get_ylabel():
            texts.append(('ylabel', ax.yaxis.label))
    for t in fig.texts:
        texts.append(('figtext', t))
    for leg in fig.legends:
        for t in leg.get_texts():
            texts.append(('legend', t))

    problems = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            k1, t1 = texts[i]
            k2, t2 = texts[j]
            b1 = t1.get_window_extent(renderer)
            b2 = t2.get_window_extent(renderer)
            if not b1.overlaps(b2):
                continue
            if k1 == k2 == 'legend' or k1 == k2 == 'tick':
                continue
            inter = b1.intersection(b2)
            area = inter.width * inter.height if inter is not None else 0
            if area < 200:
                continue
            problems += 1
            print(f'  OVERLAP: [{k1}] "{t1.get_text()}" <-> [{k2}] "{t2.get_text()}" '
                  f'area={area:.0f}px2')
    if problems == 0:
        print('  no significant text overlaps. OK')


def check_structure(fig, axes, splits, budgets, stages, ref_valid, ref_test):
    print('== structure check ==')
    assert axes.shape == (1, len(splits))
    for ax in axes.flat:
        assert len(ax.patches) == len(stages) * len(budgets)
        assert np.allclose(ax.get_ylim(), (YMIN, YMAX))
        assert len(ax.get_xticks()) == len(budgets)
    for ci, split in enumerate(splits):
        ax = axes[0][ci]
        n_ref = sum(1 for ln in ax.lines if ln.get_linestyle() == '--')
        ref = ref_valid if split == 'valid' else ref_test
        assert n_ref == (1 if ref is not None else 0), (split, n_ref)
    assert len(fig.legends) == 1
    assert len(fig.legends[0].get_texts()) == len(stages)
    print(f'  1x{len(splits)} grid, {len(stages) * len(budgets)} bars/panel, '
          f'ylim ({YMIN}, {YMAX}). OK')
    print(f'  legend entries: {[t.get_text() for t in fig.legends[0].get_texts()]}')


def main():
    parser = argparse.ArgumentParser(description='三阶段对比图 QA')
    parser.add_argument('--fixed-json', required=True)
    parser.add_argument('--bc-json', default='')
    parser.add_argument('--rl-json', default='')
    parser.add_argument('--splits', nargs='+', default=['valid', 'test'])
    parser.add_argument('--budgets', type=float, nargs='+',
                        default=[0.05, 0.10, 0.15])
    parser.add_argument('--ref-valid', type=float, default=None)
    parser.add_argument('--ref-test', type=float, default=None)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    stages = build_stages(args)
    stats, ref_valid, ref_test = load_stats(stages, args.splits, args.budgets,
                                            args.ref_valid, args.ref_test)
    splits = common_splits(stats, args.splits)
    check_data(stats, splits, args.budgets)
    fig, axes = build_figure(stats, splits, args.budgets, ref_valid, ref_test)
    check_structure(fig, axes, splits, args.budgets, list(stats),
                    ref_valid, ref_test)
    check_text_overlaps(fig, axes)
    plt.close(fig)
    print('QA done')


if __name__ == '__main__':
    main()
