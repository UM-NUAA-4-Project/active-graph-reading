"""Programmatic QA for three_stage_compare figure (no visual inspection available)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_three_stage import load_stats, build_figure, STAGES, BUDGETS, SPLITS, SEEDS  # noqa: E402


def check_data(stats):
    print('== data spot checks ==')
    raw = {}
    for name, path, strategy, _c in STAGES:
        with open(path, encoding='utf-8') as f:
            raw[name] = json.load(f)
        for split in SPLITS:
            for seed in SEEDS:
                for b in BUDGETS:
                    mean, sem = stats[name][split][seed][b]
                    cell = raw[name][split][strategy][seed][f'{b:.2f}']
                    assert abs(mean - cell['mean']) < 1e-9, (name, split, seed, b)
                    assert abs(sem - cell['std'] / (cell['n'] ** 0.5)) < 1e-9
    print(f'  {len(STAGES)} stages x {len(SPLITS)} splits x {len(SEEDS)} seeds x '
          f'{len(BUDGETS)} budgets all match JSON mean/sem. OK')
    print('  means:')
    for name, _p, _s, _c in STAGES:
        line = []
        for split in SPLITS:
            for seed in SEEDS:
                vals = [stats[name][split][seed][b][0] for b in BUDGETS]
                line.append(f'{split}/{seed}=' + ','.join(f'{v:.3f}' for v in vals))
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
            x1, y1 = t1.get_position()
            # skip pairs inside the same legend or same axis title block
            if k1 == k2 == 'legend':
                continue
            if k1 == k2 == 'tick':
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


def check_structure(fig, axes, ref_valid, ref_test):
    print('== structure check ==')
    assert axes.shape == (2, 2)
    for ax in axes.flat:
        n_bars = len(ax.patches)
        assert n_bars == len(STAGES) * len(BUDGETS), n_bars
        assert np.allclose(ax.get_ylim(), (0.10, 0.70))
        assert len(ax.get_xticks()) == len(BUDGETS)
    for ri, seed in enumerate(SEEDS):
        for ci, split in enumerate(SPLITS):
            ax = axes[ri][ci]
            n_ref = sum(1 for ln in ax.lines if ln.get_linestyle() == '--')
            expect = 1 if (split == 'valid' and ref_valid is not None) or \
                           (split == 'test' and ref_test is not None) else 0
            assert n_ref == expect, (split, seed, n_ref, expect)
    assert len(fig.legends) == 1
    assert len(fig.legends[0].get_texts()) == len(STAGES)
    print(f'  2x2 grid, {len(STAGES)*len(BUDGETS)} bars/panel, shared ylim (0.10, 0.70). OK')
    print(f'  dashed full-graph ref in all panels (valid={ref_valid:.4f}, test={ref_test:.4f}). OK')
    print(f'  legend entries: {[t.get_text() for t in fig.legends[0].get_texts()]}')


def main():
    stats, ref_valid, ref_test = load_stats()
    check_data(stats)
    fig, axes = build_figure(stats, ref_valid, ref_test)
    check_structure(fig, axes, ref_valid, ref_test)
    check_text_overlaps(fig, axes)
    plt.close(fig)
    print('QA done')


if __name__ == '__main__':
    main()
