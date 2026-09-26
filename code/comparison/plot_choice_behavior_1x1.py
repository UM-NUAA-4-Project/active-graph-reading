"""行为分析图 1×1 拆分版：choice_behavior_1x4 的四块面板各自出独立单图。

与 1×4 合并图（plot_choice_behavior_1x4.py）同数据、同编码、同聚合口径：
  a = cifar10 前沿度数：被选 vs 可选（逐步，1–10 + ≥11 桶）
  b = cifar10 chosen 节点真实信息增益（nats，0 线=揭示不带来熵减）
  c = COLLAB 前沿度数（同 a 编码）
  d = COLLAB chosen 节点真实信息增益（同 b 编码）
差别只有画布：Nature 单栏 89mm 独立画布（合并图内每格仅 ~45mm）。合并
图因格宽限制用的短说明文字与隔步刻度保持不变，保证单图与合并图逐要素
一致；度数面板用曲线旁直接标注（无图例，见 draw_degree）。
无面板字母（字母带由合并图/排版侧添加）。

产物（写入 result/cross-dataset-figures/behavior/choice/，各 pdf/png/svg）：
  choice_behavior_cifar10_degree.* / choice_behavior_cifar10_ig.*
  choice_behavior_collab_degree.*  / choice_behavior_collab_ig.*
  各 *.alignment.json（单面板 → NOT APPLICABLE 记录）

用法（仓库根运行）:
  python code/comparison/plot_choice_behavior_1x1.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SKILL_SCRIPTS = Path.home() / '.zcode' / 'skills' / 'nature-figure' / 'scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from plot_choice_behavior_1x4 import (  # noqa: E402
    CIF_JSON, COL_JSON, mpl_rc,
    aggregate, draw_degree, draw_ig,
)

BASE = Path(__file__).resolve().parents[1] / 'result'
OUT_DIR = BASE / 'cross-dataset-figures' / 'behavior' / 'choice'

FIG_W_IN, FIG_H_IN = 89 / 25.4, 58 / 25.4   # Nature 单栏 89mm
DPI = 600

PANELS = [
    dict(data='cif', kind='degree', stem='choice_behavior_cifar10_degree',
         dataset='CIFAR10', explainer='Chosen stays at degree 1;\navailable candidates grow'),
    dict(data='cif', kind='ig', stem='choice_behavior_cifar10_ig',
         dataset='CIFAR10', annotations='cifar10'),
    dict(data='col', kind='degree', stem='choice_behavior_collab_degree',
         dataset='COLLAB', explainer='Chosen above average\nat every step'),
    dict(data='col', kind='ig', stem='choice_behavior_collab_ig',
         dataset='COLLAB', annotations='collab'),
]


def main():
    cif = aggregate(CIF_JSON, 'CIFAR10')
    col = aggregate(COL_JSON, 'COLLAB')
    data = {'cif': cif, 'col': col}

    matplotlib.rcParams.update(mpl_rc)
    for p in PANELS:
        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
        fig.subplots_adjust(left=0.135, right=0.97, top=0.90, bottom=0.20)
        if p['kind'] == 'degree':
            draw_degree(ax, data[p['data']], p['dataset'],
                        explainer=p['explainer'])
        else:
            draw_ig(ax, data[p['data']], p['dataset'], p['annotations'])
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT_DIR / f"{p['stem']}.alignment.json"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        stem = OUT_DIR / p['stem']
        fig.savefig(f'{stem}.svg', bbox_inches='tight')
        fig.savefig(f'{stem}.pdf', bbox_inches='tight')
        fig.savefig(f'{stem}.png', dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"saved: {p['stem']}.{{svg,pdf,png}}")


if __name__ == '__main__':
    main()
