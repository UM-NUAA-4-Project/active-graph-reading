"""逐步前沿子图显存曲线图（benchmark_inference_cost.py 的 mem_series 可视化）。

输入 benchmark_cost.json（systems.{mode}.{split}.mem_series.per_step = 逐步
跨图 mean/std/n），画每条系统一条「单图单步前沿子图前向瞬态显存」曲线
（均值线 ± 标准差带）；full-graph 无步概念，画每图近似参考横线
（整批瞬态÷批内图数，标注 approx）。输出 PNG + 同名 CSV（绘图数据）。

用法:
  python code/comparison/plot_mem_series.py \
      --json code/result/cifar10/cost/benchmark_cost.json
  # 步位样本数 < --min-n 的尾部截去（小样本步统计不稳）；--max-step 强制截断
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# 与 plot_eval_results 的策略色一致（learnable 蓝、sagpool 青）
SERIES_COLORS = {
    'bagr': '#4C72B0',
    'sagpool': '#64B5CD',
    'softmask': '#55A868',
    'agentnet': '#8172B2',
}
SERIES_LABELS = {
    'bagr': 'BAGR (ours)',
    'sagpool': 'SAGPool',
    'softmask': 'Soft-Mask',
    'agentnet': 'AgentNet',
}
_CURVE_ORDER = ('bagr', 'sagpool', 'softmask', 'agentnet')
_SIM_BATCH = 16   # 曲线点与全图上界统一 ×16，模拟 16 图批量部署口径


def main():
    parser = argparse.ArgumentParser(description='逐步前沿子图显存曲线')
    parser.add_argument('--json', required=True, help='benchmark_cost.json')
    parser.add_argument('--split', default='test')
    parser.add_argument('--min-n', type=int, default=5,
                        help='步位跨图样本数下限（低于则不画该步）')
    parser.add_argument('--max-step', type=int, default=0,
                        help='x 轴步号上限（0=按 min-n 自动截尾）')
    parser.add_argument('--output', default='',
                        help='输出 PNG（缺省 <json 同目录>/mem_series.png）')
    args = parser.parse_args()

    json_path = Path(args.json)
    data = json.loads(json_path.read_text(encoding='utf-8'))
    cfg = data.get('config', {})
    dataset = cfg.get('dataset', '?')
    out_path = (Path(args.output) if args.output
                else json_path.parent / 'mem_series.png')

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # 右轴：full-graph 参考线独占（×16 模拟批量后量级远超协议曲线，
    # 独立轴给协议曲线完整绘图空间）
    ax2 = ax.twinx()
    csv_rows = [('mode', 'step', 'mean_mib', 'std_mib', 'n',
                 'mean_net_mib', 'std_net_mib')]
    all_means = []   # 左轴界限只按协议曲线均值定（step≥2）
    bagr_net_floor = None

    # full-graph：每图近似参考线（批瞬态÷批内图数）→ 右轴 ×16 模拟批量
    fg = data.get('systems', {}).get('full-graph', {}).get(args.split, {})
    if fg.get('mem_per_graph_batch_mean_mib') is not None:
        ref = fg['mem_per_graph_batch_mean_mib'] * _SIM_BATCH
        ax2.axhline(y=ref, color='#999999', linestyle='--', linewidth=1.2,
                    label='full-graph')
        ax2.set_ylim(0, ref * 1.6)
        ax2.set_ylabel('full-graph batch total (MiB)', fontsize=9,
                       color='#777777')
        ax2.tick_params(axis='y', colors='#999999')
        csv_rows.append(('full-graph', '',
                         fg['mem_per_graph_batch_mean_mib'],
                         fg.get('mem_per_graph_batch_std_mib', ''),
                         fg.get('mem_per_graph_batch_n', '')))

    plotted = 0
    for mode in _CURVE_ORDER:
        m = data.get('systems', {}).get(mode, {}).get(args.split, {})
        series = m.get('mem_series')
        if not series:
            continue
        per_step = series['per_step']
        # bagr 有实测固定底（cuDNN LSTM workspace）时优先净口径
        use_net = (mode == 'bagr' and 'mean_net' in next(
            iter(per_step.values())))
        k_mean = 'mean_net' if use_net else 'mean'
        k_std = 'std_net' if use_net else 'std'
        steps = sorted(int(t) for t in per_step
                       if per_step[t]['n'] >= args.min_n)
        steps = [t for t in steps if t >= 2]   # 跳过 step1（AgentNet 缓存尖峰）
        if args.max_step > 0:
            steps = [t for t in steps if t <= args.max_step]
        if not steps:
            continue
        means = [per_step[str(t)][k_mean] * _SIM_BATCH for t in steps]
        stds = [per_step[str(t)][k_std] * _SIM_BATCH for t in steps]
        color = SERIES_COLORS.get(mode, '#4C72B0')
        label = SERIES_LABELS.get(mode, mode)
        if use_net:
            bagr_net_floor = series.get('net_floor_mib')
            label = 'BAGR (ours)'
        ax.plot(steps, means, marker='o', markersize=3.5,
                color=color, label=label, linewidth=1.6, zorder=3)
        ax.fill_between(steps, [a - b for a, b in zip(means, stds)],
                        [a + b for a, b in zip(means, stds)],
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        for t, mu, sd in zip(steps, means, stds):
            st = per_step[str(t)]
            csv_rows.append((mode, t, st['mean'], st['std'], st['n'],
                             st.get('mean_net', ''), st.get('std_net', '')))
        all_means += means
        plotted += 1

    if not plotted:
        raise SystemExit(f"JSON 中没有可画的 mem_series（{json_path}）")

    # 左轴按协议曲线均值定界；右轴独立（见上方 full-graph 分支）
    ymax = max(all_means)
    ax.set_ylim(0, 1.15 * ymax)

    ax.set_xlabel('Reveal step $t$ (nodes revealed so far)', fontsize=10)
    ax.set_ylabel(f'Frontier-subgraph forward\npeak transient memory per '
                  f'{_SIM_BATCH}-graph batch (MiB)', fontsize=9.5)
    n_seeds = cfg.get('n_seeds', '?')
    budgets = cfg.get('budgets', [])
    b_str = '/'.join(f'{b:g}' for b in budgets)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    # 顶部条带布局：suptitle 最上，图例条带居中，axes 在 rect 内不受遮挡
    fig.suptitle(f'{dataset} — per-step frontier-subgraph GPU memory '
                 f'({args.split}, budget {b_str}, {n_seeds} seed)',
                 fontsize=10.5, y=0.995, va='top')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center',
              bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False,
              columnspacing=1.2, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"PNG saved to {out_path}")

    csv_path = out_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(csv_rows)
    print(f"CSV saved to {csv_path}")


if __name__ == '__main__':
    main()
