"""把四张 RL 奖励曲线 SVG 合并为 2×2 大图。

面板顺序（用户指定）：
  a = cifar10 scratch-rl 对比（rl_reward_scratch_vs_evR_cifar10.svg）
  b = collab  scratch-rl 对比（rl_reward_scratch_vs_evR_collab.svg）
  c = collab  奖励源切换（rl_reward_switch_collab.svg，仅精度曲线版）
  d = cifar10 奖励源切换（rl_reward_switch_cifar10.svg，仅精度曲线版）
四张源图同为 matplotlib 89mm 单栏导出（svg.fonttype='none'，文字可编辑）。
合并方式：字符串拼接——每张图的 <svg> 内容原样嵌入外层画布的嵌套
<svg>（自带 viewBox + preserveAspectRatio 等比缩放居中），不做任何重绘，
保证与源图逐像素一致。每格左上方加粗面板字母 a–d（字母带在源图上边界
之外，不与图内标题重叠）。

1×4 单行版已弃用：整条缩到 \\linewidth 后字号只剩约 3pt，不可读
（2026-09-25 视觉验收结论）；2×2 后缩放系数约 0.87，有效字号约 6pt。

用法:
  python code/comparison/merge_reward_svgs_1x4.py
产物: result/cross-dataset-figures/reward-curves/rl_reward_curves_2x2.svg
（png 预览可用 PyMuPDF 渲染，仅作检查用）。
"""

import re
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1] / 'result'

SOURCES = [  # (路径, 面板字母)
    (RESULT / 'cross-dataset-figures/reward-curves/rl_reward_scratch_vs_evR_cifar10.svg', 'a'),
    (RESULT / 'cross-dataset-figures/reward-curves/rl_reward_scratch_vs_evR_collab.svg', 'b'),
    (RESULT / 'cross-dataset-figures/reward-curves/rl_reward_switch_collab.svg', 'c'),
    (RESULT / 'cross-dataset-figures/reward-curves/rl_reward_switch_cifar10.svg', 'd'),
]
OUT = RESULT / 'cross-dataset-figures' / 'reward-curves' / 'rl_reward_curves_2x2.svg'

GAP_X = 16.0      # 面板间距（pt）
GAP_Y = 18.0
MARGIN_L = 6.0
MARGIN_R = 6.0
MARGIN_T = 14.0   # 字母带
MARGIN_B = 3.0
NCOL = 2


def parse_svg(path: Path):
    """返回 (width_pt, height_pt, viewBox, 内部 XML 文本)。"""
    text = path.read_text(encoding='utf-8')
    m = re.search(r'<svg\b([^>]*)>(.*)</svg>\s*$', text, re.S)
    attrs, inner = m.group(1), m.group(2)
    w = float(re.search(r'width="([\d.]+)pt"', attrs).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', attrs).group(1))
    vb = re.search(r'viewBox="([^"]+)"', attrs).group(1)
    return w, h, vb, inner


def main():
    parsed = [parse_svg(p) for p, _ in SOURCES]
    cell_w = max(w for w, _, _, _ in parsed)
    cell_h = max(h for _, h, _, _ in parsed)
    n = len(parsed)
    nrow = (n + NCOL - 1) // NCOL
    canvas_w = MARGIN_L + NCOL * cell_w + (NCOL - 1) * GAP_X + MARGIN_R
    canvas_h = MARGIN_T + nrow * cell_h + (nrow - 1) * GAP_Y + MARGIN_B

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{canvas_w:.2f}pt" height="{canvas_h:.2f}pt" '
        f'viewBox="0 0 {canvas_w:.2f} {canvas_h:.2f}">',
        f'<rect width="{canvas_w:.2f}" height="{canvas_h:.2f}" fill="white"/>',
    ]
    for i, ((w, h, vb, inner), (_, letter)) in enumerate(zip(parsed, SOURCES)):
        row, col = divmod(i, NCOL)
        x = MARGIN_L + col * (cell_w + GAP_X)
        y = MARGIN_T + row * (cell_h + GAP_Y)
        parts.append(
            f'<text x="{x + 1:.2f}" y="{y - 4:.2f}" '
            f'style="font-family: Arial, Helvetica, sans-serif; '
            f'font-weight: bold; font-size: 9px" fill="#1A1A1A">{letter}</text>')
        # 1:1 平移拼接（四图同为 89mm 栏宽，无需缩放），每格内容顶对齐
        parts.append(f'<g transform="translate({x:.2f},{y:.2f})">'
                     f'{inner}</g>')
    parts.append('</svg>')
    OUT.write_text('\n'.join(parts), encoding='utf-8')
    print(f'merged {n} svgs -> {OUT}')
    print(f'canvas: {canvas_w:.1f} x {canvas_h:.1f} pt '
          f'({canvas_w / 72 * 25.4:.0f} x {canvas_h / 72 * 25.4:.0f} mm)')


if __name__ == '__main__':
    main()
