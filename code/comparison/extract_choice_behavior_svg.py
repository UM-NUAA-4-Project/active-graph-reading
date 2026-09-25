"""从已发布行为分析图的 SVG 源码逆向提取 b/c 面板数据。

背景：cifar10 的 choice_behavior_b10.json 已被清理，但发布图
choice_behavior_b10.svg（svg.fonttype='none'）保留了全部绘制几何——
数据标记 <use x= y=（带系列颜色）、置信带 path（fill 颜色+透明度）、
刻度 <use>（精确像素位）与刻度标签文本。据此可无损恢复：
  b 面板：被选/可选平均前沿度数的逐步均值 + 95% CI 带（1–10 步 + ≥11 桶）
  c 面板：IG 排名分位逐步均值 + CI 带（2–11 步）

像素→数据坐标用各面板 tick（值, 像素）线性拟合。COLLAB 的 JSON 还在，
无需提取。

用法:
  python code/comparison/extract_choice_behavior_svg.py \
      --svg code/result/cifar10/experiments/ev-R-read/evidence/rl/behavior-analysis/choice_behavior_b10.svg

输出: SVG 同目录 choice_behavior_b10_from_svg.json（含逐步均值/CI 与
提取元数据），并打印 README 已知锚点数字用于校验。
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TEAL = '#2e7d64'    # b: chosen
GRAY = '#767676'    # b: available
ORANGE = '#b65a1f'  # c: IG


def local(tag):
    return tag.split('}')[-1]


def parent_map(root):
    return {c: p for p in root.iter() for c in p}


def axes_of(elem, pm):
    while elem is not None:
        if local(elem.tag) == 'g' and (elem.get('id') or '').startswith('axes_'):
            return elem.get('id')
        elem = pm.get(elem)
    return None


def _ancestors(elem, pm):
    elem = pm.get(elem)
    while elem is not None:
        yield elem
        elem = pm.get(elem)


def parse_path_points(d):
    """绝对坐标 M/L 段的全部顶点（straight-line path，无 C）。"""
    assert 'C' not in d, 'unexpected curved path'
    nums = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', d)]
    return list(zip(nums[0::2], nums[1::2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--svg', required=True)
    args = ap.parse_args()
    svg_path = Path(args.svg)

    root = ET.parse(svg_path).getroot()
    pm = parent_map(root)

    # ---- 收集各 axes 的 tick（像素↔值）与数据元素 ----
    ticks = {}     # axes -> {'x': [(px, val)], 'y': [(py, val)]}
    markers = {}   # axes -> [(style, px, py)]
    bands = {}     # axes -> [(fill_color, points)]
    defs_paths = {}  # defs path id -> d（FillBetween 多边形，y 轴负向，需 use 偏移）
    for e in root.iter():
        if local(e.tag) == 'path' and e.get('id'):
            pid = e.get('id')
            if e.get('d') and [a for a in _ancestors(e, pm)
                               if local(a.tag) == 'defs']:
                defs_paths[pid] = e.get('d')
    for e in root.iter():
        eid = e.get('id') or ''
        ax = axes_of(e, pm)
        if ax is None:
            continue
        if eid.startswith(('xtick_', 'ytick_')):
            kind = eid[:1]
            use = next((u for u in e.iter() if local(u.tag) == 'use'), None)
            txt = next((t for t in e.iter() if local(t.tag) == 'text'), None)
            if use is None or txt is None or not (txt.text or '').strip():
                continue
            val = txt.text.strip().replace('≥', '')
            try:
                val = float(val)
            except ValueError:
                continue   # 分类刻度（panel a 的 Gradient/Info gain/...）
            px, py = float(use.get('x')), float(use.get('y'))
            ticks.setdefault(ax, {'x': [], 'y': []})[kind].append(
                (px if kind == 'x' else py, val))
        elif local(e.tag) == 'use':
            style = e.get('style') or ''
            m = re.search(r'stroke: (#[0-9a-f]{6})', style)
            if m and m.group(1) != '#000000':
                markers.setdefault(ax, []).append(
                    (m.group(1), float(e.get('x')), float(e.get('y'))))
                continue
            # FillBetween 置信带：多边形在 defs（y 负向），use 带偏移与 fill 样式
            href = e.get('{http://www.w3.org/1999/xlink}href') or ''
            mf = re.search(r'fill: (#[0-9a-f]{6}); fill-opacity', style)
            if href.startswith('#') and href[1:] in defs_paths and mf \
                    and any((a.get('id') or '').startswith('FillBetween')
                            for a in _ancestors(e, pm) if a is not None):
                ox, oy = float(e.get('x') or 0), float(e.get('y') or 0)
                pts = [(px + ox, py + oy)
                       for px, py in parse_path_points(defs_paths[href[1:]])]
                bands.setdefault(ax, []).append((mf.group(1), pts))

    def fit_map(pairs):
        """像素→数据线性拟合；pairs=[(px, val)]。"""
        import numpy as np
        px = np.array([p[0] for p in pairs])
        v = np.array([p[1] for p in pairs])
        a, b = np.polyfit(px, v, 1)
        return float(a), float(b)

    def to_val(fit, p):
        a, b = fit
        return a * p + b

    def to_val_x(tx, s):
        return (s - tx[1]) / tx[0]

    out = {'source_svg': svg_path.name,
           'note': 'extracted from published figure SVG geometry; '
                   'means from Line2D marker centers, CI from fill_between '
                   'paths, coordinates via linear tick fits'}

    for ax_name, panel in (('axes_2', 'degree'), ('axes_3', 'ig')):
        tx = fit_map(ticks[ax_name]['x'])
        ty = fit_map(ticks[ax_name]['y'])
        series = {}
        colors = ([TEAL, GRAY] if panel == 'degree' else [ORANGE])
        for color in colors:
            pts = sorted([(px, py) for c, px, py in markers.get(ax_name, [])
                          if c == color])
            # 图例中的示例标记不在整数步位上，剔除
            pts = [(px, py) for px, py in pts
                   if abs(to_val(tx, px) - round(to_val(tx, px))) < 0.05]
            xs = [round(to_val(tx, px), 4) for px, _ in pts]
            ys = [to_val(ty, py) for _, py in pts]
            assert all(abs(x - round(x)) < 0.05 for x in xs), xs
            series[color] = {'step': [int(round(x)) for x in xs],
                             'mean': [round(v, 4) for v in ys]}
        for color in colors:
            band = next((pts for c, pts in bands.get(ax_name, [])
                         if c == color), None)
            assert band is not None, (ax_name, color)
            by_x = {}
            for px, py in band:
                by_x.setdefault(round(px, 3), []).append(to_val(ty, py))
            steps = series[color]['step']
            lo, hi = [], []
            for s in steps:
                px = to_val_x(tx, s)
                vals = [v for k, vs in by_x.items() if abs(k - px) < 1.0
                        for v in vs]
                assert len(vals) >= 2, (s, len(vals))   # 起点重复+闭合点可致 3
                lo.append(round(min(vals), 4))
                hi.append(round(max(vals), 4))
            series[color]['ci_lo'] = lo
            series[color]['ci_hi'] = hi
        entry = {
            'step': series[colors[0]]['step'],
            'mean': series[colors[0]]['mean'],
            'ci_lo': series[colors[0]]['ci_lo'],
            'ci_hi': series[colors[0]]['ci_hi'],
        }
        if panel == 'degree':
            out['b_degree'] = {
                'chosen': entry,
                'available': {
                    'step': series[GRAY]['step'],
                    'mean': series[GRAY]['mean'],
                    'ci_lo': series[GRAY]['ci_lo'],
                    'ci_hi': series[GRAY]['ci_hi'],
                },
            }
        else:
            out['c_ig_rank'] = entry

    out_path = svg_path.with_name('choice_behavior_b10_from_svg.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 校验（README 已知锚点）----
    d, g = out['b_degree'], out['c_ig_rank']
    print('== 校验锚点（README：步2 被选1.98/可选1.27；IG 步2 0.573、≥11 0.452）==')
    i2 = d['chosen']['step'].index(2)
    print(f"degree step2: chosen={d['chosen']['mean'][i2]:.3f} "
          f"avail={d['available']['mean'][i2]:.3f}")
    print(f"ig step2={g['mean'][0]:.4f}  step>=11={g['mean'][-1]:.4f}")
    print('chosen:', d['chosen']['mean'])
    print('avail:  ', d['available']['mean'])
    print('ig:     ', g['mean'])
    print('saved', out_path)


if __name__ == '__main__':
    main()
