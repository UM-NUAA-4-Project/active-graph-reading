"""合并多份统一格式 eval JSON（按策略并集）→ 单份完整 JSON。

动机：固定策略基线分多批评估时会产生多份 eval_fixed.json（如 cifar10 的
fixed/ 含 random-walk+sagpool、fixed_1/ 含 random+bfs+max-frontier-degree，
策略不相交），下游绘图（plot_eval_results / plot_three_stage）需要单一
完整基线 JSON。本工具按"策略并集"合并：同一路径两份都有且数值不同 →
报错拒绝（宁可失败也不静默覆盖）；数值一致 → 幂等合并。

合并语义：
- 非 config 顶层键（split 表 / per_class_*）：递归深合并（策略 → seed →
  预算 → 叶子），冲突即报错；
- config：以第一份为基底；strategies 取按首次出现顺序的并集；
  full_graph_ref 逐 split 取先出现的非空值（不同值打印警告）；
  n_graphs / n_seeds / budgets 不一致仅警告（budgets 取并集）；
  附加 merged_from / merged_at；
- 默认 dry-run 只打印合并报告；--write 落盘。输出默认第一份输入路径
  （原地补全，其余输入文件保留作备份）；--output 可另指路径。
  策略在图中出现的顺序由输入文件顺序决定（先出现靠前）。

用法:
  # cifar10：fixed_1(random/bfs/max-frontier-degree) 在前，fixed(random-walk/sagpool) 在后
  python code/tools/merge_eval_json.py \
      --json result/cifar10/fixed_1/eval_fixed.json \
             result/cifar10/fixed/eval_fixed.json --write
"""

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from tools.config import save_json  # noqa: E402
from tools.paths import resolve  # noqa: E402


def deep_merge(dst, src, path=''):
    """把 src 递归并入 dst（就地）；返回冲突列表 [(路径, dst值, src值)]。"""
    conflicts = []
    for k, v in src.items():
        p = f'{path}.{k}' if path else str(k)
        if k not in dst:
            dst[k] = copy.deepcopy(v)
        elif isinstance(dst[k], dict) and isinstance(v, dict):
            conflicts += deep_merge(dst[k], v, p)
        elif dst[k] != v:
            conflicts.append((p, dst[k], v))
    return conflicts


def merge_evals(datas):
    """datas: [dict]（按输入顺序）→ (合并结果 dict, 冲突列表)。"""
    merged = copy.deepcopy(datas[0])
    conflicts = []
    for data in datas[1:]:
        for key, block in data.items():
            if key == 'config':
                continue
            if key not in merged:
                merged[key] = copy.deepcopy(block)
            elif isinstance(merged[key], dict) and isinstance(block, dict):
                conflicts += deep_merge(merged[key], block, key)
            elif merged[key] != block:
                conflicts.append((key, merged[key], block))
    return merged, conflicts


def merge_configs(cfgs, paths):
    """以第一份 config 为基底合并元信息。"""
    cfg = copy.deepcopy(cfgs[0])

    strategies = []
    for c in cfgs:
        for s in c.get('strategies') or []:
            if s not in strategies:
                strategies.append(s)
    if strategies:
        cfg['strategies'] = strategies
        cfg['strategy_or_checkpoint'] = ' '.join(strategies)

    ref = dict(cfg.get('full_graph_ref') or {})
    for c in cfgs[1:]:
        for sp, v in (c.get('full_graph_ref') or {}).items():
            if v is None:
                continue
            if ref.get(sp) is None:
                ref[sp] = v
            elif ref[sp] != v:
                print(f"[warn] full_graph_ref[{sp}] 不一致: "
                      f"{ref[sp]} vs {v}（取先者 {ref[sp]}）")
    if ref:
        cfg['full_graph_ref'] = ref

    for k in ('n_graphs', 'n_seeds'):
        vals = [c.get(k) for c in cfgs if c.get(k) is not None]
        if len(set(map(str, vals))) > 1:
            print(f"[warn] config.{k} 不一致: {vals}（取先者 {vals[0]}）")
    bsets = [tuple(sorted(map(float, c.get('budgets') or ())))
             for c in cfgs if c.get('budgets')]
    if len(set(bsets)) > 1:
        print(f"[warn] config.budgets 不一致: {bsets}（取并集）")
    union_b = sorted({b for bs in bsets for b in bs})
    if bsets and union_b != sorted(map(float, cfg.get('budgets') or ())):
        cfg['budgets'] = union_b

    cfg['merged_from'] = [str(p) for p in paths]
    cfg['merged_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    return cfg


def summarize(tag, data):
    splits = [k for k in data if k in ('train', 'valid', 'test')]
    for split in splits:
        strats = list(data[split].keys())
        print(f"  [{tag}] {split}: {strats}")
    pcs = [k for k in data if k.startswith('per_class_')]
    if pcs:
        print(f"  [{tag}] per_class: {pcs}")


def main():
    parser = argparse.ArgumentParser(
        description='合并多份统一格式 eval JSON（策略并集，dry-run 默认）')
    parser.add_argument('--json', nargs='+', required=True,
                        help='两份以上 eval JSON，顺序 = 策略出现顺序')
    parser.add_argument('--output', default='',
                        help='输出路径（默认第一份输入路径，原地补全）')
    parser.add_argument('--write', action='store_true',
                        help='实际落盘（默认 dry-run 只打印报告）')
    args = parser.parse_args()

    paths = [resolve(p) for p in args.json]
    datas = []
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            datas.append(json.load(f))

    print(f'== 输入（{len(paths)} 份）==')
    for p, d in zip(paths, datas):
        summarize(str(p), d)

    merged, conflicts = merge_evals(datas)
    merged['config'] = merge_configs([d.get('config') or {} for d in datas],
                                     paths)

    print('== 合并结果 ==')
    summarize('merged', merged)
    if conflicts:
        print(f'== 冲突（{len(conflicts)} 处，拒绝合并）==')
        for p, a, b in conflicts:
            print(f'  @{p}: {a} != {b}')
        sys.exit(1)

    out = resolve(args.output) if args.output else paths[0]
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        save_json(merged, out)
        print(f'--write: 已保存到 {out}')
    else:
        print(f'[dry-run] 未落盘；确认无误后加 --write 保存到 {out}')


if __name__ == '__main__':
    main()
