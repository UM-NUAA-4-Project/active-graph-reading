"""eval 结果 JSON 格式迁移：历史单 split/单策略文件 → 统一多 split/多策略 JSON。

2026-09-05 起 eval.py 一次运行产出统一 schema（顶层 valid/test → strategy →
s0..sN/mean → 预算 → {mean,std,n}，另含 per_class_{split} 与 config）：
  - 固定策略 → result/{ds}/fixed/eval_fixed.json     （valid/test × random/bfs）
  - reader   → <checkpoint 同目录>/eval_learnable.json（valid/test × learnable）

本脚本把历史三代 eval JSON 合并/重命名为统一格式（不虚构缺失数据，
缺 valid 就缺）：
  - m7 单文件版  eval_heldout_{split}_{strategy|reader}.json
  - m6 evaluate 版  .../evaluate/eval_heldout_{split}.json（按内容分流策略）
  - 旧命名 eval_heldout_{split}.json（fixed 目录下，含 random/bfs）
  - m5 canonical 版（analysis/eval_learnable.json、fixed_eval/eval_fixed.json）
    → 已是统一格式，仅报告，保留原位

用法（默认 dry-run，只打印迁移报告）：
  python code/tools/migrate_eval_json.py
  python code/tools/migrate_eval_json.py --write             # 落盘合并结果
  python code/tools/migrate_eval_json.py --write --archive   # 被合并旧文件移入 _legacy_eval/
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.config import load_json, save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from train_reader.core import merge_eval_json  # noqa: E402

SPLITS = ('train', 'valid', 'test')
FIXED_STRATEGIES = {'random', 'bfs'}
CANONICAL_NAMES = {'eval_fixed.json', 'eval_learnable.json'}


def iter_eval_json(root: Path):
    """result/ 下所有 eval 产物 JSON（排除 classifier history/报告类）。"""
    if not root.exists():
        return []
    out = []
    for pattern in ('**/eval_heldout_*.json', '**/eval_fixed.json',
                    '**/eval_learnable.json'):
        for path in sorted(root.glob(pattern)):
            if (path.is_file() and '_legacy_eval' not in path.parts
                    and path not in out):
                out.append(path)
    return out


def parse_strategies(data: dict) -> dict:
    """JSON → {split: {strategy: cells}}（只认 {split: {strategy: {s*, mean}}} 结构）。"""
    out = {}
    for split in SPLITS:
        node = data.get(split)
        if not isinstance(node, dict):
            continue
        for strategy, sub in node.items():
            if isinstance(sub, dict) and 'mean' in sub:
                out.setdefault(split, {})[strategy] = sub
    return out


def dataset_of(path: Path, result_root: Path) -> str:
    try:
        return path.relative_to(result_root).parts[0]
    except ValueError:
        return path.parent.parent.name


def reader_target_dir(path: Path) -> Path:
    """m6 evaluate/ 子目录下的文件归并到其父目录（checkpoint 同目录）。"""
    return path.parent.parent if path.parent.name == 'evaluate' else path.parent


def classify(path: Path, data: dict, result_root: Path):
    """→ (kind, target, strategies_found)；kind ∈ fixed|reader|canonical|unknown。"""
    strategies = sorted({s for per_split in parse_strategies(data).values()
                         for s in per_split})
    if not strategies:
        return 'unknown', None, []
    strat_set = set(strategies)
    if strat_set <= FIXED_STRATEGIES:
        kind, target = 'fixed', result_root / dataset_of(path, result_root) / 'fixed' / 'eval_fixed.json'
    elif 'learnable' in strat_set:
        kind, target = 'reader', reader_target_dir(path) / 'eval_learnable.json'
    else:
        return 'unknown', None, strategies
    if path.name in CANONICAL_NAMES:
        # 统一命名文件：位于规范目标位置 → 合并基底；否则视为历史 canonical 产物
        return ('anchor' if path == target else 'canonical'), target, strategies
    return kind, target, strategies


def budget_sets(data: dict) -> list:
    """报告用：每个 split.strategy 的预算档键集合（去重后列表）。"""
    sets = set()
    for per_split in parse_strategies(data).values():
        for cells in per_split.values():
            if 'mean' in cells:
                sets.add(tuple(sorted(cells['mean'].keys())))
    return sorted('/'.join(s) for s in sets)


def merge_group(target: Path, sources: list):
    """sources=[(mtime, path, data)] 按 mtime 升序合并；返回 (merged, conflicts)。"""
    conflicts = []

    def on_conflict(where):
        conflicts.append(where)

    merged = merge_eval_json([data for _m, _p, data in sources],
                             on_conflict=on_conflict)
    strategies = sorted({s for per_split in parse_strategies(merged).values()
                         for s in per_split})
    config = dict(merged.get('config') or {})
    config['strategies'] = strategies
    config.setdefault('mode', 'fixed' if set(strategies) <= FIXED_STRATEGIES
                      else 'reader')
    merged['config'] = config
    return merged, conflicts


def _containing_root(path: Path, roots: list) -> Path:
    for root in roots:
        try:
            path.relative_to(root)
            return root
        except ValueError:
            continue
    return path.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('roots', nargs='*', default=[],
                        help='扫描根目录（默认 result/）')
    parser.add_argument('--write', action='store_true',
                        help='落盘合并结果（默认 dry-run 只打印报告）')
    parser.add_argument('--archive', action='store_true',
                        help='被合并的旧文件移入同目录 _legacy_eval/（需 --write）')
    args = parser.parse_args()

    roots = [Path(r) for r in args.roots] or [CODE_DIR / 'result']
    paths = [p for root in roots for p in iter_eval_json(root)]
    if not paths:
        print('未发现 eval 结果 JSON')
        return

    groups = {}      # target -> [(mtime, path, data)]
    kept, unknown = [], []
    for path in paths:
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001
            unknown.append((path, f'读取失败: {exc}'))
            continue
        kind, target, _strategies = classify(path, data,
                                             _containing_root(path, roots))
        if kind == 'unknown':
            unknown.append((path, '无法识别 split/strategy 结构'))
        elif kind == 'canonical':
            kept.append(path)
        else:
            groups.setdefault(target, []).append((path.stat().st_mtime, path, data))

    action = '计划（dry-run，加 --write 落盘）' if not args.write else '结果'
    print(f'=== eval JSON 迁移{action} ===\n')

    n_write, n_archive = 0, 0
    target_budgets = {}   # target -> 预算档集合（合并后），供数据集级提示复用
    for target in sorted(groups):
        sources = sorted(groups[target], key=lambda item: item[0])
        merged, conflicts = merge_group(target, sources)
        target_budgets[target] = frozenset(budget_sets(merged))
        exists = target.exists()
        print(f'[{"merge→existing" if exists else "merge→new"}] {target}')
        for _mtime, path, _data in sources:
            marks = '/'.join(f'{split}.{s}'
                             for split, per in parse_strategies(_data).items()
                             for s in per)
            print(f'    ← {path}   ({marks})')
        print(f'    合并后 strategies={merged["config"]["strategies"]}'
              f'  预算档={budget_sets(merged) or "?"}'
              + (f'  ⚠ 冲突覆盖: {", ".join(conflicts)}' if conflicts else ''))
        if len(set(budget_sets(merged))) > 1:
            print('    ⚠ 组内预算档不一致（见上），绘图时 --budgets 需取交集')
        if args.write:
            save_json(merged, target)
            n_write += 1
            if args.archive:
                for _mtime, path, _data in sources:
                    if path == target:
                        continue
                    dst_dir = path.parent / '_legacy_eval'
                    dst_dir.mkdir(exist_ok=True)
                    shutil.move(str(path), str(dst_dir / path.name))
                    n_archive += 1
        print()

    if kept:
        print('[keep] 已是统一格式的历史产物（保留原位）:')
        for path in kept:
            print(f'    {path}')
        print()
    if unknown:
        print('[skip] 未识别/读取失败的文件:')
        for path, why in unknown:
            print(f'    {path}  ({why})')
        print()

    # 数据集级提示：fixed 与 reader 目标预算档不同时，三阶段图 --budgets 取交集
    if len(roots) == 1:
        by_ds = {}
        for target, bsets in target_budgets.items():
            by_ds.setdefault(dataset_of(target, roots[0]), []).append(
                (target.name, bsets))
        for ds_name, entries in sorted(by_ds.items()):
            fixed_sets = {s for name, s in entries if name == 'eval_fixed.json'}
            reader_sets = {s for name, s in entries if name == 'eval_learnable.json'}
            if fixed_sets and reader_sets and fixed_sets != reader_sets:
                print(f'[note] {ds_name}: fixed 预算档={sorted(fixed_sets)} 与 '
                      f'reader 预算档={sorted(reader_sets)} 不一致，'
                      f'plot_three_stage --budgets 请取交集')

    print(f'目标组数: {len(groups)}；'
          f'{"已写盘" if args.write else "dry-run 未写盘"}: {n_write}；'
          f'归档旧文件: {n_archive if args.write else 0}')
    if not args.write:
        print('确认无误后加 --write 落盘（可再加 --archive 归档旧文件）')


if __name__ == '__main__':
    main()
