"""eval 脚本共用管线（eval_reader / eval_baselines 两入口共享）。

自 train_reader/eval.py 拆分（2026-09-19）：reader 与 baseline 两类评估
的 CLI 参数、JSON config 与输出路径各自独立，本模块只保留共用管线——

- add_common_args: 两类评估共有的 CLI 参数；
- run_split_eval: 逐 split 按需加载评估（先加载并评估一个 split，释放
  内存后再加载下一个；JSON 每个 split 评估完增量落盘，后续 split 崩溃
  不丢已完成结果）。

eval_split 回调约定：eval_split(split, graphs, labels) ->
Iterable[(strategy_key, results)]，strategy_key 落入 JSON 的
{split: {strategy_key: {...}}} 子树（reader 固定为 'learnable'）。
config 约定：须含 'n_graphs'/'full_graph_ref' 两个 dict 字段（按 split
累积），其余字段（mode/strategies/classifier/budgets/...）由各入口自定。
"""

import argparse
import gc

import torch

from datasets import get_dataset, release_dataset
from tools.config import save_json
from .eval_loop import build_eval_json, evaluate_classifier_full, merge_eval_json


def resolve(p):
    from tools.paths import resolve as _resolve
    return _resolve(p)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """两类 eval 脚本共有的 CLI 参数（--checkpoint/--strategy 各自添加）。"""
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--classifier', required=True, help='上界分类器 checkpoint')
    parser.add_argument('--splits', nargs='+', default=['valid', 'test'],
                        choices=['train', 'valid', 'test'],
                        help='评估的 split（数据集缺少的 split 自动跳过并告警）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每个 split 的评估图数限制（0=全部；PPA 必配）')
    parser.add_argument('--stratified', action='store_true',
                        help='--max-graphs 按类别比例分层抽样（默认前 N 截断；'
                             '与训练侧同种子同选择，保证子集一致）')
    parser.add_argument('--argmax', action=argparse.BooleanOptionalAction, default=True,
                        help='argmax 选择（默认开启；--no-argmax 改为按分布采样）')
    parser.add_argument('--n-seeds', type=int, default=4,
                        help='每图确定性 seed 数（默认 4，取平均）')
    parser.add_argument('--ref-batch-size', type=int, default=256,
                        help='full-graph 参照评估的批量（大图数据集如 collab 在'
                             '显存有限的 GPU 上 OOM 时调小，如 32）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认：reader → checkpoint 同目录；'
                             'baseline → result/{ds}/fixed）')


_CALIBRATION_NOTE = ('[[max_prob(4位舍入), correct], '
                     '...] 逐 (图, seed) 样本')


def run_split_eval(args, classifier, budget_options, config, out_path,
                   eval_split, calibration=None):
    """逐 split 按需加载评估 + 增量落盘（eval_reader/eval_baselines 共用）。

    - 每个 split：加载图 → 冻结分类器全图参照（写入
      config.full_graph_ref[split]）→ eval_split 回调逐策略评估 →
      JSON 合并落盘 → 释放该 split 图数据（缓存摘除 + gc + empty_cache）；
    - calibration 不为 None 时逐 split 落盘 out_path.parent/calibration.json
      （reader --export-confidence 校准导出）。
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    parts = []
    evaluated = []

    def _warn_conflict(path):
        print(f"[warn] 合并冲突：{path}（后者覆盖前者）")

    for split in list(dict.fromkeys(args.splits)):
        ds = get_dataset(args.dataset, root=args.root, splits=(split,),
                         max_graphs=args.max_graphs, stratified=args.stratified)
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if len(graphs) == 0:
            print(f"[skip] 数据集 '{args.dataset}' 无 {split} 划分（或为空），跳过")
            release_dataset(ds)
            continue

        ref = evaluate_classifier_full(classifier, graphs, labels, device,
                                       batch_size=args.ref_batch_size)
        config['full_graph_ref'][split] = round(ref, 6)
        config['n_graphs'][split] = len(graphs)
        print(f"[{split}] 已加载 {len(graphs)} 图；"
              f"full-graph 参照 acc = {ref:.4f}")

        for key, results in eval_split(split, graphs, labels):
            parts.append(build_eval_json(results, split, key,
                                         args.n_seeds, budget_options, config={}))
            print(f"[{split}] {key}: "
                  + '  '.join(f"B{int(b * 100)}%={results[f'acc_{int(b * 100)}%']:.4f}"
                              for b in budget_options)
                  + f"  overall={results['acc_overall']:.4f}")
            if 'acc_clf_overall' in results:
                print(f"[{split}] {key} [ref]: "
                      + '  '.join(f"B{int(b * 100)}%={results[f'acc_clf_{int(b * 100)}%']:.4f}"
                                  for b in budget_options)
                      + f"  overall={results['acc_clf_overall']:.4f}"
                      + "（evidence 终判 = 自身证据头；ref = 冻结上界分类器）")

        evaluated.append(split)
        # 增量落盘：后续 split 失败/崩溃不丢已完成结果
        json_out = merge_eval_json(parts, on_conflict=_warn_conflict)
        json_out['config'] = config
        save_json(json_out, out_path)
        print(f"[{split}] JSON saved to {out_path}")
        if calibration:
            save_json({'calibration': calibration,
                       'config': {'note': _CALIBRATION_NOTE}},
                      out_path.parent / 'calibration.json')

        # 释放本 split 的图数据：缓存摘除 + gc；GPU 侧顺带清空分配缓存
        del graphs, labels
        release_dataset(ds)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not evaluated:
        raise SystemExit(f"数据集 '{args.dataset}' 所请求 split {args.splits} 全部为空")
