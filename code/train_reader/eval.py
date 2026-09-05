"""统一评估：reader checkpoint（base/attention/evidence）或固定策略（random/bfs）。

自 m5/base/analysis/eval.py → m6/attention/eval_heldout.py → m7/eval.py
三代收敛。reader 结构完全由 checkpoint 内嵌 config 重建（无需 CLI 传
模型参数）。

一次运行覆盖全部可用 held-out split（默认 valid+test；无验证集的数据集
如 cifar10 自动检测并跳过 valid），结果合并为**单个**统一 schema JSON
（comparison 脚本直接可用）：
  - 固定策略：--strategy random bfs → result/{ds}/fixed/eval_fixed.json
    （顶层 valid/test 各含 random/bfs 键）
  - reader：--checkpoint <pt> → <checkpoint 同目录>/eval_learnable.json
    （顶层 valid/test 各含 learnable 键）
config.full_graph_ref 由冻结分类器在各 split 全量图上现场评估填入
（plot_three_stage 的上界虚线不再需要 --ref-valid/--ref-test）。

用法:
  # 固定策略基线（一次跑 random+bfs × valid+test）→ result/{ds}/fixed/eval_fixed.json
  python code/train_reader/eval.py --dataset cifar10 --root D:\\data\\superpixels \
      --strategy random bfs --classifier <上界分类器>
  # reader 评估（valid+test 合并 → checkpoint 同目录 eval_learnable.json）
  python code/train_reader/eval.py --dataset cifar10 --root D:\\data\\superpixels \
      --checkpoint result/cifar10/experiments/exp1/attention/rl/checkpoint_epoch5.pt \
      --classifier result/cifar10/classifier/best_cifar10_classifier.pt
"""

import argparse
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from active_graph_reader.reader.fixed import create_fixed_reader  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (build_eval_json, evaluate_classifier_full,  # noqa: E402
                               evaluate_fixed, evaluate_reader, load_classifier_frozen,
                               load_reader_from_checkpoint, merge_eval_json)


def resolve(p) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def main():
    parser = argparse.ArgumentParser(description='统一评估（reader/固定策略）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--checkpoint', default='',
                        help='reader checkpoint；与 --strategy 二选一')
    parser.add_argument('--strategy', nargs='+', default=None,
                        choices=['random', 'bfs', 'max-frontier-degree',
                                 'random-walk', 'sagpool'],
                        help='固定策略名（可同时多个，默认用法 --strategy random bfs；'
                             'random-walk/sagpool 为 baseline/ 协议适配基线）')
    parser.add_argument('--strategy-ckpt', default='',
                        help='sagpool scorer checkpoint'
                             '（默认 baseline/artifacts/{ds}/sagpool.pt）')
    parser.add_argument('--classifier', required=True, help='上界分类器 checkpoint')
    parser.add_argument('--splits', nargs='+', default=['valid', 'test'],
                        choices=['train', 'valid', 'test'],
                        help='评估的 split（数据集缺少的 split 自动跳过并告警）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每个 split 的评估图数限制（0=全部；PPA 必配）')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='reader 评估时的图批量大小')
    parser.add_argument('--argmax', action=argparse.BooleanOptionalAction, default=True,
                        help='argmax 选择（默认开启；--no-argmax 改为按分布采样）')
    parser.add_argument('--n-seeds', type=int, default=4,
                        help='每图确定性 seed 数（默认 4，取平均）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认：reader → checkpoint 同目录；'
                             'strategy → result/{ds}/fixed）')
    args = parser.parse_args()

    assert bool(args.checkpoint) != bool(args.strategy), \
        "--checkpoint 与 --strategy 必须二选一"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root,
                     splits=sorted(set(args.splits)), max_graphs=args.max_graphs)
    spec = ds.spec
    budget_options = spec.budget_options

    # 自动检查数据集划分：无 valid（如 cifar10）时跳过并告警，不中断
    evaluated = []  # [(split, graphs, labels)]
    for split in args.splits:
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if len(graphs) == 0:
            print(f"[skip] 数据集 '{spec.name}' 无 {split} 划分（或为空），跳过")
            continue
        evaluated.append((split, graphs, labels))
    if not evaluated:
        raise SystemExit(f"数据集 '{spec.name}' 所请求 split {args.splits} 全部为空")

    classifier = load_classifier_frozen(resolve(args.classifier), device)
    print(f"Classifier: {args.classifier}")
    builder = SubgraphBuilder()

    is_fixed = bool(args.strategy)
    strategies = list(args.strategy) if is_fixed else ['learnable']
    agents = []
    if is_fixed:
        print(f"Strategies: {' '.join(args.strategy)}")
        if any(s in ('random-walk', 'sagpool') for s in args.strategy):
            # baseline/ 位于仓库根：注册协议适配基线策略
            import sys as _sys
            from tools.paths import REPO_ROOT
            if str(REPO_ROOT) not in _sys.path:
                _sys.path.insert(0, str(REPO_ROOT))
            from baseline.strategies import register_baseline_strategies
            register_baseline_strategies()
        for strategy in args.strategy:
            if strategy == 'sagpool':
                from baseline.sagpool import SagpoolStrategy
                ckpt = (resolve(args.strategy_ckpt) if args.strategy_ckpt
                        else CODE_DIR.parent / 'baseline' / 'artifacts'
                             / spec.name / 'sagpool.pt')
                if not ckpt.exists():
                    raise SystemExit(
                        f"sagpool scorer checkpoint 不存在: {ckpt}\n"
                        f"先训练: python baseline/run_sagpool.py "
                        f"--dataset {spec.name} --root <root>")
                agent = SagpoolStrategy()
                agent.load(str(ckpt), device)
                print(f"SAGPool scorer: {ckpt}")
            else:
                agent = create_fixed_reader(strategy)
            agents.append(agent)
    else:
        ckpt_path = resolve(args.checkpoint)
        print(f"Checkpoint: {ckpt_path}")
        reader = load_reader_from_checkpoint(ckpt_path, device,
                                             stochastic=False, temperature=1.0)
        print(f"Reader: kind={reader.config.kind} "
              f"hidden={reader.config.encoder_hidden_dims} "
              f"m={reader.config.m} n_heads={reader.config.n_heads} "
              f"(from checkpoint config)")

    parts, full_graph_ref, n_graphs = [], {}, {}

    def _warn_conflict(path):
        print(f"[warn] 合并冲突：{path}（后者覆盖前者）")

    for split, graphs, labels in evaluated:
        ref = evaluate_classifier_full(classifier, graphs, labels, device)
        full_graph_ref[split] = round(ref, 6)
        n_graphs[split] = len(graphs)
        print(f"[{split}] full-graph 参照 acc = {ref:.4f}（{len(graphs)} 图）")

        for strategy, agent in zip(strategies, agents):
            if is_fixed:
                results = evaluate_fixed(None, classifier, builder,
                                         graphs, labels, budget_options, device,
                                         args.argmax, args.n_seeds, agent=agent)
            else:
                results = evaluate_reader(reader, classifier, builder,
                                          graphs, labels, budget_options, device,
                                          args.argmax, chunk_size=args.chunk_size,
                                          n_seeds=args.n_seeds)
            parts.append(build_eval_json(results, split, strategy,
                                         args.n_seeds, budget_options, config={}))
            print(f"[{split}] {strategy}: "
                  + '  '.join(f"B{int(b * 100)}%={results[f'acc_{int(b * 100)}%']:.4f}"
                              for b in budget_options)
                  + f"  overall={results['acc_overall']:.4f}")
            if 'acc_clf_overall' in results:
                print(f"[{split}] {strategy} [ref]: "
                      + '  '.join(f"B{int(b * 100)}%={results[f'acc_clf_{int(b * 100)}%']:.4f}"
                                  for b in budget_options)
                      + f"  overall={results['acc_clf_overall']:.4f}"
                      + "（evidence 终判 = 自身证据头；ref = 冻结上界分类器）")

    json_out = merge_eval_json(parts, on_conflict=_warn_conflict)
    config = {
        'dataset': spec.name,
        'mode': 'fixed' if is_fixed else 'reader',
        'strategies': strategies,
        'strategy_or_checkpoint': ' '.join(args.strategy) if is_fixed else str(ckpt_path),
        'classifier': str(resolve(args.classifier)),
        'budgets': list(budget_options),
        'n_graphs': n_graphs,
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'seed_method': SEED_METHOD_STR,
        'full_graph_ref': full_graph_ref,
    }
    if not is_fixed:
        config['checkpoint'] = str(ckpt_path)
    json_out['config'] = config

    if args.output_dir:
        out_dir = resolve(args.output_dir)
    elif is_fixed:
        out_dir = CODE_DIR / 'result' / spec.name / 'fixed'
    else:
        out_dir = ckpt_path.parent
    out_path = out_dir / ('eval_fixed.json' if is_fixed else 'eval_learnable.json')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(json_out, out_path)
    print(f"JSON saved to {out_path}")
    print(f"绘图: python code/comparison/plot_eval_results.py --json {out_path}")


if __name__ == '__main__':
    main()
