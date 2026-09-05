"""统一评估：reader checkpoint（base/attention）或固定策略（random/bfs）。

自 m5/base/analysis/eval.py → m6/attention/eval_heldout.py → m7/eval.py
三代收敛。reader 结构完全由 checkpoint 内嵌 config 重建（无需 CLI 传
模型参数）；输出 JSON schema 与旧版一致（comparison 脚本直接可用）。

用法:
  # 固定策略基线 → result/{ds}/fixed/
  python code/train_reader/eval.py --dataset cifar10 --root D:\\superpixels \
      --strategy random --classifier <上界分类器>
  # reader 评估（默认写入 checkpoint 同目录）
  python code/train_reader/eval.py --dataset cifar10 --root D:\\superpixels \
      --checkpoint result/cifar10/experiments/exp1/attention/rl/checkpoint_epoch5.pt \
      --classifier result/cifar10/classifier/best_cifar10_classifier.pt
    python code/train_reader/eval.py --dataset cifar10 --root D:\\superpixels 
    --checkpoint result\cifar10\experiments\evidence\evidence\bc\reader_checkpoint_epoch2.pt 
    --classifier result\cifar10\classifier\finetune\best_cifar10_classifier.pt
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
from datasets import get_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (build_eval_json, evaluate_fixed,  # noqa: E402
                               evaluate_reader, load_classifier_frozen,
                               load_reader_from_checkpoint)


def resolve(p) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def main():
    parser = argparse.ArgumentParser(description='统一评估（reader/固定策略）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--checkpoint', default='',
                        help='reader checkpoint；与 --strategy 二选一')
    parser.add_argument('--strategy', default='', choices=['random', 'bfs'],
                        help='固定策略名')
    parser.add_argument('--classifier', required=True, help='上界分类器 checkpoint')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='评估图数限制（0=全部；PPA 必配）')
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

    ds = get_dataset(args.dataset, root=args.root, splits=[args.split],
                     max_graphs=args.max_graphs)
    spec = ds.spec
    budget_options = spec.budget_options

    graphs = getattr(ds, f'{args.split}_graphs')
    labels = getattr(ds, f'{args.split}_labels')
    if len(graphs) == 0:
        raise SystemExit(f"数据集 '{spec.name}' 无 {args.split} split"
                         f"（cifar10 无 valid，请用 --split test）")

    classifier = load_classifier_frozen(resolve(args.classifier), device)
    print(f"Classifier: {args.classifier}")
    builder = SubgraphBuilder()

    if args.strategy:
        print(f"Strategy: {args.strategy}")
        results = evaluate_fixed(args.strategy, classifier, builder,
                                 graphs, labels, budget_options, device,
                                 args.argmax, args.n_seeds)
        key = args.strategy
        out_dir = (resolve(args.output_dir) if args.output_dir else
                   CODE_DIR / 'result' / spec.name / 'fixed')
        ckpt_path = None
    else:
        ckpt_path = resolve(args.checkpoint)
        print(f"Checkpoint: {ckpt_path}")
        reader = load_reader_from_checkpoint(ckpt_path, device,
                                             stochastic=False, temperature=1.0)
        print(f"Reader: kind={reader.config.kind} "
              f"hidden={reader.config.encoder_hidden_dims} "
              f"m={reader.config.m} n_heads={reader.config.n_heads} "
              f"(from checkpoint config)")
        results = evaluate_reader(reader, classifier, builder,
                                  graphs, labels, budget_options, device,
                                  args.argmax,chunk_size=args.chunk_size, n_seeds=args.n_seeds)
        key = 'learnable'
        out_dir = (resolve(args.output_dir) if args.output_dir
                   else ckpt_path.parent)

    for b in budget_options:
        print(f"  Budget {int(b * 100):2d}%: {results[f'acc_{int(b * 100)}%']:.4f}")
    print(f"  Overall: {results['acc_overall']:.4f}")
    if 'acc_clf_overall' in results:
        print("  (evidence 终判 = 自身证据头；参照 = 冻结上界分类器)")
        for b in budget_options:
            print(f"  [ref] Budget {int(b * 100):2d}%: "
                  f"{results[f'acc_clf_{int(b * 100)}%']:.4f}")
        print(f"  [ref] Overall: {results['acc_clf_overall']:.4f}")

    json_out = build_eval_json(
        results, args.split, key, args.n_seeds, budget_options,
        config={
            'dataset': spec.name,
            'strategy_or_checkpoint': args.strategy or str(ckpt_path),
            'classifier': str(resolve(args.classifier)),
            'budgets': list(budget_options),
            'n_graphs': results['n_graphs'],
            'n_seeds': args.n_seeds,
            'argmax': args.argmax,
            'seed_method': SEED_METHOD_STR,
            'full_graph_ref': {'test': None},
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'eval_heldout_{args.split}_{args.strategy or "reader"}.json'
    save_json(json_out, out_path)
    print(f"JSON saved to {out_path}")
    print(f"绘图: python code/comparison/plot_eval_results.py --json {out_path}")


if __name__ == '__main__':
    main()
