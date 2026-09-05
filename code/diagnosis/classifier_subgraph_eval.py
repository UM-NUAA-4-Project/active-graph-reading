"""分类器子图鲁棒性诊断：全图 vs 揭示子图准确率。

背景：上界分类器同时承担两个子图侧角色——RL 奖励来源（对最终揭示子图
打分）与 evidence 评估的参照（acc_clf_*）。VN 系分类器因虚拟节点聚合
全局统计，在揭示子图上严重退化（2026-09 目标：GIN+ 需两侧都不退化）。

本脚本对任意 classifier checkpoint 统一测量：
  1. acc_full：完整测试图准确率（上界）；
  2. 各固定策略（bfs/random/max_degree）按协议种子 + 预算比例揭示子图
     后的准确率（与 evaluate_fixed 同口径，结果可直接对比 eval JSON）。

用法:
  python code/diagnosis/classifier_subgraph_eval.py \
      --checkpoint code/result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt \
      --dataset cifar10 --root D:\\data\\superpixels \
      --strategies bfs random
输出: <checkpoint 同目录>/subgraph_eval.json
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from classifier.factory import load_classifier  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from train_classifier.train_full_graph import evaluate as evaluate_full  # noqa: E402
from train_reader.core.eval_loop import evaluate_fixed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='分类器全图/子图鲁棒性诊断')
    parser.add_argument('--checkpoint', required=True,
                        help='classifier checkpoint（需内嵌 config）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='')
    parser.add_argument('--max-graphs', type=int, default=0)
    parser.add_argument('--strategies', nargs='+',
                        default=['bfs', 'random'],
                        choices=['bfs', 'random', 'max-frontier-degree'])
    parser.add_argument('--n-seeds', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='仅全图评估用（子图评估内部固定 512 分块）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='',
                        help='默认 <checkpoint 同目录>/subgraph_eval.json')
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt_path = Path(args.checkpoint)
    try:
        clf = load_classifier(str(ckpt_path), device=str(device), freeze=True)
    except (ValueError, KeyError) as e:
        raise SystemExit(f"加载失败：{e}\n旧 checkpoint 请先运行 "
                         "code/tools/migrate_legacy_classifier_ckpt.py 补写 config")
    print(f"Classifier: {ckpt_path} → readout={clf.config.readout} "
          f"residual={clf.config.residual} vn={clf.config.virtual_node} "
          f"head={clf.config.head} norm={clf.config.gnn_norm or 'gin-default(batchnorm)'}")

    ds = get_dataset(args.dataset, root=args.root, splits=['test'],
                     max_graphs=args.max_graphs)
    spec = ds.spec
    graphs, labels = ds.test_graphs, ds.test_labels
    print(f"Test graphs: {len(graphs)} | budgets: {list(spec.budget_options)}")

    out = {
        'checkpoint': str(ckpt_path),
        'dataset': spec.name,
        'n_graphs': len(graphs),
        'n_seeds': args.n_seeds,
        'budget_options': list(spec.budget_options),
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }

    print("Evaluating full graphs ...")
    out['acc_full'] = evaluate_full(clf, graphs, labels, args.batch_size, device)
    print(f"  acc_full = {out['acc_full']:.4f}")

    builder = SubgraphBuilder()
    for strategy in args.strategies:
        print(f"Rolling fixed strategy: {strategy} ...")
        res = evaluate_fixed(strategy, clf, builder, graphs, labels,
                             list(spec.budget_options), device,
                             n_seeds=args.n_seeds)
        # 精简落盘：去掉大体量逐样本明细（per_class 与逐预算 outcomes）
        res.pop('per_seed_per_class', None)
        res.pop('outcomes', None)
        out[strategy] = res
        accs = {k: v for k, v in res.items() if k.startswith('acc_')}
        print(f"  {strategy}: " + "  ".join(f"{k}={v:.4f}" for k, v in accs.items()))

    out_path = Path(args.output) if args.output else ckpt_path.parent / 'subgraph_eval.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
