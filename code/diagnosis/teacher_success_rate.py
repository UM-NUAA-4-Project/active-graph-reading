"""教师轨迹成功率统计：各 teacher 在各预算档下，教师采出的最终子图被
上界分类器判对的比例（与轨迹生成的 is_correct 完全同口径）。

复用 TrajectoryGenerator 的生成+验证通路，只统计不落盘轨迹。
输出：
  - result/{ds}/teacher_success/teacher_success.json
  - stdout 打印 Markdown 表（对齐 Summary of results 的 teacher 成功概率表）

用法（从仓库根）:
  python code/diagnosis/teacher_success_rate.py --dataset cifar10 \
      --root D:\\data\\superpixels \
      --classifier result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt \
      --teachers gradient info-gain hybrid-a0.7 --max-graphs 500
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
from datasets import get_dataset  # noqa: E402
from tools.config import require_args, save_json  # noqa: E402
from tools.paths import CODE_DIR, resolve  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from trajectories.core.generator import TrajectoryGenerator  # noqa: E402
from trajectories.core.rollout import build_classifier  # noqa: E402
from trajectories.teachers import build_teacher  # noqa: E402


class _GraphList:
    """dataset[i] -> (graph, label) 的最小包装（同 generate_trajectories）。"""

    def __init__(self, graphs, labels):
        self._graphs = graphs
        self._labels = labels

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, i):
        return self._graphs[i], self._labels[i]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='教师轨迹成功率统计')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='', help='数据集根目录（必填）')
    parser.add_argument('--classifier', default='',
                        help='上界分类器 checkpoint（必填；成功 = 判对）')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'info-gain', 'hybrid-a0.7'],
                        help='教师名（build_teacher 工厂名）')
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=None,
                        help='预算档（默认 spec.budget_options）')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='每图每教师的 seed 数（与轨迹生成同默认）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 最多加载图数（0=不限；PPA 必配）')
    parser.add_argument('--stratified', action='store_true',
                        help='--max-graphs 按类别比例分层抽样')
    parser.add_argument('--graph-batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='',
                        help='覆盖默认输出目录 result/{ds}/teacher_success')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    require_args(args, ['root', 'classifier'])

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs,
                     stratified=args.stratified and args.max_graphs > 0)
    spec = ds.spec
    ratios = [float(r) for r in
              (args.budget_ratios or list(spec.budget_options))]

    graphs = ds.train_graphs
    labels = ds.train_labels
    n_graphs = len(graphs)
    print(f"Train graphs: {n_graphs} | ratios: {ratios} | "
          f"teachers: {args.teachers}")

    clf_path = resolve(args.classifier)
    cls_ckpt = torch.load(str(clf_path), map_location=device, weights_only=False)
    classifier = build_classifier(cls_ckpt, device, requires_grad=True)

    builder = SubgraphBuilder()
    teachers = [build_teacher(name, classifier=classifier,
                              subgraph_builder=builder, device=str(device))
                for name in args.teachers]

    generator = TrajectoryGenerator(
        teachers=teachers, classifier=classifier, subgraph_builder=builder,
        device=str(device), graph_batch_size=args.graph_batch_size)

    dataset = _GraphList(graphs, labels)
    n_nodes = [g.num_nodes() for g in graphs]

    stats = {}   # ratio -> teacher -> {correct, total, rate, sem}
    for ratio in ratios:
        print(f"\n=== reveal_ratio={ratio} ===")
        # 与 generate_all 相同的有效图条件（budget < n_nodes）
        n_used = sum(1 for n in n_nodes
                     if n > 0 and max(1, int(ratio * n)) < n)
        result = generator.generate_all(dataset, list(range(n_graphs)),
                                        ratio, n_seeds=args.n_seeds)
        stats[str(ratio)] = {}
        for t in teachers:
            per_graph = [len(result[t.name][g]) for g in range(n_graphs)]
            correct = sum(per_graph)
            total = n_used * args.n_seeds
            rates = [c / args.n_seeds for c in per_graph[:n_used]]
            mean = sum(rates) / max(len(rates), 1)
            var = (sum((r - mean) ** 2 for r in rates) / max(len(rates) - 1, 1))
            sem = var ** 0.5 / max(len(rates), 1) ** 0.5
            stats[str(ratio)][t.name] = {
                'correct': correct, 'total': total,
                'rate': round(correct / max(total, 1), 4),
                'graph_mean': round(mean, 4), 'sem': round(sem, 4),
            }
            print(f"  {t.name:<16} {correct}/{total} = "
                  f"{correct / max(total, 1):.4f} (±{sem:.4f})")

    out = {
        'dataset': spec.name,
        'classifier': str(clf_path),
        'teachers': list(args.teachers),
        'budget_ratios': ratios,
        'n_graphs': n_graphs,
        'n_seeds': args.n_seeds,
        'success_definition': '教师最终揭示子图被上界分类器判对（is_correct）',
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'stats': stats,
    }

    output_dir = (resolve(args.output_dir) if args.output_dir else
                  CODE_DIR / 'result' / spec.name / 'teacher_success')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, output_dir / 'teacher_success.json')
    print(f"\nSaved: {output_dir / 'teacher_success.json'}")

    # Markdown 表（直接可贴进 Summary of results 的 teacher 成功概率表）；
    # 列名用构建后的 teacher.name（hybrid 的 α 会被格式化，如 a0.7 → a0.70）
    cols = [t.name for t in teachers]
    print('\n|     | ' + ' | '.join(cols) + ' |')
    print('| --- | ' + ' | '.join(['---'] * len(cols)) + ' |')
    for ratio in ratios:
        row = [f"{stats[str(ratio)][c]['rate']:.4f}" for c in cols]
        print(f"| {ratio * 100:.0f}%  | " + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
