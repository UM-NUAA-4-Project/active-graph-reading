"""分类器 held-out 验证：checkpoint 在 valid/test 全图上的准确率。

train_full_graph.py 训练中只做 test 集周期评估，且以 test acc 兼任
选型/调度/早停信号，没有独立的验证集评估；本脚本补齐：给定分类器
checkpoint，在各 split（默认 valid+test，数据集无验证集自动跳过并告警）
的全图上前向评估，输出总体 + 逐类准确率 → checkpoint 同目录
eval_classifier.json。顶层 full_graph_ref 键与 reader 评估 JSON 的
config.full_graph_ref 同名同义，可直接抄入三阶段图参照。

PPA 参考值（m3 best_model_ft.pt，全 split）：valid=0.642 / test=0.656。

用法:
  python code/train_classifier/eval_classifier.py --dataset collab \
      --root D:\\data\\COLLAB \
      --checkpoint result/collab/classifier/finetune/best_collab_classifier.pt
  # dataset 可省略（从 checkpoint 内嵌 config 推断）
  python code/train_classifier/eval_classifier.py --root D:\\ogb-master\\dataset \
      --checkpoint result/ppa/classifier/best_model_ft.pt
"""

import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
import dgl  # noqa: E402

from classifier import load_classifier  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import resolve  # noqa: E402


@torch.no_grad()
def evaluate_split(clf, graphs, labels, device, batch_size=256):
    """全图 eval 模式评估 → (acc, n, per_class{lbl: {'acc','n'}})。"""
    clf.eval()
    correct = total = 0
    per_class = defaultdict(lambda: [0, 0])   # lbl -> [correct, total]
    for start in range(0, len(graphs), batch_size):
        idx = list(range(start, min(start + batch_size, len(graphs))))
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = labels[start:start + len(idx)].to(device)
        pred = clf(batch_g, nids).argmax(dim=-1)
        hit = (pred == y)
        correct += int(hit.sum().item())
        total += len(idx)
        for i, lbl in enumerate(y.tolist()):
            per_class[lbl][0] += int(hit[i].item())
            per_class[lbl][1] += 1
    per_class = {str(lbl): {'acc': c / max(n, 1), 'n': n}
                 for lbl, (c, n) in sorted(per_class.items())}
    return correct / max(total, 1), total, per_class


def main():
    parser = argparse.ArgumentParser(description='分类器 held-out 验证（valid/test 全图）')
    parser.add_argument('--dataset', default='',
                        help='数据集名（默认从 checkpoint 内嵌 config 推断）')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--checkpoint', required=True, help='分类器 checkpoint')
    parser.add_argument('--splits', nargs='+', default=['valid', 'test'],
                        choices=['train', 'valid', 'test'],
                        help='评估的 split（数据集缺少的 split 自动跳过并告警）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每个 split 的图数限制（0=全部；PPA 全量 train 需配）')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 checkpoint 同目录）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = resolve(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    ds_name = args.dataset or (ckpt.get('config') or {}).get('dataset', '')
    if not ds_name:
        raise SystemExit("无法确定数据集：checkpoint 无内嵌 config.dataset，"
                         "请显式传 --dataset")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset: {ds_name}")

    ds = get_dataset(ds_name, root=args.root,
                     splits=sorted(set(args.splits)), max_graphs=args.max_graphs)
    spec = ds.spec
    assert spec.name == ds_name, f"注册名 {spec.name} != 推断名 {ds_name}"

    clf = load_classifier(str(ckpt_path), device=device, freeze=True)

    evaluated, full_graph_ref = [], {}
    for split in args.splits:
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if len(graphs) == 0:
            print(f"[skip] 数据集 '{spec.name}' 无 {split} 划分（或为空），跳过")
            continue
        acc, n, per_class = evaluate_split(clf, graphs, labels, device,
                                           args.batch_size)
        evaluated.append((split, acc, n, per_class))
        full_graph_ref[split] = round(acc, 6)
        print(f"[{split}] acc = {acc:.4f}（{n} 图）")

    if not evaluated:
        raise SystemExit(f"数据集 '{spec.name}' 所请求 split {args.splits} 全部为空")

    # 与训练记录交叉核对（history.json 的 best 以 test acc 选型）
    hist_file = ckpt_path.parent / 'history.json'
    if hist_file.exists():
        import json
        best = (json.loads(hist_file.read_text(encoding='utf-8')).get('best')
                or {})
        if best.get('test_acc') is not None:
            print(f"训练记录 best: test_acc={best['test_acc']:.4f} "
                  f"@ epoch {best.get('epoch')}（{best.get('checkpoint', '?')}）"
                  + ("  [与重算一致]" if abs(best['test_acc'] -
                       dict((s, a) for s, a, _n, _pc in evaluated)
                       .get('test', -1)) < 5e-4 else ""))

    json_out = {split: {'acc': round(acc, 6), 'n': n, 'per_class': per_class}
                for split, acc, n, per_class in evaluated}
    json_out['full_graph_ref'] = full_graph_ref
    json_out['config'] = {
        'dataset': spec.name,
        'mode': 'classifier_full_graph',
        'checkpoint': str(ckpt_path),
        'splits': [split for split, _a, _n, _pc in evaluated],
        'max_graphs': args.max_graphs,
        'batch_size': args.batch_size,
        'device': str(device),
        'classifier': (ckpt.get('config') or {}).get('classifier', {}),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }

    out_dir = resolve(args.output_dir) if args.output_dir else ckpt_path.parent
    out_path = out_dir / 'eval_classifier.json'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(json_out, out_path)
    print(f"JSON saved to {out_path}")


if __name__ == '__main__':
    main()
