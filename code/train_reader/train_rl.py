"""RL（REINFORCE）微调：BC/迁移起点 → 自采样强化学习（任意数据集 × 两种 reader）。

自 m5/base/train_rl.py、m6/attention/train_rl.py、m7/train_rl.py 三份合并：
- 起点：--bootstrap 任意 reader checkpoint（跨架构/跨数据集时部分权重
  迁移：同形复制 / 末维零填充 / 跳过）；
- 奖励：冻结上界分类器给出 binary/tanh/asymmetric；batch 优势归一化
  （负侧乘 --reward-neg-scale）；
- 断点续训 --resume（结构/超参取 checkpoint config）；--freeze 冻结共享层；
- 产物：result/{dataset}/experiments/{exp}/{reader-kind}/rl/。

用法:
  python code/train_reader/train_rl.py --dataset cifar10 --root D:\\superpixels \
      --reader-kind attention --bootstrap <bc ckpt> --classifier <上界分类器> \
      --reward asymmetric --graphs-per-batch 32 --seeds-per-graph 4
"""

import argparse
import datetime
import math
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools import cli as cli_utils  # noqa: E402
from tools.ckpt_io import load_optimizer, load_training_state, save_epoch_checkpoint, transfer_weights  # noqa: E402
from tools.config import build_experiment_config, load_json, save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from train_reader.core import (bootstrap_reader, freeze_shared_layers,  # noqa: E402
                               load_classifier_frozen, load_reader_from_checkpoint,
                               rl_train_epoch)


def resolve(p) -> Path:
    path = Path(p)
    return path if path.is_absolute() else CODE_DIR / path


def main():
    parser = argparse.ArgumentParser(description='RL 微调（任意数据集 × 两种 reader）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='训练图数限制（0=全部；冒烟传小值）')
    cli_utils.add_reader_args(parser, reader_kind_default='attention',
                              temperature_default=0.2)
    parser.add_argument('--bootstrap', default='',
                        help='迁移起点 reader checkpoint（与 --resume 二选一）')
    parser.add_argument('--classifier', required=True, help='上界分类器 checkpoint')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--graphs-per-batch', type=int, default=32)
    parser.add_argument('--seeds-per-graph', type=int, default=4)
    parser.add_argument('--reward', choices=['binary', 'tanh', 'asymmetric'],
                        default='binary')
    parser.add_argument('--reward-neg-scale', type=float, default=0.1,
                        help='batch 优势归一化后负侧优势的降权系数 λ（λ=1 即无不对称）')
    parser.add_argument('--freeze', action='store_true',
                        help='冻结共享层，仅训练特征投影层（冻结消融用）')
    parser.add_argument('--edge-feat-dim', type=int, default=None,
                        help='覆盖数据集边特征维度')
    parser.add_argument('--node-feat-dim', type=int, default=None,
                        help='覆盖数据集节点特征维度')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp', default='default', help='实验名')
    parser.add_argument('--output-dir', default='', help='覆盖默认输出目录')
    parser.add_argument('--resume', default='',
                        help='RL checkpoint 续训；结构与超参取 checkpoint config')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='', help='实验 config JSON')
    args = parser.parse_args()

    cfg_baseline = load_json(args.config) if args.config else {}
    if cfg_baseline:
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
    cli_utils.attach_defaults(args, parser)

    assert bool(args.bootstrap) != bool(args.resume), \
        "--bootstrap 与 --resume 必须二选一"

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs)
    spec = ds.spec
    budget_options = spec.budget_options
    ln_k = math.log(spec.n_classes)
    train_graphs = ds.train_graphs
    train_labels = ds.train_labels
    print(f"Train graphs: {len(train_graphs)} | budgets: {list(budget_options)}")

    # ---- 冻结分类器（reward 来源） ----
    clf_path = resolve(args.classifier)
    classifier = load_classifier_frozen(clf_path, device)
    print(f"Classifier loaded: {clf_path}")

    # ---- reader：resume 直接加载 / bootstrap 迁移 ----
    start_epoch, training_log = 1, []
    resume_ckpt = None
    n_transferred, n_padded, skipped = 0, 0, []
    if args.resume:
        resume_path = resolve(args.resume)
        resume_ckpt = torch.load(str(resume_path), map_location='cpu',
                                 weights_only=False)
        reader = load_reader_from_checkpoint(resume_ckpt, device,
                                             stochastic=True,
                                             temperature=args.temperature)
        config = (resume_ckpt.get('config')
                  if isinstance(resume_ckpt.get('config'), dict) else None) or {}
        optimizer = torch.optim.Adam(reader.parameters(),
                                     lr=config.get('lr', args.lr))
        load_optimizer(optimizer, resume_ckpt, device=str(device))
        start_epoch, training_log = load_training_state(resume_ckpt)
        if start_epoch > args.epochs:
            raise SystemExit(f"checkpoint 已完成 {start_epoch - 1} 轮，"
                             f"--epochs={args.epochs} 无需续训")
        print(f"Resuming from {resume_path} (done={start_epoch - 1}, "
              f"target={args.epochs})")
    else:
        bootstrap_path = resolve(args.bootstrap)
        src_ckpt = torch.load(str(bootstrap_path), map_location='cpu',
                              weights_only=False)
        reader, (n_transferred, n_padded, skipped) = bootstrap_reader(
            src_ckpt, device, stochastic=True, temperature=args.temperature)
        print(f"Bootstrap {args.bootstrap}: transferred={n_transferred}, "
              f"padded={n_padded}, skipped={len(skipped)} (新层/维度不匹配)")
        freeze_shared_layers(reader, args.freeze)
        optimizer = torch.optim.Adam(
            [p for p in reader.parameters() if p.requires_grad], lr=args.lr)

    builder = SubgraphBuilder()
    total_params = sum(p.numel() for p in reader.parameters())
    trainable_params = sum(p.numel() for p in reader.parameters() if p.requires_grad)
    print(f"Reader: {reader.config.kind}, params={total_params:,} "
          f"(trainable {trainable_params:,})")

    # ---- config 落盘 ----
    output_dir = (resolve(args.output_dir) if args.output_dir else
                  CODE_DIR / 'result' / spec.name / 'experiments' / args.exp /
                  reader.config.kind / 'rl')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and isinstance(resume_ckpt.get('config'), dict) \
            and resume_ckpt['config'].get('reader'):
        config = resume_ckpt['config']
        config['epochs'] = args.epochs
    else:
        config = build_experiment_config(args, extra={
            'reader': reader.config.to_dict(),
            'dataset': spec.name,
            'bootstrap': args.bootstrap,
            'classifier': str(clf_path),
            'n_transferred': n_transferred,
            'n_padded': n_padded,
            'n_skipped': len(skipped),
            'freeze': args.freeze,
            'reward': args.reward,
            'reward_neg_scale': args.reward_neg_scale,
            'budget_options': list(budget_options),
            'epochs': args.epochs,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        })
    save_json(config, output_dir / 'config.json')
    print(f"Config saved: {output_dir / 'config.json'}")

    # ---- 训练循环 ----
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = rl_train_epoch(
            reader, classifier, builder, optimizer,
            train_graphs, train_labels[:len(train_graphs)],
            args.graphs_per_batch, args.seeds_per_graph,
            device, args.reward, ln_k, budget_options,
            args.reward_neg_scale)
        print(f"  Train: loss={metrics['loss']:.4f}  "
              f"avg_R={metrics['avg_R']:.3f}  acc={metrics['acc']:.3f}")
        training_log.append({'epoch': epoch,
                             'train_loss': metrics['loss'],
                             'train_avg_R': metrics['avg_R'],
                             'train_acc': metrics['acc']})
        save_epoch_checkpoint(
            output_dir / f'checkpoint_epoch{epoch}.pt',
            epoch=epoch, reader=reader, optimizer=optimizer,
            config=config, training_log=training_log)
        save_json(training_log, output_dir / 'training_log.json')
        print(f"  Saved: checkpoint_epoch{epoch}.pt")

    print(f"\nREINFORCE complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
