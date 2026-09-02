"""BC（行为克隆）训练：任意注册数据集 × base/attention reader。

自 m5/base/train_bc.py（PPA+base）、m6/attention/train_bc.py（PPA+attention）、
m7/train_bc.py（registry）三份合并：
- 数据：datasets/registry；轨迹目录默认 data/{dataset}/trajectories（也接受
  filtered/balanced 变体）；PPA 轨迹按 OGB graph_idx 经 load_graph_subset
  选择性加载图（不占全量内存）；
- 结构配置：tools.cli 的 reader 参数（支持逐层 hidden-dims 列表）或 --config
  JSON；effective config 落盘 <output>/config.json 并内嵌每个 checkpoint；
- 断点续训：--resume（结构/超参取 checkpoint 自身 config，--epochs 为总轮数）；
- 产物：result/{dataset}/experiments/{exp}/{reader-kind}/bc/。

用法:
  python code/train_reader/train_bc.py --dataset cifar10 --root D:\\superpixels \
      --data-dir data/cifar10/trajectories_filtered --reader-kind attention --epochs 10
  # 断点续训
  ... --resume result/cifar10/experiments/exp1/attention/bc/reader_checkpoint_epoch5.pt --epochs 10
"""

import argparse
import datetime
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader import ReaderConfig, build_reader  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools import cli as cli_utils  # noqa: E402
from tools.ckpt_io import load_optimizer, load_training_state, save_epoch_checkpoint  # noqa: E402
from tools.config import build_experiment_config, load_json, save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from tools.trajectory_io import (load_trajectory_file,  # noqa: E402
                                 normalize_trajectory)
from train_reader.core import bc_train_epoch  # noqa: E402


def resolve(p) -> Path:
    path = Path(p)
    return path if path.is_absolute() else CODE_DIR / path


def merge_trajectories(data_dir: Path, teachers: list) -> list:
    """加载目录下 trajectories_r*.pt，按教师名过滤展平。"""
    all_trajs = []
    for fpath in sorted(data_dir.glob('trajectories_r*.pt')):
        data = load_trajectory_file(fpath)
        n_before = len(all_trajs)
        for teacher_name in teachers:
            for per_graph_list in data['trajectories'].get(teacher_name, []):
                for t in per_graph_list:
                    all_trajs.append(normalize_trajectory(t))
        print(f"  {fpath.name}: +{len(all_trajs) - n_before} trajectories")
    print(f"Total merged: {len(all_trajs)} trajectories")
    return all_trajs


def build_graph_cache(spec, ds, needed: set) -> dict:
    """graph_idx → (graph, label)。PPA（有 load_graph_subset）选择性加载。"""
    if spec.load_graph_subset is not None:
        return spec.load_graph_subset(spec, needed)
    return {i: (ds.train_graphs[i], int(ds.train_labels[i])) for i in needed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='BC 训练（任意数据集 × 两种 reader）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 最多加载图数（0=不限；PPA 必配）')
    parser.add_argument('--data-dir', default='',
                        help='轨迹目录（默认 data/{dataset}/trajectories）')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'structural-degree'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    cli_utils.add_reader_args(parser, reader_kind_default='base')
    parser.add_argument('--edge-feat-dim', type=int, default=None,
                        help='覆盖数据集边特征维度（默认跟随 DatasetSpec）')
    parser.add_argument('--node-feat-dim', type=int, default=None,
                        help='覆盖数据集节点特征维度（默认跟随 DatasetSpec）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp', default='default',
                        help='实验名（输出目录 result/{ds}/experiments/{exp}/）')
    parser.add_argument('--output-dir', default='',
                        help='覆盖默认输出目录')
    parser.add_argument('--resume', default='',
                        help='BC checkpoint 续训；结构与超参取 checkpoint config')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='', help='实验 config JSON')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg_baseline = load_json(args.config) if args.config else {}
    if cfg_baseline:
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
    cli_utils.attach_defaults(args, parser)

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs)
    spec = ds.spec

    # ---- 1. 轨迹 & 图缓存（无验证集：全部轨迹用于 BC） ----
    data_dir = resolve(args.data_dir) if args.data_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories'
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir, args.teachers)
    if not all_trajs:
        raise SystemExit(f"无轨迹可用：{data_dir}（先运行 trajectories/generate_trajectories.py）")

    print("\n=== Building graph cache ===")
    graph_cache = build_graph_cache(spec, ds, {t.graph_idx for t in all_trajs})
    print(f"  {len(graph_cache)} graphs cached")

    # ---- 2. reader（从零 / --resume） ----
    start_epoch, training_log = 1, []
    resume_ckpt = None
    if args.resume:
        resume_path = resolve(args.resume)
        resume_ckpt = torch.load(str(resume_path), map_location='cpu',
                                 weights_only=False)
        # resume 统一走 core.build（权重结构推断，兼容新旧 checkpoint）
        from train_reader.core import load_reader_from_checkpoint
        reader = load_reader_from_checkpoint(resume_ckpt, device)
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
        reader_cfg = ReaderConfig.from_dict(cli_utils.reader_config_from_args(args))
        reader_cfg.edge_feat_dim = args.edge_feat_dim or spec.edge_feat_dim
        reader_cfg.node_feat_dim = args.node_feat_dim or spec.node_feat_dim
        reader_cfg.validate()
        reader = build_reader(reader_cfg).to(device)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"Reader: {reader.config.kind}, params={total_params:,}, "
          f"edge_feat_dim={reader.config.edge_feat_dim}, "
          f"node_feat_dim={reader.config.node_feat_dim}")

    # ---- 3. config 落盘（resume 沿用原 config，仅更新 epochs） ----
    output_dir = (resolve(args.output_dir) if args.output_dir else
                  CODE_DIR / 'result' / spec.name / 'experiments' / args.exp /
                  reader.config.kind / 'bc')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and isinstance(resume_ckpt.get('config'), dict) \
            and resume_ckpt['config'].get('reader'):
        config = resume_ckpt['config']
        config['epochs'] = args.epochs
    else:
        config = build_experiment_config(args, extra={
            'reader': reader.config.to_dict(),
            'dataset': spec.name,
            'dataset_spec': {'n_classes': spec.n_classes,
                             'node_feat_dim': spec.node_feat_dim,
                             'edge_feat_dim': spec.edge_feat_dim,
                             'budget_options': list(spec.budget_options)},
            'epochs': args.epochs,
            'total_params': total_params,
            'n_train_trajs': len(all_trajs),
            'data_dir': str(data_dir),
            'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        })
    save_json(config, output_dir / 'config.json')
    print(f"Config saved: {output_dir / 'config.json'}")

    # ---- 4. 训练循环 ----
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = bc_train_epoch(reader, optimizer, graph_cache, all_trajs,
                                 args.batch_size, device,
                                 temperature=1.0)
        print(f"  Train loss: {metrics['loss']:.4f}")
        training_log.append({'epoch': epoch, 'train_loss': metrics['loss']})

        save_epoch_checkpoint(
            output_dir / f'reader_checkpoint_epoch{epoch}.pt',
            epoch=epoch, reader=reader, optimizer=optimizer,
            config=config, training_log=training_log)
        save_json(training_log, output_dir / 'training_log.json')
        print(f"  Saved: reader_checkpoint_epoch{epoch}.pt + training_log.json")

    print(f"\nBC complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
