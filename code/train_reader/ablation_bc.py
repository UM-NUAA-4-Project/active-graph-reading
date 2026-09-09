"""消融专用 BC 训练：evidence reader 变体（A2–A4，见
docs/design/evidence-reader-ablation.md）。

独立于正常训练脚本 train_bc.py——本脚本只新增 --evidence-variant 开关并
复用其轨迹装载工具与 core.bc_train_epoch 训练循环，train_bc.py /
tools/cli.py / eval 链路零改动。变体经 ReaderConfig.evidence_variant 内嵌
进 checkpoint，eval.py / train_rl.py --bootstrap 的重建链路自动识别。

变体语义（ReaderConfig.evidence_variant）：
- full     ：LSTM 积分 + 历史引导 AttnPool（默认，等价 train_bc.py 产物）
- no_pool  ：保留 LSTM；读出退化为对已揭示嵌入的非引导均值池化
- no_lstm  ：保留 AttnPool；h_t = Δ_t 直通（递归关闭）
- v1       ：两者全去（= 设计文档 v1 消融档）

用法（教师轨迹与 train_bc.py 完全共用）:
  python code/train_reader/ablation_bc.py --dataset collab \
      --root D:\\data\\COLLAB \
      --data-dir code/data/collab/trajectories_filtered \
      --evidence-variant no_pool --exp ablation_no_pool --epochs 10
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
from train_reader.core import bc_train_epoch  # noqa: E402
# 复用 train_bc 的轨迹装载工具（仅 import，不改动该脚本）
from train_reader.train_bc import build_graph_cache, merge_trajectories  # noqa: E402

VARIANTS = ('full', 'no_pool', 'no_lstm', 'v1')


def resolve(p) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='消融 BC 训练（evidence 变体：no_pool / no_lstm / v1）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='', help='数据集根目录（必填）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 最多加载图数（0=不限；同时过滤轨迹）')
    parser.add_argument('--stratified', action='store_true',
                        help='--max-graphs 按类别比例分层抽样（默认前 N 截断；'
                             '轨迹按子集内 graph_idx 过滤）')
    parser.add_argument('--data-dir', default='',
                        help='轨迹目录（默认 data/{dataset}/trajectories）')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'structural-degree'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    cli_utils.add_reader_args(parser, reader_kind_default='evidence')
    parser.add_argument('--evidence-variant', default='full', choices=VARIANTS,
                        help='evidence 消融变体（full=与 train_bc 等价）')
    parser.add_argument('--evidence-lambda', type=float, default=1.0,
                        help='L_ev 权重（λ=0 即 A5 档：证据流整体关闭）')
    parser.add_argument('--edge-feat-dim', type=int, default=None)
    parser.add_argument('--node-feat-dim', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp', default='',
                        help='实验名（默认 ablation_{variant}）')
    parser.add_argument('--output-dir', default='',
                        help='覆盖默认输出目录')
    parser.add_argument('--resume', default='',
                        help='BC checkpoint 续训（结构与变体取 checkpoint config）')
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
    if not args.exp:
        args.exp = f'ablation_{args.evidence_variant}'
    cli_utils.attach_defaults(args, parser)
    from tools.config import require_args
    require_args(args, ['root'])

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | evidence-variant={args.evidence_variant}")

    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs,
                     stratified=args.stratified)
    spec = ds.spec

    # ---- 1. 轨迹 & 图缓存 ----
    data_dir = resolve(args.data_dir) if args.data_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories'
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir, args.teachers)
    if args.max_graphs > 0:
        n_before = len(all_trajs)
        if args.stratified:
            allowed = set(ds.train_indices)
            all_trajs = [t for t in all_trajs if t.graph_idx in allowed]
            mode = f'stratified（子集内 {len(allowed)} 图）'
        else:
            all_trajs = [t for t in all_trajs if t.graph_idx < args.max_graphs]
            mode = 'prefix'
        print(f"  --max-graphs={args.max_graphs} [{mode}]: "
              f"{n_before} -> {len(all_trajs)} trajectories")
    if not all_trajs:
        raise SystemExit(f"无轨迹可用：{data_dir}")

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
        cfg_dict = cli_utils.reader_config_from_args(args)
        cfg_dict['kind'] = 'evidence'
        cfg_dict['n_classes'] = spec.n_classes
        cfg_dict['evidence_variant'] = args.evidence_variant
        reader_cfg = ReaderConfig.from_dict(cfg_dict)
        reader_cfg.edge_feat_dim = args.edge_feat_dim or spec.edge_feat_dim
        reader_cfg.node_feat_dim = args.node_feat_dim or spec.node_feat_dim
        reader_cfg.validate()
        reader = build_reader(reader_cfg).to(device)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"Reader: {reader.config.kind}[{reader.config.evidence_variant}], "
          f"params={total_params:,}")

    # ---- 3. config 落盘 ----
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

    # ---- 4. 训练循环（与 train_bc 同一 core 循环） ----
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = bc_train_epoch(reader, optimizer, graph_cache, all_trajs,
                                 args.batch_size, device,
                                 temperature=1.0,
                                 evidence_lambda=args.evidence_lambda)
        print(f"  Train loss: {metrics['loss']:.4f}"
              f", act_acc: {metrics['act_acc']:.4f}"
              + (f", ev_acc: {metrics['ev_acc']:.4f}"
                 if 'ev_acc' in metrics else '')
              + (f", ev_acc_final: {metrics['ev_acc_final']:.4f}"
                 if 'ev_acc_final' in metrics else ''))
        log_entry = {'epoch': epoch, 'train_loss': metrics['loss'],
                     'act_acc': metrics['act_acc']}
        if 'ev_acc' in metrics:
            log_entry['ev_acc'] = metrics['ev_acc']
        if 'ev_acc_final' in metrics:
            log_entry['ev_acc_final'] = metrics['ev_acc_final']
        training_log.append(log_entry)

        save_epoch_checkpoint(
            output_dir / f'reader_checkpoint_epoch{epoch}.pt',
            epoch=epoch, reader=reader, optimizer=optimizer,
            config=config, training_log=training_log)
        save_json(training_log, output_dir / 'training_log.json')
        print(f"  Saved: reader_checkpoint_epoch{epoch}.pt + training_log.json")

    print(f"\nAblation BC complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
