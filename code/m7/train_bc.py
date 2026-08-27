"""M7 BC 训练：在任意 m7 registry 数据集上，从教师轨迹行为克隆 base reader。

与 m5/base/train_bc.py 的差异：
  - 数据走 m7 registry（get_dataset → ds.train_graphs 已在内存，无 CSV/dgl.load_graphs）
  - graph_idx 语义 = ds.train_graphs 本地位置（与 generate_trajectories.py / train_rl.py 一致）
  - 轨迹目录 = m7/data/trajectories（trajectories_r{ratio}.pt，generate_trajectories.py 输出）
  - 去掉 classifier 参考加载；reader 用 spec 的 edge_feat_dim / node_feat_dim
  - 无验证集划分：全部轨迹用于 BC 训练；设备由 --device 指定
  - checkpoint config 兼容 m7 eval.py（reader_kind / n_layers_reader / dropout_reader 等键）

用法:
  python code/m7/train_bc.py --dataset cifar10 --root D:\\superpixels \\
      --data-dir m7/data/trajectories --device cuda:0 --epochs 10
  python code/m7/train_bc.py --dataset cifar10 --root D:\\superpixels \\
      --data-dir m7/data/trajectories --device cuda:0 --epochs 10 \\
      --reader-kind attention --m 4 --n-heads 4

断点续训:
  python code/m7/train_bc.py --dataset cifar10 --root D:\\superpixels \\
      --data-dir m7/data/trajectories --device cuda:0 \\
      --resume m7/checkpoint/bc/base/reader_checkpoint_epoch5.pt --epochs 10
  结构与超参取 checkpoint 自身 config；--epochs 为续训后的总轮数。
  python code/m7/train_bc.py --dataset cifar10 --root /mnt/data/superpixels 
  --data-dir m7/data/trajectories_filtered --device cuda:0 --resume m7/checkpoint/bc/attention/reader_checkpoint_epoch7.pt
"""
import argparse
import json
import random
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
_project_root = _repo_root.parent                   # 仓库根
sys.path.insert(0, str(_repo_root))
# 移除 code/m7 自身：dgl→torchdata 的 `import datasets` 不得解析到 code/m7/datasets
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import torch
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.reader.attention.attention_reader import AttentionReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m7.datasets.registry import get_dataset
from m6.training_utils import reveal_and_update_beliefs_batch


def ratio_key(r: float) -> str:
    """0.05 -> '0_050'（与 generate_trajectories.py 一致）。"""
    return f'{r:.3f}'.replace('.', '_')


def merge_trajectories(data_dir: Path, teachers: list, ratios: list) -> list:
    """加载 trajectories_r{ratio}.pt，按教师名展平为 List[Trajectory]。"""
    from m4.teachers.base import Trajectory

    all_trajs = []
    for ratio in ratios:
        fname = f'trajectories_r{ratio_key(ratio)}.pt'
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        trajs_dict = data['trajectories']
        n_before = len(all_trajs)
        for teacher_name in teachers:
            for per_graph_list in trajs_dict.get(teacher_name, []):
                all_trajs.extend(per_graph_list)
        print(f"  {fname}: +{len(all_trajs) - n_before} trajectories")
    print(f"Total merged: {len(all_trajs)} trajectories")
    return all_trajs


def build_bc_reader(kind, spec, args):
    """按 reader-kind 构造 BC reader（从零初始化；调用方自行 .to(device)）。"""
    common = dict(
        edge_feat_dim=spec.edge_feat_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers_reader,
        dropout=args.dropout_reader,
        node_feat_dim=spec.node_feat_dim,
    )
    if kind == 'attention':
        return AttentionReader(m=args.m, n_heads=args.n_heads, **common)
    return LearnableReader(**common)


def train_one_epoch_base(reader, optimizer, graph_cache, train_trajs, batch_size, device):
    """单轮 BC 训练（跨图锁步批处理），返回平均 step loss。m5 移植。"""
    reader.train()
    total_loss = 0.0
    total_steps = 0

    random.shuffle(train_trajs)
    n_batches = (len(train_trajs) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        # ---- Pass 1: create states for each trajectory ----
        all_states = []
        reveal_orders = []
        step_counters = []

        for traj in batch_trajs:
            g, _label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)
            state._budget_limit = len(traj.reveal_order)
            all_states.append(state)
            reveal_orders.append(traj.reveal_order)
            step_counters.append(0)

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(len(ro) for ro in reveal_orders)
        step_losses = []

        # ---- Pass 2: cross-graph lockstep ----
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty
                          and step_counters[j] < len(reveal_orders[j])]
            if not active_idx:
                break

            batch_results = reader.forward_states(
                [all_states[j] for j in active_idx], temperature=1.0)

            for k, j in enumerate(active_idx):
                scores, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue

                teacher_node = reveal_orders[j][step_counters[j]]
                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    active[j] = False
                    continue

                step_losses.append(-torch.log(probs[target_idx] + 1e-8))
                reveal_node(all_states[j], teacher_node)
                step_counters[j] += 1

                if step_counters[j] >= len(reveal_orders[j]):
                    active[j] = False

        if not step_losses:
            continue

        batch_loss = torch.stack(step_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n = len(step_losses)
        total_loss += batch_loss.item() * n
        total_steps += n

        avg_step_loss = total_loss / max(total_steps, 1)
        pbar.set_postfix({
            'batch_loss': f'{batch_loss.item():.4f}',
            'avg_step_loss': f'{avg_step_loss:.4f}',
            'batch': f'{batch_idx + 1}/{n_batches}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
        })

    return total_loss / max(total_steps, 1)


def train_one_epoch_attention(reader, optimizer, graph_cache, train_trajs,
                              batch_size, device, temperature=1.0):
    """AttentionReader 单轮 BC 训练（belief lockstep，m6 移植）。"""
    reader.train()
    total_loss = 0.0
    total_steps = 0

    random.shuffle(train_trajs)
    n_batches = (len(train_trajs) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(train_trajs), batch_size), desc="  Training")
    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, len(train_trajs))
        batch_trajs = train_trajs[batch_start:batch_end]

        # ---- Pass 1: create states + beliefs ----
        all_states = []
        reveal_orders = []
        step_counters = []
        for traj in batch_trajs:
            g, _label = graph_cache[traj.graph_idx]
            state = GraphState(g, seed_node=traj.seed_node)
            state._budget_limit = len(traj.reveal_order)
            all_states.append(state)
            reveal_orders.append(traj.reveal_order)
            step_counters.append(0)

        n_total = len(all_states)
        active = [True] * n_total
        max_budget = max(len(ro) for ro in reveal_orders)
        all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_total)]
        all_subgraphs = [None] * n_total
        step_losses = []

        # ---- Pass 2: cross-graph lockstep with belief update ----
        for _step in range(max_budget):
            active_idx = [j for j in range(n_total)
                          if active[j] and not all_states[j].is_frontier_empty
                          and step_counters[j] < len(reveal_orders[j])]
            if not active_idx:
                break

            active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
            active_subgraphs = None
            if all_subgraphs[active_idx[0]] is not None:
                active_subgraphs = [all_subgraphs[j] for j in active_idx]

            batch_results = reader.forward_states(
                [all_states[j] for j in active_idx],
                temperature=temperature,
                beliefs=active_beliefs,
                subgraphs=active_subgraphs,
            )

            batch_states, batch_nodes, batch_local_idx = [], [], []
            for k, j in enumerate(active_idx):
                _, probs, frontier_nodes = batch_results[k]
                if not frontier_nodes:
                    active[j] = False
                    continue
                teacher_node = reveal_orders[j][step_counters[j]]
                try:
                    target_idx = frontier_nodes.index(teacher_node)
                except ValueError:
                    active[j] = False
                    continue
                step_losses.append(-torch.log(probs[target_idx] + 1e-8))
                step_counters[j] += 1
                batch_states.append(all_states[j])
                batch_nodes.append(teacher_node)
                batch_local_idx.append(j)

            if batch_states:
                batch_beliefs = torch.cat(
                    [all_beliefs[j] for j in batch_local_idx], dim=0)
                new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                    reader, batch_states, batch_nodes, batch_beliefs, device)
                for k, j in enumerate(batch_local_idx):
                    all_beliefs[j] = new_beliefs[k:k + 1]
                    all_subgraphs[j] = new_subgraphs[k]

            for j in active_idx:
                if active[j] and step_counters[j] >= len(reveal_orders[j]):
                    active[j] = False

        if not step_losses:
            continue

        batch_loss = torch.stack(step_losses).mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n = len(step_losses)
        total_loss += batch_loss.item() * n
        total_steps += n

        avg_step_loss = total_loss / max(total_steps, 1)
        pbar.set_postfix({
            'batch_loss': f'{batch_loss.item():.4f}',
            'avg_step_loss': f'{avg_step_loss:.4f}',
            'batch': f'{batch_idx + 1}/{n_batches}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
        })

    return total_loss / max(total_steps, 1)


def main():
    parser = argparse.ArgumentParser(description='M7 BC 训练（m7 registry 数据集）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录（如 D:\\superpixels）')
    parser.add_argument('--data-dir', default='m7/data/trajectories',
                        help='轨迹目录（相对 code/ 或绝对路径）')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'structural-degree'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers-reader', type=int, default=2, dest='n_layers_reader')
    parser.add_argument('--dropout-reader', type=float, default=0.25, dest='dropout_reader')
    parser.add_argument('--reader-kind', choices=['base', 'attention'], default='base',
                        help='reader 类型：base=MLP LearnableReader；'
                             'attention=AttentionReader（belief lockstep BC）')
    parser.add_argument('--m', type=int, default=4, help='belief token 数（仅 attention）')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='attention 头数（仅 attention）')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='训练 softmax 温度（仅 attention，默认 1.0）')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录（相对 code/ 或绝对路径；默认 '
                             'm7/checkpoint/bc/{reader-kind}）')
    parser.add_argument('--resume', default='',
                        help='BC checkpoint 续训（相对 code/ 或绝对路径）；'
                             '结构与超参取 checkpoint 自身 config，'
                             '--epochs 为续训后的总轮数')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = f'm7/checkpoint/bc/{args.reader_kind}'

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root or None, splits=['train'])
    spec = ds.spec

    # ---- 1. 加载轨迹 & 图（无验证集：全部轨迹用于 BC 训练） ----
    data_dir = _repo_root / args.data_dir
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir, args.teachers, spec.budget_options)
    if not all_trajs:
        raise SystemExit(
            f"无轨迹可用：{data_dir} 为空或未找到（先运行 m7/generate_trajectories.py）")
    train_trajs = all_trajs

    print("\n=== Building graph cache ===")
    needed = {t.graph_idx for t in all_trajs}
    graph_cache = {i: (ds.train_graphs[i], int(ds.train_labels[i])) for i in needed}
    print(f"  {len(graph_cache)} graphs cached")

    # ---- 2. 创建 reader（从零初始化 / --resume 续训） ----
    start_epoch = 0
    training_log = []
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = _repo_root / resume_path
        ckpt = torch.load(str(resume_path), map_location='cpu', weights_only=False)
        ckpt_cfg = ckpt.get('config', {})
        for key in ('reader_kind', 'hidden_dim', 'n_layers_reader',
                    'dropout_reader', 'm', 'n_heads', 'lr'):
            if key in ckpt_cfg:
                setattr(args, key, ckpt_cfg[key])
        reader = build_bc_reader(args.reader_kind, spec, args).to(device)
        reader.load_state_dict(ckpt['reader_state_dict'], strict=True)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"  Optimizer state loaded (lr={args.lr})")
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"  WARNING: optimizer state unavailable ({e}), fresh optimizer")
        start_epoch = int(ckpt.get('epoch', 0)) + 1  # 已完成轮数
        if start_epoch >= args.epochs:
            raise SystemExit(f"checkpoint 已完成 {start_epoch} 轮，"
                             f"--epochs={args.epochs} 无需续训")
        training_log = ckpt.get('training_log', [])
        print(f"Resuming from {resume_path}"
              f" (done={start_epoch}, target={args.epochs})")
    else:
        reader = build_bc_reader(args.reader_kind, spec, args).to(device)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in reader.parameters())
    kind_info = (f"\nReader: {args.reader_kind}, params={total_params:,}, "
                 f"edge_feat_dim={spec.edge_feat_dim}, node_feat_dim={spec.node_feat_dim}")
    if args.reader_kind == 'attention':
        kind_info += f", m={args.m}, n_heads={args.n_heads}, temperature={args.temperature}"
    print(kind_info)

    # ---- 3. 训练 ----
    output_dir = _repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and ckpt.get('config'):
        config = ckpt['config']  # 续训沿用原 config（config.json 与 checkpoint 一致）
    else:
        config = {
            'reader_kind': args.reader_kind,
            'dataset': args.dataset,
            'hidden_dim': args.hidden_dim,
            'n_layers_reader': args.n_layers_reader,
            'dropout_reader': args.dropout_reader,
            'm': args.m,
            'n_heads': args.n_heads,
            'temperature': args.temperature,
            'edge_feat_dim': spec.edge_feat_dim,
            'node_feat_dim': spec.node_feat_dim,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'total_params': total_params,
            'n_train_trajs': len(train_trajs),
        }
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved: {output_dir / 'config.json'}")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        if args.reader_kind == 'base':
            train_loss = train_one_epoch_base(
                reader, optimizer, graph_cache, train_trajs, args.batch_size, device)
        else:
            train_loss = train_one_epoch_attention(
                reader, optimizer, graph_cache, train_trajs, args.batch_size, device,
                temperature=args.temperature)
        print(f"  Train loss: {train_loss:.4f}")
        training_log.append({'epoch': epoch, 'train_loss': train_loss})

        checkpoint = {
            'epoch': epoch - 1,
            'reader_state_dict': reader.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_log': training_log,
            'config': config,
        }
        ckpt_path = output_dir / f'reader_checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, str(ckpt_path))
        print(f"  Saved: {ckpt_path}")

        log_path = output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)
        print(f"  Log saved: {log_path}")


if __name__ == '__main__':
    main()
