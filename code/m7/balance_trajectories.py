"""M7 丰富并平衡轨迹：每个预算档每个类别恰好 2000 条（3 档 × 10 类 × 2000 = 60000）。

断点续跑：每次迭代将已验证的新增轨迹写入 --state-file（默认
<output-dir>/balance_state.pt），下次运行自动加载续跑，已完成的类不重复生成。
--labels 可指定本次平衡的类别（其余类仅按同图去重+每图 top-k 保留，不做
每类 2000 截断），便于分批处理并减少缓存。

流程（三阶段，逐档处理）：
  1. 池化：载入既有过滤轨迹（--input-dir，如 trajectories_filtered）作为候选池，
     保留其 teacher 结构与 ce。
  2. 生成补充：对过滤统计后不足的类，用 BC base reader（--reader）在同类训练图上
     从随机节点出发生成新轨迹（argmax 贪婪，随机种子），用上界分类器验证
     （argmax == label），只保留判对的并入池。迭代直到每档每类满足候选量
     （>= --per-class-target），或达到 --max-rounds。
  3. 过滤 + 平衡：对每档的池执行「同图 sim 去重（--sim-threshold）→ 同图 ce 升序
     取 top-k（--top-k）」，随后按类平衡：类内轨迹按 ce 升序取前
     --per-class-target 条（不足则后续轮补足）。

输出 directory 与 filter_trajectories.py 兼容（trajectories_r{ratio}.pt +
config），可直接供 train_bc.py --data-dir 使用。

与 augment_trajectories.py（单类增量补额）的差异：按类精确平衡（超量裁剪、
多个目标类同时补额），且最终每图每档 ≤ --top-k。

用法:
  python code/m7/balance_trajectories.py --dataset cifar10 --root D:\\superpixels \\
      --reader m7/checkpoint/bc/ex1/base/1-teacher/reader_checkpoint_epoch6.pt \\
      --classifier m7/checkpoint/classifier/ex1/cifar10_classifier.pt \\
      --per-class-target 2000 --sim-threshold 0.8 --top-k 4 --device cuda:0
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_script_dir = Path(__file__).resolve().parent       # code/m7
_repo_root = _script_dir.parent                     # code/
sys.path.insert(0, str(_repo_root))
# 移除 code/m7 自身：dgl→torchdata 的 `import datasets` 不得解析到 code/m7/datasets
sys.path = [p for p in sys.path
            if str(Path(p).resolve()) != str(_script_dir)]

import numpy as np
import torch
import dgl
from tqdm import tqdm

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m4.teachers.base import Trajectory
from m7.datasets.registry import get_dataset
from m7.filter_trajectories import dedup_group, select_top_k
from m7.augment_trajectories import (
    build_reader, build_classifier, rollout_batch, ratio_key,
)


def count_after_filter(pool, train_labels, sim_threshold, top_k):
    """模拟「同图 dedup(0.8) → 每图 top-k」后的 per-class 计数（达标判定用）。

    与最终平衡阶段相同的过滤逻辑，避免"池计数达标但过滤后不足"的假达标。
    """
    by_graph: dict = defaultdict(list)
    for t in pool:
        by_graph[t.graph_idx].append(t)
    cnt = defaultdict(int)
    for trajs in by_graph.values():
        recs = sorted(((-t.ce, t.teacher_type, t) for t in trajs),
                      key=lambda r: r[0])
        ded = dedup_group(recs, sim_threshold)
        for traj in select_top_k(ded, top_k, require_diversity=False):
            cnt[int(train_labels[traj.graph_idx])] += 1
    return cnt


def main():
    parser = argparse.ArgumentParser(description='M7 丰富并平衡轨迹（每档每类 2000 条）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录（如 D:\\superpixels）')
    parser.add_argument('--reader', required=True,
                        help='base reader checkpoint（相对 code/，如 '
                             'm7/checkpoint/bc/ex1/base/1-teacher/reader_checkpoint_epoch6.pt）')
    parser.add_argument('--classifier', required=True,
                        help='上界 GIN checkpoint（相对 code/ 或绝对路径）')
    parser.add_argument('--budgets', type=float, nargs='+', default=None,
                        help='预算档位列表（默认 spec.budget_options；冒烟测试可传单档）')
    parser.add_argument('--input-dir', default='m7/data/trajectories_filtered',
                        help='既有过滤轨迹目录（相对 code/ 或绝对路径）')
    parser.add_argument('--output-dir', default='m7/data/rich&balanced_trajectories_filtered',
                        help='输出目录（相对 code/ 或绝对路径）')
    parser.add_argument('--per-class-target', type=int, default=2000,
                        help='每类每档目标轨迹数（默认 2000）')
    parser.add_argument('--labels', type=int, nargs='+', default=None,
                        help='本次平衡的类别列表（默认 0..n_classes-1 全部；'
                             '只对指定类做达标判定与每类截断，其余类仅按'
                             '同图去重+每图 top-k 保留）')
    parser.add_argument('--state-file', default=None,
                        help='进度持久化文件（默认 <output-dir>/balance_state.pt；'
                             '每次运行结束保存已验证新增轨迹，下次运行时自动'
                             '加载续跑，不重复生成已完成的类）')
    parser.add_argument('--sim-threshold', type=float, default=0.8,
                        help='同图去重的余弦相似度阈值')
    parser.add_argument('--top-k', type=int, default=4,
                        help='去重后每图保留最大轨迹数')
    parser.add_argument('--attempts', type=int, default=20,
                        help='每张图每档位尝试的随机种子数（k）')
    parser.add_argument('--graphs-per-round', type=int, default=1000,
                        help='每轮每类采样的训练图数')
    parser.add_argument('--max-rounds', type=int, default=10,
                        help='轮数上限（防止永不收敛）')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root or None, splits=['train'])
    spec = ds.spec
    train_graphs, train_labels = ds.train_graphs, ds.train_labels
    budgets = list(args.budgets or spec.budget_options)
    n_classes = spec.n_classes
    print(f"Dataset: {spec.name} | train={len(train_graphs)} "
          f"| classes={n_classes} | budgets={budgets}")

    # ---- 模型 ----
    r_path = Path(args.reader)
    if not r_path.is_absolute():
        r_path = _repo_root / args.reader
    reader = build_reader(torch.load(str(r_path), map_location=device,
                                     weights_only=False), spec, device)
    c_path = Path(args.classifier)
    if not c_path.is_absolute():
        c_path = _repo_root / args.classifier
    classifier = build_classifier(torch.load(str(c_path), map_location=device,
                                             weights_only=False), spec, device)
    builder = SubgraphBuilder()
    print(f"Reader: {r_path} | Classifier: {c_path}")

    by_label = defaultdict(list)
    for i in range(len(train_graphs)):
        by_label[int(train_labels[i])].append(i)

    targets = list(args.labels) if args.labels else list(range(n_classes))
    print(f"Target labels: {targets}")

    # ---- 1. 既有轨迹池（per budget ratio）+ 进度续跑 ----
    input_dir = _repo_root / args.input_dir
    output_dir = _repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file) if args.state_file else (
        output_dir / 'balance_state.pt')
    if args.state_file and not state_path.is_absolute():
        state_path = _repo_root / args.state_file

    pools = {}
    for ratio in budgets:
        key = ratio_key(ratio)
        fpath = input_dir / f'trajectories_r{key}.pt'
        if not fpath.exists():
            print(f"  WARNING: {fpath} 不存在（无既有轨迹，全靠生成）")
            pools[key] = []
            continue
        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        pool = []
        for tname, per_graph in data['trajectories'].items():
            for gi, lst in enumerate(per_graph):
                pool.extend(lst)
        pools[key] = pool
        print(f"  {key}: existing pool = {len(pool)}")

    if state_path.exists():
        state = torch.load(str(state_path), map_location='cpu',
                           weights_only=False)
        state_gen = state.get('generated', {})
        n_loaded = 0
        for key, trajs in state_gen.items():
            pools.setdefault(key, []).extend(trajs)
            n_loaded += len(trajs)
        print(f"  resume: loaded {n_loaded} generated trajectories from "
              f"{state_path}")
        # 避免续跑时从同一 seed 生成与上次完全相同的序列（去重虽能吸收，
        # 但浪费计算）：按已生成量偏移 rng
        rng = random.Random(args.seed + n_loaded)
    else:
        state_gen = {}

    # ---- 2+3. 补充生成 → 过滤统计 循环 ----
    report = {'per_class_target': args.per_class_target, 'rounds': {}}
    for round_i in range(1, args.max_rounds + 1):
        print(f"\n===== Round {round_i} =====")
        # 统计当前（每档过滤后）每类数量
        cur_counts = {}
        for ratio in budgets:
            key = ratio_key(ratio)
            cur_counts[key] = count_after_filter(
                pools[key], train_labels, args.sim_threshold, args.top_k)

        _all_done = True
        for ratio in budgets:
            key = ratio_key(ratio)
            per_label = cur_counts[key]
            print(f"  {key}: " + ", ".join(
                f"l{i}={per_label.get(i, 0)}" for i in range(n_classes))
                + f"  (target={args.per_class_target})")
            for lab in targets:
                if per_label.get(lab, 0) < args.per_class_target:
                    _all_done = False
        report['rounds'][round_i] = {
            key: dict(cur_counts[key]) for key in cur_counts}
        if _all_done:
            print("=== all target labels reach target ===")
            break

        # 生成：对缺额类采样新图，每图 attempts 个随机种子 × 全部预算档
        # （某类只要任一档未达标则继续生成；达标档也会因每图 top-k 裁剪而吸收）
        all_candidates = []  # (graph_idx, ratio, traj, state)
        for lab in targets:
            ok_everywhere = all(
                cur_counts[ratio_key(r)].get(lab, 0) >= args.per_class_target
                for r in budgets)
            if ok_everywhere:
                continue
            pool_g = by_label[lab]
            n_sample = min(args.graphs_per_round, len(pool_g))
            sampled = rng.sample(pool_g, n_sample)
            for gi in tqdm(sampled, desc=f"  gen label {lab}",
                           unit='graphs', leave=False):
                g = train_graphs[gi]
                n_nodes = g.num_nodes()
                g_states = []  # (ratio, state)
                for _try in range(args.attempts):
                    seed_node = rng.randint(0, n_nodes - 1)
                    for ratio in budgets:
                        B = max(1, int(ratio * n_nodes))
                        state = GraphState(g, seed_node=seed_node)
                        state._budget_limit = B
                        g_states.append((ratio, state))
                results = rollout_batch(reader, [s for _r, s in g_states], rng)
                for (ratio, state), (reveal_order, fin_state) in zip(
                        g_states, results):
                    if not reveal_order:
                        continue
                    traj = Trajectory(
                        graph_idx=gi,
                        seed_node=state.seed_node,
                        reveal_ratio=ratio,
                        reveal_order=reveal_order,
                        final_pred=-1,
                        is_correct=True,
                        teacher_type='gradient',
                    )
                    all_candidates.append((gi, ratio, traj, fin_state))

        # 验证（上界分类器 argmax == label），同时计算 ce
        n_valid = 0
        for c_start in range(0, len(all_candidates), args.batch_size):
            chunk = all_candidates[c_start:c_start + args.batch_size]
            sub_graphs = []
            for _gi, _ratio, _traj, state in chunk:
                sub_g, _ = builder.build(state)
                sub_graphs.append(sub_g)
            batch_g = dgl.batch(sub_graphs).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            with torch.no_grad():
                logits = classifier(batch_g, nids)
            for k, (gi, ratio, traj, state) in enumerate(chunk):
                logits_i = logits[k:k + 1]
                pred = int(logits_i.argmax(dim=-1).item())
                ce = float(torch.nn.functional.cross_entropy(
                    logits_i, torch.tensor([int(train_labels[gi])],
                                           device=device)).item())
                traj.ce = ce
                if pred == int(train_labels[gi]):
                    traj.final_pred = pred
                    pools[ratio_key(ratio)].append(traj)
                    state_gen.setdefault(ratio_key(ratio), []).append(traj)
                    n_valid += 1
        print(f"  generated {len(all_candidates)}, validated {n_valid}")
        # 每次迭代结束即持久化进度（中断不丢本轮已验证轨迹，续跑自动加载）
        torch.save({'generated': state_gen,
                    'config': {'target_labels': targets,
                               'per_class_target': args.per_class_target}},
                   str(state_path))

    # ---- 最终平衡：每档「同图去重(0.8) → 每图top4 → 每类top2000」 ----
    output_dir = _repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    balance_report = {}
    for ratio in budgets:
        key = ratio_key(ratio)
        pool = pools[key]
        # 按图分组 → dedup(0.8) → ce 升序取 top-k
        by_graph: dict = defaultdict(list)
        for t in pool:
            by_graph[t.graph_idx].append(t)
        kept_by_graph = []
        n_dedup = n_after_dedup = 0
        for g_idx, trajs in by_graph.items():
            recs = sorted(
                ((-t.ce, t.teacher_type, t) for t in trajs),
                key=lambda r: r[0])
            ded = dedup_group(recs, args.sim_threshold)
            n_dedup += len(recs) - len(ded)
            n_after_dedup += len(ded)
            kept_by_graph.extend(select_top_k(
                ded, args.top_k, require_diversity=False))

        # 每类 ce 升序取前 per_class_target（仅对 --labels 指定类截断）
        by_class: dict = defaultdict(list)
        for t in kept_by_graph:
            by_class[int(train_labels[t.graph_idx])].append(t)
        kept_final, per_class = [], {}
        for lab in range(n_classes):
            lst = sorted(by_class[lab], key=lambda t: t.ce)
            n = (min(args.per_class_target, len(lst)) if lab in targets
                 else len(lst))
            kept_final.extend(lst[:n])
            per_class[lab] = n

        # 按 teacher_type 分组还原 list-of-list 结构（兼容 filter/train_bc）
        teachers = sorted({t.teacher_type for t in kept_final})
        per_graph = defaultdict(list)
        for t in kept_final:
            per_graph[t.graph_idx].append(t)
        max_g = max(per_graph.keys(), default=-1)
        result = {tn: [[] for _ in range(max_g + 1)] for tn in teachers}
        for g_idx, lst in per_graph.items():
            for t in lst:
                result[t.teacher_type][g_idx].append(t)

        save = {
            'trajectories': result,
            'config': {
                'dataset': spec.name,
                'budget_ratio': ratio,
                'teachers': teachers,
                'per_class_target': args.per_class_target,
                'target_labels': targets,
                'top_k': args.top_k,
                'sim_threshold': args.sim_threshold,
                'balance_report': {
                    'n_pool': len(pool),
                    'n_after_dedup': n_after_dedup,
                    'dedup_removed': n_dedup,
                    'n_after_top_k': len(kept_by_graph),
                    'n_kept': len(kept_final),
                    'per_class': per_class,
                },
            },
        }
        save_path = output_dir / f'trajectories_r{key}.pt'
        torch.save(save, str(save_path))
        br = save['config']['balance_report']
        balance_report[key] = br
        print(f"\n{key}: pool={br['n_pool']} dedup_removed={br['dedup_removed']} "
              f"top-k={br['n_after_top_k']} kept={br['n_kept']}")
        print("  per-class: " + " ".join(
            f"{lab}:{n}" for lab, n in per_class.items()))
        print(f"Saved: {save_path}")

    report_path = output_dir / 'balance_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({'per_ratio': balance_report,
                   'round_stats': report}, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
