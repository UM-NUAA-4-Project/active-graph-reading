"""M7 稀缺类轨迹增强：用 BC base reader 从随机节点生成轨迹，补齐标签分布。

背景：初代过滤轨迹（trajectories_filtered）类别极不均衡（如 label 4 仅 1621 条 vs
label 8 过万）。本脚本用已训练好的 base reader（checkpoint/bc/ex1/base/1-teacher/
reader_checkpoint_epoch6.pt）在指定类别图上从随机节点出发生成轨迹（argmax 贪婪阅读），
与既有过滤轨迹合并后走与 filter_trajectories.py 相同的过滤流程
（相似去重 + 每图 top-k）。每轮结束后统计过滤结果中每个目标类的轨迹总数
（含既有），未达到 --target 则采样新图续轮重试。新增轨迹计入
梯度教师（--teacher-name 对齐初代过滤所用的教师键）。

与 generate_trajectories.py（teacher 分数选 top-n seed）的差异：
  - seed 节点不再用确定性公式，而是真正的随机种子（每轮 RNG 由 --seed 控制）
  - 轨迹由 BC reader 的 argmax 策略产生（而非 teacher score/greedy）
  - 只保留被上界分类器判对（is_correct）的轨迹
  - graph_idx 语义 = ds.train_graphs 本地位置（与 train_bc / train_rl 一致）

用法:
  python code/m7/augment_trajectories.py --dataset cifar10 --root D:\\superpixels \\
      --reader m7/checkpoint/bc/ex1/base/1-teacher/reader_checkpoint_epoch6.pt \\
      --classifier m7/checkpoint/classifier/finetune/best_cifar10_classifier.pt \\
      --labels 4 --target 5000 --device cuda:0
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

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m4.teachers.base import Trajectory
from m7.datasets.registry import get_dataset
from m7.filter_trajectories import filter_file


def ratio_key(r: float) -> str:
    """0.05 -> '0_050'（与 generate_trajectories.py 一致）。"""
    return f'{r:.3f}'.replace('.', '_')


def build_reader(ckpt: dict, spec, device) -> LearnableReader:
    """按 checkpoint 自身 config 构建 base reader（同 train_bc.py 模式）。"""
    cfg = ckpt.get('config', {})
    reader = LearnableReader(
        edge_feat_dim=cfg.get('edge_feat_dim', spec.edge_feat_dim),
        hidden_dim=cfg.get('hidden_dim', 128),
        n_layers=cfg.get('n_layers_reader', cfg.get('n_layers', 2)),
        dropout=cfg.get('dropout_reader', cfg.get('dropout', 0.25)),
        node_feat_dim=cfg.get('node_feat_dim', spec.node_feat_dim),
    ).to(device)
    reader.load_state_dict(ckpt['reader_state_dict'], strict=True)
    reader.eval()
    return reader


def build_classifier(ckpt: dict, spec, device) -> GraphClassifier:
    """按 checkpoint 自身 config 构建上界 GIN（同 generate_trajectories.py）。"""
    cfg = ckpt.get('config', {})
    clf = GraphClassifier(
        edge_feat_dim=cfg.get('edge_feat_dim', spec.edge_feat_dim),
        n_classes=cfg.get('n_classes', spec.n_classes),
        hidden_dim=cfg.get('hidden_dim', 300),
        n_layers=cfg.get('n_layers', 5),
        dropout=cfg.get('dropout', 0.5),
        node_feat_dim=cfg.get('node_feat_dim', spec.node_feat_dim),
    ).to(device)
    clf.load_state_dict(ckpt['model_state_dict'], strict=True)
    clf.eval()
    return clf


def rollout_batch(reader, states: list, rng: random.Random) -> list:
    """跨图 lockstep 批量 rollout（同 train_bc.py 训练模式）。

    states: 已设置 seed 与 _budget_limit 的 GraphState 列表。
    返回与 states 等长的 list：(reveal_order, final_state)，无动作的为 ([], state)。
    """
    n_total = len(states)
    orders = [[] for _ in range(n_total)]
    active = [True] * n_total
    max_budget = max(s._budget_limit for s in states)
    for _step in range(max_budget):
        active_idx = [j for j in range(n_total)
                      if active[j] and not states[j].is_frontier_empty]
        if not active_idx:
            break
        batch = reader.forward_states([states[j] for j in active_idx],
                                      temperature=reader.temperature)
        for k, j in enumerate(active_idx):
            _scores, probs, frontier_nodes = batch[k]
            if not frontier_nodes:
                active[j] = False
                continue
            node = frontier_nodes[int(probs.argmax().item())]
            orders[j].append(node)
            reveal_node(states[j], node)
            if states[j].get_budget_used() >= states[j]._budget_limit:
                active[j] = False
    return [(o, s) for o, s in zip(orders, states)]


def count_labels(trajs: list, train_labels) -> dict:
    """按 label 统计轨迹数（graph_idx → 本地位置 → ds.train_labels）。"""
    cnt = defaultdict(int)
    for t in trajs:
        cnt[int(train_labels[t.graph_idx])] += 1
    return dict(cnt)


def main():
    parser = argparse.ArgumentParser(description='M7 稀缺类轨迹增强（BC reader 随机种子）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录（如 D:\\superpixels）')
    parser.add_argument('--reader', required=True,
                        help='base reader checkpoint（相对 code/，如 '
                             'm7/checkpoint/bc/ex1/base/1-teacher/reader_checkpoint_epoch6.pt）')
    parser.add_argument('--classifier', required=True,
                        help='上界 GIN checkpoint（相对 code/ 或绝对路径）')
    parser.add_argument('--labels', type=int, nargs='+', required=True,
                        help='要补充的稀缺类别（如 --labels 4）')
    parser.add_argument('--target', type=int, default=5000,
                        help='每类过滤后的目标轨迹总数（含既有，如 --target 5000 '
                             '= 该类轨迹补到 5000；未达则继续采样续轮）')
    parser.add_argument('--attempts', type=int, default=20,
                        help='每张图每档位尝试的随机种子数（k）')
    parser.add_argument('--graphs-per-label', type=int, default=1000,
                        help='每轮每类采样的训练图数')
    parser.add_argument('--max-rounds', type=int, default=10,
                        help='轮数上限（防止永不收敛）')
    parser.add_argument('--sim-threshold', type=float, default=0.75,
                        help='同图去重的余弦相似度阈值')
    parser.add_argument('--top-k', type=int, default=5,
                        help='去重后每图保留最大轨迹数')
    parser.add_argument('--input-dir', default='m7/data/trajectories_filtered',
                        help='既有过滤轨迹目录（相对 code/ 或绝对路径）')
    parser.add_argument('--work-dir', default='m7/data/trajectories_augmented',
                        help='合并数据目录（相对 code/ 或绝对路径，供过滤读取）')
    parser.add_argument('--output-dir', default='m7/data/trajectories_filtered',
                        help='过滤输出目录（相对 code/ 或绝对路径）')
    parser.add_argument('--teacher-name', default='gradient',
                        help='新轨迹投放的教师键（与初代过滤所用的教师名对齐）')
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
    budgets = list(spec.budget_options)
    print(f"Dataset: {spec.name} | train={len(train_graphs)} "
          f"| classes={spec.n_classes} | budgets={budgets}")

    # ---- 模型 ----
    r_path = Path(args.reader)
    if not r_path.is_absolute():
        r_path = _repo_root / args.reader
    reader_ckpt = torch.load(str(r_path), map_location=device, weights_only=False)
    reader = build_reader(reader_ckpt, spec, device)

    c_path = Path(args.classifier)
    if not c_path.is_absolute():
        c_path = _repo_root / args.classifier
    cls_ckpt = torch.load(str(c_path), map_location=device, weights_only=False)
    classifier = build_classifier(cls_ckpt, spec, device)
    builder = SubgraphBuilder()
    print(f"Reader: {r_path} | Classifier: {c_path}")

    # ---- 按 label 分组的训练图池 ----
    by_label = defaultdict(list)
    for i in range(len(train_graphs)):
        by_label[int(train_labels[i])].append(i)
    for lab in args.labels:
        cnt = len(by_label.get(lab, []))
        print(f"label {lab}: {cnt} train graphs")
        if cnt == 0:
            raise SystemExit(f"训练集中无 label {lab}")

    # ---- 构建 per-ratio 的既有过滤轨迹（graph_idx 本地位置） ----
    input_dir = _repo_root / args.input_dir
    work_dir = _repo_root / args.work_dir
    output_dir = _repo_root / args.output_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # work-dir 存在旧合并数据时直接续用；缺失时从 input-dir 复制
    if input_dir != work_dir:
        copied = []
        for fpath in sorted(input_dir.glob('trajectories_r*.pt')):
            dst = work_dir / fpath.name
            if not dst.exists():
                import shutil
                shutil.copy2(fpath, dst)
                copied.append(fpath.name)
        print(f"  work-dir: {'续用已有合并数据' if not copied else f'从 input-dir 复制 {copied}'}")
    else:
        print("  input-dir == work-dir, 直接在原目录上迭代")

    # ---- 每轮：采样图 → 随机种子生成 → 验证过滤 → 合并过滤 → 统计 ----
    report = {'target': args.target, 'labels': args.labels, 'rounds': {}}
    for round_i in range(1, args.max_rounds + 1):
        # base_data：本轮起点 = 上轮合并并保存后的 work-dir 文件
        base_data = {}
        ok_ratios = []
        for ratio in budgets:
            key = ratio_key(ratio)
            fpath = work_dir / f'trajectories_r{key}.pt'
            if not fpath.exists():
                print(f"  WARNING: {fpath} 不存在，跳过该档位"
                      "（若确需补该档先复制或重新生成）")
                continue
            base_data[key] = torch.load(str(fpath), map_location='cpu',
                                        weights_only=False)
            ok_ratios.append(ratio)
        if not ok_ratios:
            raise SystemExit("work-dir 中无任何轨迹文件，无法增强")
        budgets = ok_ratios
        print(f"\n===== Round {round_i} =====")
        all_candidates = []  # (graph_idx, traj, state)

        for lab in args.labels:
            pool = by_label[lab]
            n_sample = min(args.graphs_per_label, len(pool))
            sampled = rng.sample(pool, n_sample)
            print(f"  label {lab}: sampling {len(sampled)}/{len(pool)} graphs "
                  f"(attempts={args.attempts})")

            # 每个 (图, 随机 seed)：与 3 个预算档组合建 state，
            # 图内全部 (seed × ratio) 一次 lockstep 批量 rollout
            for gi in tqdm(sampled, desc=f"  label {lab}"):
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
                for (ratio, state), (reveal_order, fin_state) in zip(g_states, results):
                    if not reveal_order:
                        continue
                    traj = Trajectory(
                        graph_idx=gi,
                        seed_node=state.seed_node,
                        reveal_ratio=ratio,
                        reveal_order=reveal_order,
                        final_pred=-1,
                        is_correct=True,  # 初步标记，之后由分类器验证
                        teacher_type=args.teacher_name,
                    )
                    all_candidates.append((gi, traj, fin_state))

        if not all_candidates:
            print("  no candidates generated")
            break

        # ---- 批量验证（上界分类器 argmax == label） ----
        val_records = []
        for c_start in range(0, len(all_candidates), args.batch_size):
            chunk = all_candidates[c_start:c_start + args.batch_size]
            sub_graphs, sub_labels = [], []
            for gi, traj, state in chunk:
                sub_g, _ = builder.build(state)
                sub_graphs.append(sub_g)
                sub_labels.append(traj.graph_idx)
            batch_g = dgl.batch(sub_graphs).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            with torch.no_grad():
                logits = classifier(batch_g, nids)
            for k in range(len(chunk)):
                gi, traj, state = chunk[k]
                pred = int(logits[k].argmax(dim=-1).item())
                if pred == int(train_labels[gi]):
                    traj.final_pred = pred
                    val_records.append((gi, traj))
        print(f"  validated: {len(val_records)}/{len(all_candidates)} correct")

        # ---- 新增池 per-ratio 保存到 work-dir 供过滤 ----
        new_by_ratio = defaultdict(list)
        for gi, traj in val_records:
            new_by_ratio[ratio_key(traj.reveal_ratio)].append((gi, traj))
        for key, items in new_by_ratio.items():
            data = base_data[key]
            # 合并进每 teacher 的 per_graph list（对齐 filter 期望的 list-of-list）
            for tname, per_graph in data['trajectories'].items():
                for gi, traj in items:
                    if gi < len(per_graph):
                        per_graph[gi].append(traj)
                    else:
                        while len(per_graph) <= gi:
                            per_graph.append([])
                        per_graph[gi].append(traj)
            fpath = work_dir / f'trajectories_r{key}.pt'
            torch.save(data, str(fpath))

        # ---- 过滤（相似去重 + 每图 top-k + 教师多样性，保留全部教师键） ----
        for ratio in budgets:
            key = ratio_key(ratio)
            fpath = work_dir / f'trajectories_r{key}.pt'
            data = torch.load(str(fpath), map_location='cpu', weights_only=False)
            teachers = list(data['trajectories'].keys())
            out = filter_file(fpath, teachers, args.top_k,
                              require_diversity=False,
                              sim_threshold=args.sim_threshold)
            save_path = output_dir / fpath.name
            torch.save(out['data'], str(save_path))
            r = out['report']
            lab_cnt = count_labels(
                [t for tname, per_graphs in out['data']['trajectories'].items()
                 for pg in per_graphs for t in pg],
                train_labels,
            )
            print(f"  {key}: {r['n_before']} -> dedup {r['n_after_dedup']} "
                  f"(removed {r['dedup_removed']}) -> top-k {r['n_after']} "
                  f"(graphs {r['n_graphs_after']})")
            print("    per-label: " + ", ".join(
                f"l{lab}={lab_cnt.get(lab, 0)}" for lab in args.labels)
                + f" | target={args.target}")

            report['rounds'].setdefault(round_i, {})[key] = {
                **r, 'label_counts': lab_cnt,
            }

        # ---- 判定：所有目标类均达到 target 则停止 ----
        done = True
        for ratio in budgets:
            key = ratio_key(ratio)
            lab_cnt = report['rounds'][round_i][key]['label_counts']
            for lab in args.labels:
                if lab_cnt.get(lab, 0) < args.target:
                    done = False
        if done:
            print(f"\n=== All target labels reached >= {args.target} ===")
            break

    report_path = output_dir / 'augment_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {report_path}")
    print(f"下一步: python code/m7/train_bc.py --data-dir m7/data/trajectories_filtered")


if __name__ == '__main__':
    main()
