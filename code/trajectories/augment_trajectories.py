"""稀缺类轨迹增强：用 BC reader 从随机节点生成轨迹，补齐标签分布。

自 m7/augment_trajectories.py 迁入。逻辑：
  每轮对目标类别采样图 → 随机种子 × 全部预算档 lockstep rollout（argmax
  贪婪）→ 上界分类器验证（argmax == label）→ 合并进 work-dir → 过滤
  （相似去重 + 每图 top-k）→ 统计各类轨迹数，未达 --target 续轮。
  work-dir 每轮落盘 → 中断后重跑自动续用（隐式断点续跑）。

用法:
  python code/trajectories/augment_trajectories.py --dataset cifar10 \
      --root D:\\superpixels --reader result/cifar10/experiments/.../bc/*.pt \
      --classifier result/cifar10/classifier/best_cifar10_classifier.pt \
      --labels 4 --target 5000
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from tools.trajectory_io import load_trajectory_file, ratio_key  # noqa: E402
from trajectories.core.filtering import filter_file  # noqa: E402
from trajectories.core.rollout import (build_classifier, build_reader,  # noqa: E402
                                       count_labels, rollout_batch)
from trajectories.core.trajectory import Trajectory  # noqa: E402


def resolve(p: str) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def main():
    parser = argparse.ArgumentParser(description='稀缺类轨迹增强（BC reader 随机种子）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='数据集根目录（如 D:\\superpixels）')
    parser.add_argument('--reader', required=True,
                        help='reader checkpoint（bc 产物）')
    parser.add_argument('--classifier', required=True,
                        help='上界分类器 checkpoint')
    parser.add_argument('--labels', type=int, nargs='+', required=True,
                        help='要补充的稀缺类别（如 --labels 4）')
    parser.add_argument('--target', type=int, default=5000,
                        help='每类过滤后的目标轨迹总数（含既有）')
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
    parser.add_argument('--input-dir', default='',
                        help='既有过滤轨迹目录（默认 data/{dataset}/trajectories_filtered）')
    parser.add_argument('--work-dir', default='',
                        help='合并数据目录（默认 data/{dataset}/trajectories_augmented）')
    parser.add_argument('--output-dir', default='',
                        help='过滤输出目录（默认 data/{dataset}/trajectories_filtered）')
    parser.add_argument('--teacher-name', default='gradient',
                        help='新轨迹投放的教师键（与既有过滤所用教师名对齐）')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
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
    r_path = resolve(args.reader)
    reader = build_reader(torch.load(str(r_path), map_location=device,
                                     weights_only=False), device)
    c_path = resolve(args.classifier)
    classifier = build_classifier(torch.load(str(c_path), map_location=device,
                                             weights_only=False), device)
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

    # ---- 目录（默认 data/{dataset}/...）----
    input_dir = resolve(args.input_dir) if args.input_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories_filtered'
    work_dir = resolve(args.work_dir) if args.work_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories_augmented'
    output_dir = resolve(args.output_dir) if args.output_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories_filtered'
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # work-dir 存在旧合并数据时直接续用；缺失时从 input-dir 复制
    if input_dir != work_dir:
        import shutil
        copied = []
        for fpath in sorted(input_dir.glob('trajectories_r*.pt')):
            dst = work_dir / fpath.name
            if not dst.exists():
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
            base_data[key] = load_trajectory_file(fpath)
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

            # 每个 (图, 随机 seed)：与全部预算档组合建 state，
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

        # ---- 批量验证（上界分类器 argmax == label）----
        import dgl
        val_records = []
        for c_start in range(0, len(all_candidates), args.batch_size):
            chunk = all_candidates[c_start:c_start + args.batch_size]
            sub_graphs, sub_gis = [], []
            for gi, traj, state in chunk:
                sub_g, _ = builder.build(state)
                sub_graphs.append(sub_g)
                sub_gis.append(gi)
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

        # ---- 过滤（相似去重 + 每图 top-k，保留全部教师键）----
        for ratio in budgets:
            key = ratio_key(ratio)
            fpath = work_dir / f'trajectories_r{key}.pt'
            data = load_trajectory_file(fpath)
            teachers = list(data['trajectories'].keys())
            out = filter_file(fpath, teachers, args.top_k,
                              require_diversity=False,
                              sim_threshold=args.sim_threshold)
            save_path = output_dir / fpath.name
            torch.save(out['data'], str(save_path))
            r = out['report']
            all_kept = [t for tname, per_graphs in out['data']['trajectories'].items()
                        for pg in per_graphs for t in pg]
            lab_cnt = count_labels(all_kept, train_labels)
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


if __name__ == '__main__':
    main()
