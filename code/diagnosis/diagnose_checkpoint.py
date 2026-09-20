"""Reader checkpoint 诊断 — base/evidence 通用 × 任意注册数据集。

自 m5/base/analysis/diagnose_checkpoint.py（base reader）与
m6/attention/diagnose_checkpoint.py（attention reader）合并泛化：
- reader 经 train_reader.core.load_reader_from_checkpoint 从 checkpoint 自动
  重建（结构/类别以权重为准）；--reader-kind 仅作显式覆盖校验；
- 数据经 datasets 注册表加载（--dataset/--root 任意切换）；图索引语义：
  有 load_graph_subset 的数据集（PPA）= OGB train 全局索引（分层采样读
  split/标签 CSV，不加载全量图，仅加载被采样图）；其它（cifar10）=
  已加载 train split 的本地位置。

诊断项目（对应 2026-07-17 meeting notes）：
  1. 每图 rollout 成功率分布 → easy / middle / hard 分组
  2. 归一化动作熵 H(π) / log(|frontier|)  按 step 对齐
  3. 固定起点轨迹相似度（同一 seed，仅采样随机性变化）
  4. 每类准确率 + 最大类基线
  5. 配置清单（从 checkpoint 目录的 config.json 读取或自动推断）

输出：diagnosis.json + difficulty_mapping_{pct}.csv（schema 与旧版一致，
config 节新增 dataset/reader 信息）。绘图请用 visualize_diagnosis.py。

用法:
  python code/diagnosis/diagnose_checkpoint.py \
    --checkpoint result/ppa/experiments/ex1/base/rl/checkpoint_epoch4.pt \
    --classifier m3/best_model_ft.pt \
    --dataset ppa --root D:\\ogb-master\\dataset \
    --n-graphs 500 --samples-per-graph 4 --temperature 0.2
"""

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import dgl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import reveal_node  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools import cli  # noqa: E402
from tools.config import load_json, save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import SEED_METHOD_STR, deterministic_seed  # noqa: E402
from train_reader.core import (load_classifier_frozen,  # noqa: E402
                               load_reader_from_checkpoint)


def resolve(p) -> Path:
    """双基准解析（code/ 优先，兼容仓库根形式）。"""
    from tools.paths import resolve as _resolve
    return _resolve(p)


# ---------------------------------------------------------------------------
# 候选池构建（分层采样空间；不触发图数据加载）
# ---------------------------------------------------------------------------

def _ppa_csv_pool(spec):
    """PPA 类数据集：split CSV + 标签 CSV 读全量 train 候选（旧脚本口径）。"""
    import pandas as _pd
    root = Path(spec.root)
    split_csv = (root / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz')
    label_csv = (root / 'ogbg_ppa' / 'raw' / 'graph-label.csv.gz')
    candidates = _pd.read_csv(split_csv, compression='gzip', header=None
                              ).values.flatten().tolist()
    if spec.max_graphs > 0:                      # 与 ppa loader 截断语义一致
        candidates = sorted(candidates)[:spec.max_graphs]
    label_arr = _pd.read_csv(label_csv, compression='gzip', header=None
                             ).values.flatten()
    lookup = {int(i): int(label_arr[i]) for i in candidates}
    return candidates, lookup


def build_candidate_pool(spec, ds):
    """返回 (candidates, labels_lookup)。

    - 有 load_graph_subset（PPA 类，轨迹按 OGB 全局索引）：候选 = 完整
      train split 的 OGB 索引；标签读 raw CSV（不加载图数据）；
    - 其它（cifar10 类，本地位置索引）：候选 = 已加载 train_graphs 的
      本地位置；标签来自 ds.train_labels。
    """
    if spec.load_graph_subset is not None:
        return _ppa_csv_pool(spec)
    labels = ds.train_labels
    candidates = list(range(len(labels)))
    lookup = {i: int(labels[i]) for i in candidates}
    return candidates, lookup


def resolve_sampled_graphs(spec, ds, sampled):
    """sampled: [(pos, graph_idx, label)] → List[graph]（与 sampled 对齐）。

    PPA（load_graph_subset）：只加载被采样到的 OGB 索引；
    其它：直接取已加载的 train_graphs[graph_idx]（本地位置）。
    加载器已统一加自环，此处不再 add_self_loop。
    """
    if spec.load_graph_subset is not None:
        pool = spec.load_graph_subset(spec, [g for _, g, _ in sampled])
        return [pool[g][0] for _, g, _ in sampled]
    return [ds.train_graphs[g] for _, g, _ in sampled]


# ---------------------------------------------------------------------------
# 分层采样（与旧脚本同口径：按类配额 max(1, int(n*|pool|/total))，超裁缺补）
# ---------------------------------------------------------------------------

def stratified_sample_graphs(candidates, labels_lookup, n_graphs, rng,
                             n_classes):
    """按类别比例分层采样 n_graphs 张图。

    Returns: list of (local_pos, graph_idx, label)
      local_pos 为 candidates 中的位置；graph_idx 语义随数据集
      （PPA=OGB 索引 / cifar10=本地位置）。
    """
    class_to_pos = defaultdict(list)
    for pos, gidx in enumerate(candidates):
        class_to_pos[int(labels_lookup[int(gidx)])].append(pos)

    total = len(candidates)
    sampled = []
    for cls in range(n_classes):
        pool = class_to_pos.get(cls, [])
        n = max(1, int(n_graphs * len(pool) / total))
        n = min(n, len(pool))
        if n > 0:
            for pos in rng.choice(pool, size=n, replace=False):
                gidx = candidates[int(pos)]
                sampled.append((int(pos), int(gidx), cls))

    if len(sampled) > n_graphs:
        # 旧 m5 版本对 tuple 列表直接 rng.choice 会在 numpy≥某版本报
        # "a must be 1-dimensional"；统一走 m6 的索引式裁剪（等价且安全）
        keep = rng.choice(len(sampled), size=n_graphs, replace=False)
        sampled = [sampled[i] for i in keep]
    elif len(sampled) < n_graphs:
        chosen = {s[0] for s in sampled}
        remaining = [p for p in range(len(candidates)) if p not in chosen]
        n_extra = min(n_graphs - len(sampled), len(remaining))
        for pos in rng.choice(remaining, size=n_extra, replace=False):
            gidx = candidates[int(pos)]
            sampled.append((int(pos), int(gidx),
                            int(labels_lookup[int(gidx)])))
    return [(int(pos), int(gidx), int(lbl)) for pos, gidx, lbl in sampled]


# ---------------------------------------------------------------------------
# 单条采样轨迹（lockstep batch：返回正确性 + 归一化熵 + revealed 节点集）
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batched_trajectories(reader, states_info, temperature, device):
    """Lockstep batched trajectory sampling。

    所有轨迹逐 step 同步推进；每步把活动状态的扩展子图批量过
    FrontierEncoder（forward_states）。
    （attention reader 的信念状态路径随其废弃移除，2026-09-20。）

    就地修改 states_info 各项：
      - ``norm_entropies``: 每步 H(π)/log(|frontier|)
      - ``revealed``: 已揭示节点 id 集合（含 seed）
    """
    n_states = len(states_info)
    if n_states == 0:
        return

    active = [True] * n_states
    max_steps = max(si['state']._budget_limit for si in states_info)

    for _step in range(max_steps):
        active_idx = [j for j in range(n_states)
                      if active[j] and not states_info[j]['state'].is_frontier_empty]
        if not active_idx:
            break

        batch_results = reader.forward_states(
            [states_info[j]['state'] for j in active_idx],
            temperature=temperature,
        )

        for k, j in enumerate(active_idx):
            scores, probs, frontier_nodes = batch_results[k]
            if not frontier_nodes:
                active[j] = False
                continue

            si = states_info[j]
            state = si['state']
            n_f = len(frontier_nodes)

            # 归一化熵
            p = probs.detach().cpu().numpy()
            p = np.clip(p, 1e-12, 1.0)
            raw_ent = float(-np.sum(p * np.log(p)))
            si['norm_entropies'].append(
                raw_ent / math.log(n_f) if n_f > 1 else 0.0,
            )

            # 采样动作
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            chosen = frontier_nodes[action_idx.item()]
            si['revealed'].add(chosen)

            reveal_node(state, chosen)
            if state.get_budget_used() >= state._budget_limit:
                active[j] = False

    # frontier 提前为空（提前结束）的状态：补 NaN 对齐
    for j in range(n_states):
        si = states_info[j]
        remaining = si['state']._budget_limit - len(si['norm_entropies'])
        for _ in range(remaining):
            si['norm_entropies'].append(float('nan'))


# ---------------------------------------------------------------------------
# pairwise Jaccard
# ---------------------------------------------------------------------------

def compute_pairwise_jaccard(node_sets):
    """固定预算下的 pairwise Jaccard = |intersection| / budget。"""
    if len(node_sets) < 2:
        return []
    B = len(node_sets[0])
    sims = []
    for i in range(len(node_sets)):
        for j in range(i + 1, len(node_sets)):
            sims.append(len(node_sets[i] & node_sets[j]) / max(B, 1))
    return sims


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Reader checkpoint 诊断 — 冻结参数跑批次 rollout 收集全部诊断'
                    '（base/attention 自动识别）'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Reader checkpoint 路径（相对 repo root）')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt',
                        help='冻结 classifier checkpoint 路径')
    parser.add_argument('--reader-kind', choices=['base', 'evidence'], default='',
                        help='显式指定 reader 类别（默认从 checkpoint 自动识别；'
                             '与检测结果不一致时报错退出）')
    cli.add_dataset_args(parser, default_dataset='ppa')
    parser.add_argument('--n-graphs', type=int, default=500,
                        help='分析的图数量（0=全部训练集）')
    parser.add_argument('--samples-per-graph', type=int, default=10,
                        help='每张图固定起点下的采样轨迹数')
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=[0.10],
                        help='预算比例（旧脚本恒为 0.10）')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='采样温度')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 result/{dataset}/diagnosis/'
                             '{checkpoint 目录名}）')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = resolve(args.checkpoint)
    if args.output_dir:
        output_dir = resolve(args.output_dir)
    else:
        output_dir = (CODE_DIR / 'result' /
                      (args.dataset or 'dataset') / 'diagnosis' /
                      ckpt_path.parent.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 数据集（只加载 train split；PPA 候选池走 CSV 不加载图） ----
    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs)
    spec = ds.spec
    candidates, labels_lookup = build_candidate_pool(spec, ds)
    print(f"\n=== Dataset '{spec.name}' train split: {len(candidates)} graphs ===")

    # ---- 2. 分层采样 ----
    n_classes = spec.n_classes
    if 0 < args.n_graphs < len(candidates):
        sampled = stratified_sample_graphs(candidates, labels_lookup,
                                           args.n_graphs, rng, n_classes)
    else:
        sampled = [(p, int(g), labels_lookup[int(g)])
                   for p, g in enumerate(candidates)]
    print(f"=== Analyzing {len(sampled)} graphs ===")
    class_counts = defaultdict(int)
    for _, _, lbl in sampled:
        class_counts[lbl] += 1
    print(f"  Classes covered: {len(class_counts)}/{n_classes}")

    # ---- 3. 加载采样到的图（加载器已加自环） ----
    print("\n=== Loading sampled graphs ===")
    all_graphs = resolve_sampled_graphs(spec, ds, sampled)
    print(f"  Loaded {len(all_graphs)} graphs")

    # ---- 4. 加载 classifier（冻结） ----
    print("\n=== Loading classifier ===")
    clf_path = resolve(args.classifier)
    clf_ckpt = torch.load(str(clf_path), map_location='cpu', weights_only=False)
    classifier = load_classifier_frozen(clf_ckpt, device)
    print(f"  {clf_path} (val_acc={clf_ckpt.get('val_acc', 'N/A')})")

    # ---- 5. 加载 Reader（结构以权重为准，自动识别 base/evidence） ----
    print("\n=== Loading reader ===")
    rl_ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    reader = load_reader_from_checkpoint(rl_ckpt, device,
                                         stochastic=True,
                                         temperature=args.temperature)
    kind = reader.config.kind
    if args.reader_kind and args.reader_kind != kind:
        raise SystemExit(
            f"--reader-kind={args.reader_kind} 与 checkpoint 检测结果 '{kind}' 不一致")
    reader.eval()
    print(f"  kind={kind} (auto-detected), "
          f"hidden={reader.config.encoder_hidden_dims}")
    print(f"  Temperature: {args.temperature}")

    # ---- 6. 读 config.json（如果存在） ----
    ckpt_dir = ckpt_path.parent
    config_manifest = None
    config_path = ckpt_dir / 'config.json'
    if config_path.exists():
        config_manifest = load_json(config_path)
        print(f"\nConfig manifest loaded from {config_path}")
    else:
        print(f"\nWARNING: {config_path} not found — config manifest unavailable")

    # ---- 7. 批量锁步采样 + 分类 ----
    print(f"\n=== Sampling {args.samples_per_graph} trajectories per graph "
          f"(fixed seed, budgets {list(args.budget_ratios)}, batched lockstep) ===")

    builder = SubgraphBuilder()
    CHUNK_SIZE = 16  # graphs per chunk

    # 结果容器：按 budget 分组
    results_by_budget = {br: {
        'per_graph_success': [],
        'per_graph_ce': [],
        'per_class_correct': defaultdict(int),
        'per_class_total': defaultdict(int),
        'all_similarities': [],
        'all_norm_entropies': [],
        'graph_info': [],
    } for br in args.budget_ratios}

    for chunk_start in tqdm(range(0, len(sampled), CHUNK_SIZE), desc="  Graphs"):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(sampled))
        chunk = sampled[chunk_start:chunk_end]

        for budget_ratio in args.budget_ratios:
            # -- create all states for this (chunk, budget) --
            states_info = []
            for gi, (_local_i, gidx, label) in enumerate(chunk):
                g = all_graphs[chunk_start + gi]
                n_nodes = g.num_nodes()
                B = max(1, int(budget_ratio * n_nodes))
                seed_node = deterministic_seed(gidx, 0, n_nodes)

                for _s in range(args.samples_per_graph):
                    state = GraphState(g, seed_node=seed_node)
                    state._budget_limit = B
                    states_info.append({
                        'state': state,
                        'graph_gi': gi,
                        'norm_entropies': [],
                        'revealed': {seed_node},
                    })

            # -- run all trajectories in lockstep --
            run_batched_trajectories(reader, states_info, args.temperature, device)

            # -- batch classify all subgraphs --
            for c_start in range(0, len(states_info), 512):
                c_end = min(c_start + 512, len(states_info))
                chunk_si = states_info[c_start:c_end]

                batch_g = dgl.batch([
                    builder.build(si['state'])[0]
                    for si in chunk_si
                ]).to(device)
                nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long,
                                   device=device)
                logits = classifier(batch_g, nids)

                for j, si in enumerate(chunk_si):
                    pred = logits[j].argmax(dim=-1).item()
                    si['correct'] = (pred == chunk[si['graph_gi']][2])
                    y = torch.tensor([chunk[si['graph_gi']][2]],
                                     device=logits.device)
                    si['ce'] = float(F.cross_entropy(logits[j:j + 1], y).item())

            # -- aggregate per-graph results --
            r = results_by_budget[budget_ratio]
            for gi in range(len(chunk)):
                _local_i, gidx, label = chunk[gi]
                n_nodes = all_graphs[chunk_start + gi].num_nodes()
                gi_sis = [si for si in states_info if si['graph_gi'] == gi]
                correct_list = [si['correct'] for si in gi_sis]
                ce_list = [si.get('ce', 0.0) for si in gi_sis]

                success_rate = sum(correct_list) / len(correct_list)
                mean_ce = float(np.mean(ce_list)) if ce_list else 0.0
                sims = compute_pairwise_jaccard(
                    [si['revealed'] for si in gi_sis])

                r['per_graph_success'].append(success_rate)
                r['per_graph_ce'].append(mean_ce)
                r['all_similarities'].append(sims)
                r['all_norm_entropies'].extend(
                    [si['norm_entropies'] for si in gi_sis],
                )
                r['graph_info'].append((gidx, label, n_nodes))
                r['per_class_correct'][label] += sum(correct_list)
                r['per_class_total'][label] += len(correct_list)

    # ---- 8. 汇总统计 ----
    print(f"\n{'=' * 80}")
    print("=== Diagnosis Results ===")
    print(f"{'=' * 80}")

    final_results = {
        'config': {
            'checkpoint': str(ckpt_path),
            'classifier': str(clf_path),
            'dataset': spec.name,
            'root': args.root,
            'max_graphs': args.max_graphs,
            'reader_kind': kind,
            'reader_config': reader.config.to_dict(),
            'n_graphs': len(sampled),
            'samples_per_graph': args.samples_per_graph,
            'budget_ratios': list(args.budget_ratios),
            'temperature': args.temperature,
            'seed': args.seed,
            'seed_method': SEED_METHOD_STR,
            'device': str(device),
            'config_manifest': config_manifest,
        },
        'by_budget': {},
    }

    for budget_ratio in args.budget_ratios:
        r = results_by_budget[budget_ratio]

        # Easy / middle / hard
        successes = r['per_graph_success']
        mean_ces = r['per_graph_ce']
        graph_indices = r['graph_info']  # list of (graph_idx, label, n_nodes)

        easy_list = []
        middle_list = []
        hard_list = []
        all_entries = []
        for i, v in enumerate(successes):
            gidx, label, n_nodes = graph_indices[i]
            level = 'easy' if v > 0.8 else ('middle' if v >= 0.2 else 'hard')
            entry = {'ogb_idx': gidx, 'label': label, 'n_nodes': n_nodes,
                     'success_rate': round(v, 4), 'level': level,
                     'mean_ce': round(mean_ces[i], 4)}
            all_entries.append(entry)
            if v > 0.8:
                easy_list.append(entry)
            elif 0.2 <= v <= 0.8:
                middle_list.append(entry)
            else:
                hard_list.append(entry)

        easy, middle, hard = len(easy_list), len(middle_list), len(hard_list)
        n_graphs = len(successes)

        # 每类准确率
        per_class_acc = {}
        for cls in sorted(r['per_class_total'].keys()):
            per_class_acc[str(cls)] = (r['per_class_correct'][cls] /
                                       r['per_class_total'][cls])

        # 最大类基线（按样本数加权）
        largest_class = max(r['per_class_total'], key=r['per_class_total'].get)
        largest_class_acc = (r['per_class_correct'][largest_class] /
                             r['per_class_total'][largest_class])

        # 相似度
        all_flat_sims = [s for sims in r['all_similarities'] for s in sims]
        mean_sim = float(np.mean(all_flat_sims)) if all_flat_sims else 0.0
        std_sim = float(np.std(all_flat_sims)) if all_flat_sims else 0.0

        # 归一化熵
        ent_means = []
        max_steps = (max(len(e) for e in r['all_norm_entropies'])
                     if r['all_norm_entropies'] else 0)
        for step in range(max_steps):
            vals = [e[step] for e in r['all_norm_entropies']
                    if step < len(e) and not math.isnan(e[step])]
            ent_means.append(float(np.mean(vals)) if vals else None)

        budget_pct = f"{int(budget_ratio * 100)}%"
        print(f"\n--- Budget {budget_pct} ---")
        print(f"  Graphs: {n_graphs}")
        print(f"  Easy (>80%):  {easy:4d} ({easy / max(n_graphs, 1) * 100:5.1f}%)")
        print(f"  Middle:       {middle:4d} ({middle / max(n_graphs, 1) * 100:5.1f}%)")
        print(f"  Hard (<20%):  {hard:4d} ({hard / max(n_graphs, 1) * 100:5.1f}%)")
        print(f"  Mean success: {float(np.mean(successes)):.4f}")
        print(f"  Fixed-start similarity: {mean_sim:.4f} ± {std_sim:.4f}")
        print(f"  Norm entropy (step avg): {ent_means}")
        print(f"  Largest class (c{largest_class}): acc={largest_class_acc:.4f}")
        print(f"  Classes with data: {len(per_class_acc)}")

        final_results['by_budget'][budget_pct] = {
            'accuracy_description': (
                "Per-graph success_rate = fraction of sampled trajectories (same fixed "
                "seed node, different Categorical sampling noise) that were classified "
                "correctly by the frozen classifier. Difficulty bins: "
                "easy = success_rate > 0.8, middle = 0.2 ≤ success_rate ≤ 0.8, "
                "hard = success_rate < 0.2."
            ),
            'n_graphs': n_graphs,
            'easy_count': easy,
            'middle_count': middle,
            'hard_count': hard,
            'easy_ratio': easy / max(n_graphs, 1),
            'middle_ratio': middle / max(n_graphs, 1),
            'hard_ratio': hard / max(n_graphs, 1),
            'mean_success_rate': float(np.mean(successes)),
            'fixed_start_similarity_mean': mean_sim,
            'fixed_start_similarity_std': std_sim,
            'norm_entropy_by_step': ent_means,
            'per_class_acc': per_class_acc,
            'largest_class': int(largest_class),
            'largest_class_acc': largest_class_acc,
        }

        csv_path = output_dir / f'difficulty_mapping_{budget_pct}.csv'
        pd.DataFrame(all_entries).to_csv(str(csv_path), index=False)
        print(f"  Difficulty mapping saved: {csv_path}")

    # ---- 9. 保存结果 ----
    output_path = output_dir / 'diagnosis.json'
    save_json(final_results, output_path)
    print(f"\nResults saved to {output_path}")

    # 同时输出一个紧凑的对比表
    print(f"\n{'=' * 80}")
    print("Summary Table")
    print(f"{'=' * 80}")
    header = (f"{'Budget':>6}  {'Easy%':>7}  {'Mid%':>7}  {'Hard%':>7}  "
              f"{'Succ':>6}  {'Sim':>7}  {'H_norm(1st)':>12}")
    print(header)
    print("-" * len(header))
    for budget_ratio in args.budget_ratios:
        bp = f"{int(budget_ratio * 100)}%"
        d = final_results['by_budget'][bp]
        h_first = d['norm_entropy_by_step'][0] if d['norm_entropy_by_step'] else None
        print(f"{bp:>6}  {d['easy_ratio']:6.3f}  {d['middle_ratio']:6.3f}  "
              f"{d['hard_ratio']:6.3f}  "
              f"{d['mean_success_rate']:5.3f}  "
              f"{d['fixed_start_similarity_mean']:6.3f}  "
              f"{h_first if h_first is not None else 'N/A':>12}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == '__main__':
    main()
