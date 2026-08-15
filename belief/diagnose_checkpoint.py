"""加载 reader checkpoint，在训练集上用冻结参数跑批次 rollout，收集 Required Diagnostics。

诊断项目（对应 2026-07-17 meeting notes）：
  1. 每图 rollout 成功率分布 → easy / middle / hard 分组
  2. 归一化动作熵 H(π) / log(|frontier|)  按 step 对齐
  3. 固定起点轨迹相似度（同一 seed，仅采样随机性变化）
  4. 每类准确率 + 最大类基线
  5. 配置清单（从 checkpoint 目录的 config.json 读取或自动推断）

信念族版（belief/ Phase-2 reader）：lockstep 中管理信念状态（init_beliefs +
reveal 后 NewInfoEncoder 编码 + update_belief）；reader 自带分类头，其权重随
checkpoint 一起加载（不 strip）。

用法:
  python belief/diagnose_checkpoint.py \
    --checkpoint belief/rl/belief/checkpoint_epoch5.pt \
    --classifier code/m3/best_model_ft.pt \
    --n-graphs 0 --samples-per-graph 4 \
    --temperature 0.2 \
    --output-dir belief/diagnosis
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
sys.path.insert(0, str(_repo_root / 'code' / 'm5'))
sys.path.insert(0, str(_repo_root / 'code'))
sys.path.insert(0, str(_script_dir))

from learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from m5.training_utils import infer_reader_config, patch_old_checkpoint
from training_utils import reveal_and_update_beliefs_batch

K = 37
BUDGET_OPTIONS = [0.10]


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


# ---------------------------------------------------------------------------
# 分层采样
# ---------------------------------------------------------------------------

def stratified_sample_graphs(all_indices, labels, n_graphs, rng):
    class_to_local = defaultdict(list)
    for i, ogb_idx in enumerate(all_indices):
        lbl = int(labels[ogb_idx].item())
        class_to_local[lbl].append(i)

    total = len(all_indices)
    sampled = []
    for cls in range(K):
        pool = class_to_local.get(cls, [])
        n = max(1, int(n_graphs * len(pool) / total))
        n = min(n, len(pool))
        if n > 0:
            for local_i in rng.choice(pool, size=n, replace=False):
                sampled.append((int(local_i), int(all_indices[local_i]), cls))

    if len(sampled) > n_graphs:
        idx = rng.choice(len(sampled), size=n_graphs, replace=False)
        sampled = [sampled[i] for i in idx]
    elif len(sampled) < n_graphs:
        remaining = [i for i in range(len(all_indices)) if i not in {s[0] for s in sampled}]
        n_extra = min(n_graphs - len(sampled), len(remaining))
        for local_i in rng.choice(remaining, size=n_extra, replace=False):
            sampled.append((int(local_i), int(all_indices[local_i]),
                            int(labels[all_indices[local_i]].item())))
    return [(int(local_i), int(ogb_idx), int(lbl)) for local_i, ogb_idx, lbl in sampled]


# ---------------------------------------------------------------------------
# 单条采样轨迹（返回正确性 + 归一化熵 + revealed 节点集）
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batched_trajectories(reader, states_info, temperature, device):
    """Lockstep batched trajectory sampling with belief management.

    All trajectories in *states_info* advance step-by-step together.  At each
    step the extended subgraphs of all active states are batched through the
    FrontierEncoder GNN (forward_batch) with their current beliefs; after
    reveals, beliefs are batch-updated via NewInfoEncoder (same pattern as
    belief/train_bc.py Pass 2).

    Modifies each entry of *states_info* in-place by adding:
      - ``norm_entropies``: list of H(π)/log(|frontier|) per step
      - ``revealed``: set of revealed node ids (including seed)
    """
    n_states = len(states_info)
    if n_states == 0:
        return

    active = [True] * n_states
    max_steps = max(si['state']._budget_limit for si in states_info)
    all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_states)]
    all_subgraphs = [None] * n_states

    for _step in range(max_steps):
        active_idx = [j for j in range(n_states)
                      if active[j] and not states_info[j]['state'].is_frontier_empty]
        if not active_idx:
            break

        active_beliefs = torch.cat([all_beliefs[j] for j in active_idx], dim=0)
        active_subgraphs = None
        if all_subgraphs[active_idx[0]] is not None:
            active_subgraphs = [all_subgraphs[j] for j in active_idx]

        batch_results = reader.forward_states(
            [states_info[j]['state'] for j in active_idx],
            temperature=temperature,
            beliefs=active_beliefs,
            subgraphs=active_subgraphs,
        )

        batch_states = []
        batch_nodes = []
        batch_local_idx = []

        for k, j in enumerate(active_idx):
            scores, probs, frontier_nodes, logits = batch_results[k]
            if not frontier_nodes:
                active[j] = False
                continue

            si = states_info[j]
            state = si['state']
            n_f = len(frontier_nodes)

            # norm entropy
            p = probs.detach().cpu().numpy()
            p = np.clip(p, 1e-12, 1.0)
            raw_ent = float(-np.sum(p * np.log(p)))
            si['norm_entropies'].append(
                raw_ent / math.log(n_f) if n_f > 1 else 0.0,
            )

            # sample action
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            chosen = frontier_nodes[action_idx.item()]
            si['revealed'].add(chosen)

            batch_states.append(state)
            batch_nodes.append(chosen)
            batch_local_idx.append(j)

        # batch reveal + encode new info + update beliefs
        if batch_states:
            batch_beliefs = torch.cat(
                [all_beliefs[j] for j in batch_local_idx], dim=0)
            new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                reader, batch_states, batch_nodes, batch_beliefs, device,
            )
            for k, j in enumerate(batch_local_idx):
                all_beliefs[j] = new_beliefs[k:k + 1]
                all_subgraphs[j] = new_subgraphs[k]

        for j in active_idx:
            if active[j] and states_info[j]['state'].get_budget_used() \
                    >= states_info[j]['state']._budget_limit:
                active[j] = False

    # pad norm_entropies for states that finished early (frontier emptied)
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
        description='RL checkpoint diagnosis — 冻结参数跑批次 rollout 收集全部诊断'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Reader checkpoint 路径（相对于 repo root）')
    parser.add_argument('--classifier', default='code/m3/best_model_ft.pt',
                        help='冻结 classifier checkpoint 路径')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-graphs', type=int, default=500,
                        help='分析的图数量（0=全部训练集）')
    parser.add_argument('--samples-per-graph', type=int, default=10,
                        help='每张图固定起点下的采样轨迹数')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='采样温度')
    parser.add_argument('--m', type=int, default=4,
                        help='belief token 数量（checkpoint config 缺失时的兜底）')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='attention heads 数（checkpoint config 缺失时的兜底）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='belief/diagnosis')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读训练集 split + 标签 ----
    root_path = Path(args.root)
    train_csv = root_path / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'
    all_train_indices = pd.read_csv(train_csv, compression='gzip', header=None
                                    ).values.flatten().tolist()
    print(f"\n=== Train split: {len(all_train_indices)} graphs ===")

    label_csv = root_path / 'ogbg_ppa' / 'raw' / 'graph-label.csv.gz'
    label_arr = pd.read_csv(label_csv, compression='gzip', header=None
                            ).values.flatten()

    # ---- 2. 分层采样 ----
    if args.n_graphs > 0 and args.n_graphs < len(all_train_indices):
        sampled = stratified_sample_graphs(all_train_indices, label_arr,
                                           args.n_graphs, rng)
    else:
        sampled = [(i, int(ogb_idx), int(label_arr[ogb_idx].item()))
                   for i, ogb_idx in enumerate(all_train_indices)]
    print(f"=== Analyzing {len(sampled)} graphs ===")
    class_counts = defaultdict(int)
    for _, _, lbl in sampled:
        class_counts[lbl] += 1
    print(f"  Classes covered: {len(class_counts)}/{K}")

    # ---- 3. 加载图 ----
    sampled_indices = [ogb_idx for _, ogb_idx, _ in sampled]
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    all_graphs, all_labels_tensor = dgl.load_graphs(processed_path, idx_list=sampled_indices)
    print(f"  Loaded {len(all_graphs)} graphs")
    all_graphs = [dgl.add_self_loop(g) for g in all_graphs]

    # ---- 4. 加载 classifier（冻结） ----
    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt_path = repo_root / args.classifier
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"  val_acc={ckpt.get('val_acc', 'N/A')}")

    # ---- 5. 加载 Reader ----
    print("\n=== Loading LearnableReader ===")
    ckpt_path = repo_root / args.checkpoint
    rl_ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    reader_sd = rl_ckpt.get('reader_state_dict', rl_ckpt)
    reader_sd, was_patched = patch_old_checkpoint(reader_sd)
    hidden_dim, n_layers = infer_reader_config(reader_sd)

    ckpt_config = rl_ckpt.get('config', {}) if isinstance(rl_ckpt, dict) else {}
    m_val = int(ckpt_config.get('m', args.m))
    n_heads_val = int(ckpt_config.get('n_heads', args.n_heads))

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=0.1, stochastic=True, temperature=args.temperature,
        m=m_val, n_heads=n_heads_val,
    ).to(device)
    missing, unexpected = reader.load_state_dict(reader_sd, strict=False)
    if missing:
        print(f"  Missing keys (random init): {missing}")
    if unexpected:
        print(f"  Unused keys: {unexpected}")
    reader.eval()
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, "
          f"m={m_val}, n_heads={n_heads_val}")
    print(f"  Temperature: {args.temperature}")

    # ---- 6. 读 config.json（如果存在） ----
    ckpt_dir = (repo_root / args.checkpoint).parent
    config_manifest = None
    config_path = ckpt_dir / 'config.json'
    if config_path.exists():
        with open(config_path, encoding='utf-8') as f:
            config_manifest = json.load(f)
        print(f"\nConfig manifest loaded from {config_path}")
    else:
        print(f"\nWARNING: {config_path} not found — config manifest unavailable")

    # ---- 7. 批量锁步采样 + 分类 ----
    print(f"\n=== Sampling {args.samples_per_graph} trajectories per graph "
          f"(fixed seed, budgets {BUDGET_OPTIONS}, batched lockstep) ===")

    builder = SubgraphBuilder()
    CHUNK_SIZE = 16  # graphs per chunk

    # 结果容器：按 budget 分组
    results_by_budget = {}
    for br in BUDGET_OPTIONS:
        results_by_budget[br] = {
            'per_graph_success': [],
            'per_graph_ce': [],
            'per_class_correct': defaultdict(int),
            'per_class_total': defaultdict(int),
            'all_similarities': [],
            'all_norm_entropies': [],
            'graph_info': [],
        }

    for chunk_start in tqdm(range(0, len(sampled), CHUNK_SIZE), desc="  Graphs"):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(sampled))
        chunk = sampled[chunk_start:chunk_end]

        for budget_ratio in BUDGET_OPTIONS:
            # -- create all states for this (chunk, budget) --
            states_info = []
            for gi, (_local_i, ogb_idx, label) in enumerate(chunk):
                g = all_graphs[chunk_start + gi]
                n_nodes = g.num_nodes()
                B = max(1, int(budget_ratio * n_nodes))
                seed_node = deterministic_seed(ogb_idx, 0, n_nodes)

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
                nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
                logits = classifier(batch_g, nids)

                for j, si in enumerate(chunk_si):
                    pred = logits[j].argmax(dim=-1).item()
                    si['correct'] = (pred == chunk[si['graph_gi']][2])
                    y = torch.tensor([chunk[si['graph_gi']][2]], device=logits.device)
                    si['ce'] = float(F.cross_entropy(logits[j:j + 1], y).item())

            # -- aggregate per-graph results --
            r = results_by_budget[budget_ratio]
            for gi in range(len(chunk)):
                _local_i, ogb_idx, label = chunk[gi]
                n_nodes = all_graphs[chunk_start + gi].num_nodes()
                gi_sis = [si for si in states_info if si['graph_gi'] == gi]
                correct_list = [si['correct'] for si in gi_sis]
                ce_list = [si.get('ce', 0.0) for si in gi_sis]

                success_rate = sum(correct_list) / len(correct_list)
                mean_ce = float(np.mean(ce_list)) if ce_list else 0.0
                sims = compute_pairwise_jaccard([si['revealed'] for si in gi_sis])

                r['per_graph_success'].append(success_rate)
                r['per_graph_ce'].append(mean_ce)
                r['all_similarities'].append(sims)
                r['all_norm_entropies'].extend(
                    [si['norm_entropies'] for si in gi_sis],
                )
                r['graph_info'].append((ogb_idx, label, n_nodes))
                r['per_class_correct'][label] += sum(correct_list)
                r['per_class_total'][label] += len(correct_list)

    # ---- 8. 汇总统计 ----
    print(f"\n{'='*80}")
    print("=== Diagnosis Results ===")
    print(f"{'='*80}")

    final_results = {
        'config': {
            'checkpoint': args.checkpoint,
            'classifier': args.classifier,
            'n_graphs': len(sampled),
            'samples_per_graph': args.samples_per_graph,
            'temperature': args.temperature,
            'seed': args.seed,
            'config_manifest': config_manifest,
        },
        'by_budget': {},
    }

    for budget_ratio in BUDGET_OPTIONS:
        r = results_by_budget[budget_ratio]

        # Easy / middle / hard
        successes = r['per_graph_success']
        mean_ces = r['per_graph_ce']
        graph_indices = r['graph_info']  # list of (ogb_idx, label, n_nodes)

        easy_list = []
        middle_list = []
        hard_list = []
        all_entries = []
        for i, v in enumerate(successes):
            ogb_idx, label, n_nodes = graph_indices[i]
            level = 'easy' if v > 0.8 else ('middle' if v >= 0.2 else 'hard')
            entry = {'ogb_idx': ogb_idx, 'label': label, 'n_nodes': n_nodes,
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
            per_class_acc[str(cls)] = r['per_class_correct'][cls] / r['per_class_total'][cls]

        # 最大类基线（按样本数加权）
        largest_class = max(r['per_class_total'], key=r['per_class_total'].get)
        largest_class_acc = r['per_class_correct'][largest_class] / r['per_class_total'][largest_class]

        # 相似度
        all_flat_sims = [s for sims in r['all_similarities'] for s in sims]
        mean_sim = float(np.mean(all_flat_sims)) if all_flat_sims else 0.0
        std_sim = float(np.std(all_flat_sims)) if all_flat_sims else 0.0

        # 归一化熵
        ent_means = []
        max_steps = max(len(e) for e in r['all_norm_entropies']) if r['all_norm_entropies'] else 0
        for step in range(max_steps):
            vals = [e[step] for e in r['all_norm_entropies']
                    if step < len(e) and not math.isnan(e[step])]
            ent_means.append(float(np.mean(vals)) if vals else None)

        budget_pct = f"{int(budget_ratio * 100)}%"
        print(f"\n--- Budget {budget_pct} ---")
        print(f"  Graphs: {n_graphs}")
        print(f"  Easy (>80%):  {easy:4d} ({easy/max(n_graphs,1)*100:5.1f}%)")
        print(f"  Middle:       {middle:4d} ({middle/max(n_graphs,1)*100:5.1f}%)")
        print(f"  Hard (<20%):  {hard:4d} ({hard/max(n_graphs,1)*100:5.1f}%)")
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # 同时输出一个紧凑的对比表
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    header = f"{'Budget':>6}  {'Easy%':>7}  {'Mid%':>7}  {'Hard%':>7}  {'Succ':>6}  {'Sim':>7}  {'H_norm(1st)':>12}"
    print(header)
    print("-" * len(header))
    for budget_ratio in BUDGET_OPTIONS:
        bp = f"{int(budget_ratio * 100)}%"
        d = final_results['by_budget'][bp]
        h_first = d['norm_entropy_by_step'][0] if d['norm_entropy_by_step'] else None
        print(f"{bp:>6}  {d['easy_ratio']:6.3f}  {d['middle_ratio']:6.3f}  {d['hard_ratio']:6.3f}  "
              f"{d['mean_success_rate']:5.3f}  {d['fixed_start_similarity_mean']:6.3f}  "
              f"{h_first if h_first is not None else 'N/A':>12}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == '__main__':
    main()
