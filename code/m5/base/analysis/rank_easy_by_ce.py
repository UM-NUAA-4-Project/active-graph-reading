"""加载 easy 图，用 RL checkpoint 采样 4 条轨迹，按平均 CE 从高到低排序，取前 keep_ratio。

用途（2026-08-05 ex2 结论）：easy 图占训练集 ~48%，梯度易饱和。保留其中对 reader
最难的（平均 CE 最高的）前 keep_ratio，构成新训练集：

    train_filtered = top-keep% easy ∪ middle ∪ hard

供 train_rl.py --train-split 使用。

用法:
  python code/m5/base/analysis/rank_easy_by_ce.py \
    --checkpoint m5/base/rl/asymmetric/ex2/checkpoint_epoch8.pt \
    --classifier m3/best_model_ft.pt \
    --difficulty-csv "m5/base/rl/asymmetric/ex2/diagnosis/difficulty_mapping_10%.csv" \
    --root 'D:\ogb-master\dataset' \
    --keep-ratio 0.3 \
    --output-dir m5/base/filtered/easy_top30
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent.parent
sys.path.insert(0, str(_repo_root))

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from m5.training_utils import infer_reader_config, patch_old_checkpoint

K = 37


def deterministic_seed(graph_idx: int, seed_i: int, n_nodes: int) -> int:
    return (graph_idx * 31 + seed_i * 17 + 7) % n_nodes


@torch.no_grad()
def run_batched_trajectories(reader, states_info, temperature, device):
    """Lockstep batched trajectory sampling — mirrors diagnose/train pattern.

    Each entry of *states_info* gets ``ce`` filled in-place.
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
            [states_info[j]['state'] for j in active_idx], temperature=temperature,
        )

        for k, j in enumerate(active_idx):
            scores, probs, frontier_nodes = batch_results[k]
            if not frontier_nodes:
                active[j] = False
                continue

            state = states_info[j]['state']
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            chosen = frontier_nodes[action_idx.item()]
            reveal_node(state, chosen)
            if state.get_budget_used() >= state._budget_limit:
                active[j] = False


def main():
    parser = argparse.ArgumentParser(
        description='按 reader 平均 CE 对 easy 图排序并取前 keep_ratio（组成新训练集）'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='RL reader checkpoint 路径（相对于 repo root）')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt',
                        help='冻结 classifier checkpoint 路径')
    parser.add_argument('--difficulty-csv', required=True,
                        help='difficulty_mapping CSV 路径（相对于 repo root）')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples-per-graph', type=int, default=4,
                        help='每张 easy 图的采样轨迹数')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--keep-ratio', type=float, default=0.30,
                        help='按平均 CE 从高到低保留的 easy 图比例')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='m5/base/filtered/easy_top30')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读 difficulty mapping，取 easy 图 ----
    print(f"\n=== Reading difficulty mapping: {args.difficulty_csv} ===")
    diff = pd.read_csv(repo_root / args.difficulty_csv)
    print(f"  Total rows: {len(diff)}")
    easy_df = diff[diff['level'] == 'easy'].copy()
    middle_df = diff[diff['level'] == 'middle'].copy()
    hard_df = diff[diff['level'] == 'hard'].copy()
    print(f"  easy={len(easy_df)}  middle={len(middle_df)}  hard={len(hard_df)}")

    easy_entries = [(int(ogb_idx), int(lbl))
                    for ogb_idx, lbl in zip(easy_df['ogb_idx'], easy_df['label'])]

    # ---- 2. 加载 easy 图 ----
    root_path = Path(args.root)
    processed_path = str(root_path / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    easy_ogb = easy_df['ogb_idx'].tolist()
    print(f"\n=== Loading {len(easy_ogb)} easy graphs ===")
    easy_graphs, _ = dgl.load_graphs(processed_path, idx_list=easy_ogb)
    easy_graphs = [dgl.add_self_loop(g) for g in easy_graphs]
    print(f"  Loaded {len(easy_graphs)} graphs (CPU, self-loops added)")

    # ---- 3. 加载 classifier（冻结） ----
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

    # ---- 4. 加载 reader ----
    print("\n=== Loading LearnableReader ===")
    rl_ckpt = torch.load(str(repo_root / args.checkpoint), map_location='cpu',
                         weights_only=False)
    reader_sd = rl_ckpt.get('reader_state_dict', rl_ckpt)
    reader_sd, was_patched = patch_old_checkpoint(reader_sd)
    hidden_dim, n_layers = infer_reader_config(reader_sd)
    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=0.1, stochastic=True, temperature=args.temperature,
    ).to(device)
    reader.load_state_dict(reader_sd)
    reader.eval()
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}, "
          f"epoch={rl_ckpt.get('epoch', 'N/A')}")

    # ---- 5. 批量锁步采样 + 分类 ----
    print(f"\n=== Sampling {args.samples_per_graph} trajectories per easy graph "
          f"(budget={args.budget_ratio:.0%}, temperature={args.temperature}) ===")

    builder = SubgraphBuilder()
    CHUNK_SIZE = 16
    CLASSIFY_CHUNK = 512  # matches eval.py: accumulate subgraphs, batch-classify
    results = []  # (ogb_idx, label, mean_ce)
    classify_queue = []  # subgraph states accumulated for batched classification
    chunk_sis = []  # (chunk, states_info) deferred until all CE computed

    def _flush_classify():
        nonlocal classify_queue
        if not classify_queue:
            return
        for c_start in range(0, len(classify_queue), CLASSIFY_CHUNK):
            c_end = min(c_start + CLASSIFY_CHUNK, len(classify_queue))
            chunk_si = classify_queue[c_start:c_end]
            batch_g = dgl.batch([
                builder.build(si['state'])[0] for si in chunk_si
            ]).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            logits = classifier(batch_g, nids)
            for j, si in enumerate(chunk_si):
                y = torch.tensor([si['label']], device=logits.device)
                si['ce'] = float(F.cross_entropy(logits[j:j + 1], y).item())
        classify_queue = []

    for chunk_start in tqdm(range(0, len(easy_entries), CHUNK_SIZE), desc="  Graphs"):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(easy_entries))
        chunk = easy_entries[chunk_start:chunk_end]

        states_info = []
        for gi, (ogb_idx, label) in enumerate(chunk):
            g = easy_graphs[chunk_start + gi]
            n_nodes = g.num_nodes()
            B = max(1, int(args.budget_ratio * n_nodes))
            seed_node = deterministic_seed(ogb_idx, 0, n_nodes)
            for _s in range(args.samples_per_graph):
                state = GraphState(g, seed_node=seed_node)
                state._budget_limit = B
                states_info.append({
                    'state': state, 'graph_gi': gi, 'label': label,
                })

        run_batched_trajectories(reader, states_info, args.temperature, device)

        classify_queue.extend(states_info)
        chunk_sis.append((chunk, states_info))
        if len(classify_queue) >= CLASSIFY_CHUNK:
            _flush_classify()

    _flush_classify()

    # CE is only filled in after _flush_classify(); compute means now.
    for chunk, states_info in chunk_sis:
        for gi, (ogb_idx, label) in enumerate(chunk):
            gi_sis = [si for si in states_info if si['graph_gi'] == gi]
            mean_ce = float(np.mean([si['ce'] for si in gi_sis]))
            results.append((ogb_idx, label, mean_ce))

    # ---- 6. 排序 + 取前 keep_ratio ----
    results.sort(key=lambda r: r[2], reverse=True)
    n_keep = max(1, int(round(len(results) * args.keep_ratio)))
    kept = results[:n_keep]

    print(f"\n{'='*80}")
    print(f"=== Ranking Summary ===")
    print(f"  Easy graphs ranked: {len(results)}")
    print(f"  Keep ratio: {args.keep_ratio:.0%} -> {n_keep} graphs")
    print(f"  Mean CE range of kept: "
          f"{kept[-1][2]:.4f} .. {kept[0][2]:.4f} (high = hardest)")
    print(f"  Kept label spread: {len({l for _, l, _ in kept})}/{K} classes")

    # ---- 7. 保存 ----
    ranked_df = pd.DataFrame(results, columns=['ogb_idx', 'label', 'mean_ce'])
    ranked_path = output_dir / 'easy_ranked_by_ce.csv'
    ranked_df.to_csv(str(ranked_path), index=False)
    print(f"  Ranked easy CSV saved: {ranked_path}")

    kept_set = {r[0] for r in kept}

    middle_hard = pd.concat([middle_df['ogb_idx'], hard_df['ogb_idx']],
                            ignore_index=True).astype(int)
    combined = pd.concat([
        pd.Series(sorted(kept_set)),
        pd.Series(middle_hard.tolist()),
    ], ignore_index=True).sort_values().reset_index(drop=True)
    combined_path = output_dir / 'train_filtered.csv.gz'
    combined.to_csv(str(combined_path), index=False, header=False,
                    compression='gzip')
    print(f"  Filtered train split saved: {combined_path} "
          f"({len(combined)} graphs)")

    report = {
        'checkpoint': args.checkpoint,
        'classifier': args.classifier,
        'difficulty_csv': args.difficulty_csv,
        'samples_per_graph': args.samples_per_graph,
        'temperature': args.temperature,
        'budget_ratio': args.budget_ratio,
        'easy_total': len(results),
        'easy_kept': n_keep,
        'keep_ratio': args.keep_ratio,
        'easy_mean_ce_min': float(kept[-1][2]),
        'easy_mean_ce_max': float(kept[0][2]),
        'train_filtered_size': int(len(combined)),
        'description': (
            'Easy graphs ranked by mean CE of reader trajectories (higher = '
            'reader still struggles). Kept = top keep_ratio of easy by mean CE, '
            'combined with all middle + hard graphs to form the filtered split.'
        ),
    }
    report_path = output_dir / 'filter_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved: {report_path}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == '__main__':
    main()
