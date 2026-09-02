"""加载 easy 图，用 reader checkpoint 采样 4 条轨迹，按平均 CE 从高到低排序，
取前 keep_ratio 构成新训练集。

自 m5/base/analysis/rank_easy_by_ce.py 泛化迁移：
- 任意注册数据集（datasets 注册表）；有 load_graph_subset 的数据集（PPA）
  按 OGB train 索引经 load_graph_subset 精确加载 easy 图，其它按本地位置
  取 ds.train_graphs[idx]；
- reader/classifier 经 train_reader.core 从 checkpoint 重建
  （base/attention 自动识别）。

用途（2026-08-05 ex2 结论）：easy 图占训练集 ~48%，梯度易饱和。保留其中对
reader 最难的（平均 CE 最高的）前 keep_ratio，构成新训练集：

    train_filtered = top-keep% easy ∪ middle ∪ hard

供 train_reader/train_rl.py 等训练脚本做图级过滤参考。

输入 difficulty CSV = diagnose_checkpoint.py 产出的 difficulty_mapping_{pct}.csv
（列：ogb_idx/label/success_rate/level/mean_ce；ogb_idx 语义随数据集）。

用法:
  python code/diagnosis/rank_easy_by_ce.py \
    --checkpoint result/ppa/experiments/ex2/base/rl/checkpoint_epoch8.pt \
    --classifier m3/best_model_ft.pt \
    --difficulty-csv result/ppa/diagnosis/ex2/difficulty_mapping_10%.csv \
    --root D:\\ogb-master\\dataset \
    --keep-ratio 0.3
"""

import argparse
import gc
import random
import sys
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
from tools.config import save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import SEED_METHOD_STR, deterministic_seed  # noqa: E402
from train_reader.core import (load_classifier_frozen,  # noqa: E402
                               load_reader_from_checkpoint,
                               reveal_and_update_beliefs_batch)


def resolve(p) -> Path:
    """相对路径按 repo root 解析（绝对路径原样）。"""
    path = Path(p)
    return path if path.is_absolute() else CODE_DIR / path


@torch.no_grad()
def run_batched_trajectories(reader, states_info, temperature, device):
    """Lockstep batched trajectory sampling（base/attention 通用）。

    Each entry of *states_info* gets ``ce`` filled in-place later by the
    classify stage; this function only advances the rollout.
    """
    n_states = len(states_info)
    if n_states == 0:
        return

    use_belief = hasattr(reader, 'init_beliefs')
    active = [True] * n_states
    max_steps = max(si['state']._budget_limit for si in states_info)
    if use_belief:
        all_beliefs = [reader.init_beliefs(1).to(device) for _ in range(n_states)]
        all_subgraphs = [None] * n_states

    for _step in range(max_steps):
        active_idx = [j for j in range(n_states)
                      if active[j] and not states_info[j]['state'].is_frontier_empty]
        if not active_idx:
            break

        if use_belief:
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
        else:
            batch_results = reader.forward_states(
                [states_info[j]['state'] for j in active_idx],
                temperature=temperature,
            )

        batch_states, batch_nodes, batch_local_idx = [], [], []
        for k, j in enumerate(active_idx):
            scores, probs, frontier_nodes = batch_results[k]
            if not frontier_nodes:
                active[j] = False
                continue

            state = states_info[j]['state']
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            chosen = frontier_nodes[action_idx.item()]

            if use_belief:
                batch_states.append(state)
                batch_nodes.append(chosen)
                batch_local_idx.append(j)
            else:
                reveal_node(state, chosen)
                if state.get_budget_used() >= state._budget_limit:
                    active[j] = False

        if use_belief and batch_states:
            batch_beliefs = torch.cat(
                [all_beliefs[j] for j in batch_local_idx], dim=0)
            new_beliefs, new_subgraphs = reveal_and_update_beliefs_batch(
                reader, batch_states, batch_nodes, batch_beliefs, device)
            for k, j in enumerate(batch_local_idx):
                all_beliefs[j] = new_beliefs[k:k + 1]
                all_subgraphs[j] = new_subgraphs[k]

        if use_belief:
            for j in active_idx:
                if active[j] and states_info[j]['state'].get_budget_used() \
                        >= states_info[j]['state']._budget_limit:
                    active[j] = False


def main():
    parser = argparse.ArgumentParser(
        description='按 reader 平均 CE 对 easy 图排序并取前 keep_ratio（组成新训练集）'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Reader checkpoint 路径（相对 repo root）')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt',
                        help='冻结 classifier checkpoint 路径')
    parser.add_argument('--difficulty-csv', required=True,
                        help='difficulty_mapping CSV 路径（相对 repo root）')
    cli.add_dataset_args(parser, default_dataset='ppa')
    parser.add_argument('--samples-per-graph', type=int, default=4,
                        help='每张 easy 图的采样轨迹数')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--budget-ratio', type=float, default=0.10)
    parser.add_argument('--keep-ratio', type=float, default=0.30,
                        help='按平均 CE 从高到低保留的 easy 图比例')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='',
                        help='输出目录（默认 result/{dataset}/filtered/'
                             'easy_top{keep_pct}）')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.output_dir:
        output_dir = resolve(args.output_dir)
    else:
        keep_pct = int(round(args.keep_ratio * 100))
        output_dir = (CODE_DIR / 'result' / args.dataset / 'filtered' /
                      f'easy_top{keep_pct}')
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs)
    spec = ds.spec
    n_classes = spec.n_classes

    # ---- 1. 读 difficulty mapping，取 easy 图 ----
    diff_path = resolve(args.difficulty_csv)
    print(f"\n=== Reading difficulty mapping: {diff_path} ===")
    diff = pd.read_csv(diff_path)
    print(f"  Total rows: {len(diff)}")
    easy_df = diff[diff['level'] == 'easy'].copy()
    middle_df = diff[diff['level'] == 'middle'].copy()
    hard_df = diff[diff['level'] == 'hard'].copy()
    print(f"  easy={len(easy_df)}  middle={len(middle_df)}  hard={len(hard_df)}")

    easy_entries = [(int(gidx), int(lbl))
                    for gidx, lbl in zip(easy_df['ogb_idx'], easy_df['label'])]

    # ---- 2. 加载 easy 图（PPA 走 load_graph_subset 精确加载；其它本地位置） ----
    easy_gidx = [g for g, _ in easy_entries]
    print(f"\n=== Loading {len(easy_gidx)} easy graphs ===")
    if spec.load_graph_subset is not None:
        pool = spec.load_graph_subset(spec, easy_gidx)
        missing = [i for i in easy_gidx if i not in pool]
        if missing:
            raise SystemExit(
                f"{len(missing)} 个 ogb_idx 无法加载（不在 train split 内？），"
                f"示例: {missing[:5]}")
        easy_graphs = [pool[i][0] for i in easy_gidx]
    else:
        easy_graphs = [ds.train_graphs[i] for i in easy_gidx]
    print(f"  Loaded {len(easy_graphs)} graphs (CPU, 自环已由加载器添加)")

    # ---- 3. 加载 classifier（冻结） ----
    print("\n=== Loading classifier ===")
    clf_path = resolve(args.classifier)
    clf_ckpt = torch.load(str(clf_path), map_location='cpu', weights_only=False)
    classifier = load_classifier_frozen(clf_ckpt, device)
    print(f"  {clf_path} (val_acc={clf_ckpt.get('val_acc', 'N/A')})")

    # ---- 4. 加载 reader（base/attention 自动识别） ----
    print("\n=== Loading reader ===")
    ckpt_path = resolve(args.checkpoint)
    rl_ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    reader = load_reader_from_checkpoint(rl_ckpt, device,
                                         stochastic=True,
                                         temperature=args.temperature)
    reader.eval()
    print(f"  kind={reader.config.kind} (auto-detected), "
          f"hidden={reader.config.encoder_hidden_dims}, "
          f"epoch={rl_ckpt.get('epoch', 'N/A')}")

    # ---- 5. 批量锁步采样 + 分类 ----
    print(f"\n=== Sampling {args.samples_per_graph} trajectories per easy graph "
          f"(budget={args.budget_ratio:.0%}, temperature={args.temperature}) ===")

    builder = SubgraphBuilder()
    CHUNK_SIZE = 16
    CLASSIFY_CHUNK = 512  # matches eval loop: accumulate subgraphs, batch-classify
    results = []  # (gidx, label, mean_ce)
    classify_queue = []  # subgraph states accumulated for batched classification
    # States are only kept until their CE is computed, then released. Keeping all
    # ~4x38213 GraphState objects (each with a full _edge_key_to_id dict) until the
    # end exhausted RAM at ~45% and triggered swap (50x slowdown, 2026-08-06).
    pending = []  # (chunk, states_info) waiting for CE

    def _flush_classify():
        nonlocal classify_queue, pending
        if not classify_queue:
            return
        for c_start in range(0, len(classify_queue), CLASSIFY_CHUNK):
            c_end = min(c_start + CLASSIFY_CHUNK, len(classify_queue))
            chunk_si = classify_queue[c_start:c_end]
            batch_g = dgl.batch([
                builder.build(si['state'])[0] for si in chunk_si
            ]).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long,
                               device=device)
            logits = classifier(batch_g, nids)
            for j, si in enumerate(chunk_si):
                y = torch.tensor([si['label']], device=logits.device)
                si['ce'] = float(F.cross_entropy(logits[j:j + 1], y).item())
        classify_queue = []
        # Every flushed state now has 'ce'; fold pending chunks into results.
        for chunk, states_info in pending:
            for gi, (gidx, label) in enumerate(chunk):
                gi_sis = [si for si in states_info if si['graph_gi'] == gi]
                mean_ce = float(np.mean([si['ce'] for si in gi_sis]))
                results.append((gidx, label, mean_ce))
        pending = []
        gc.collect()

    for chunk_start in tqdm(range(0, len(easy_entries), CHUNK_SIZE),
                            desc="  Graphs"):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(easy_entries))
        chunk = easy_entries[chunk_start:chunk_end]

        states_info = []
        for gi, (gidx, label) in enumerate(chunk):
            g = easy_graphs[chunk_start + gi]
            n_nodes = g.num_nodes()
            B = max(1, int(args.budget_ratio * n_nodes))
            seed_node = deterministic_seed(gidx, 0, n_nodes)
            for _s in range(args.samples_per_graph):
                state = GraphState(g, seed_node=seed_node)
                state._budget_limit = B
                states_info.append({
                    'state': state, 'graph_gi': gi, 'label': label,
                })

        run_batched_trajectories(reader, states_info, args.temperature, device)

        classify_queue.extend(states_info)
        pending.append((chunk, states_info))
        if len(classify_queue) >= CLASSIFY_CHUNK:
            _flush_classify()

    _flush_classify()

    # ---- 6. 排序 + 取前 keep_ratio ----
    results.sort(key=lambda r: r[2], reverse=True)
    n_keep = max(1, int(round(len(results) * args.keep_ratio)))
    kept = results[:n_keep]

    print(f"\n{'=' * 80}")
    print(f"=== Ranking Summary ===")
    print(f"  Easy graphs ranked: {len(results)}")
    print(f"  Keep ratio: {args.keep_ratio:.0%} -> {n_keep} graphs")
    print(f"  Mean CE range of kept: "
          f"{kept[-1][2]:.4f} .. {kept[0][2]:.4f} (high = hardest)")
    print(f"  Kept label spread: {len({l for _, l, _ in kept})}/{n_classes} classes")

    # ---- 7. 保存 ----
    ranked_df = pd.DataFrame(results, columns=['ogb_idx', 'label', 'mean_ce'])
    ranked_path = output_dir / 'easy_ranked_by_ce.csv'
    ranked_df.to_csv(str(ranked_path), index=False)
    print(f"  Ranked easy CSV saved: {ranked_path}")

    kept_set = {r[0] for r in kept}

    # combine: kept easy + all middle (no hard)
    combined = pd.concat([
        pd.Series(sorted(kept_set)),
        pd.Series(middle_df['ogb_idx'].astype(int).tolist()),
    ], ignore_index=True).sort_values().reset_index(drop=True)
    combined_path = output_dir / 'train_filtered.csv.gz'
    combined.to_csv(str(combined_path), index=False, header=False,
                    compression='gzip')
    print(f"  Filtered train split saved: {combined_path} "
          f"({len(combined)} graphs)")

    report = {
        'checkpoint': str(ckpt_path),
        'classifier': str(clf_path),
        'dataset': spec.name,
        'difficulty_csv': str(diff_path),
        'reader_kind': reader.config.kind,
        'samples_per_graph': args.samples_per_graph,
        'temperature': args.temperature,
        'budget_ratio': args.budget_ratio,
        'seed': args.seed,
        'seed_method': SEED_METHOD_STR,
        'easy_total': len(results),
        'easy_kept': n_keep,
        'keep_ratio': args.keep_ratio,
        'easy_mean_ce_min': float(kept[-1][2]),
        'easy_mean_ce_max': float(kept[0][2]),
        'train_filtered_size': int(len(combined)),
        'easy_kept_ce_threshold': float(kept[-1][2]),
        'middle_count': int(len(middle_df)),
        'hard_excluded': int(len(hard_df)),
        'description': (
            'Easy graphs ranked by mean CE of reader trajectories (higher = '
            'reader still struggles). Kept = top keep_ratio of easy by mean CE, '
            'combined with all middle graphs. Hard graphs excluded.'
        ),
    }
    report_path = output_dir / 'filter_report.json'
    save_json(report, report_path)
    print(f"  Report saved: {report_path}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == '__main__':
    main()
