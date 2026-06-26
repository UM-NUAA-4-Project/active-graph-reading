"""用 v0 Reader 在指定标签的训练图上生成轨迹。

内存优化：只加载候选图，完全不加载验证集/测试集/非目标训练图。
数据迁移到 GPU。

使用方式：
  python code/m4/generate_reader_trajectories.py --labels 0,5,12
    --reader-checkpoint m5/checkpoint/best_reader.pt
    --classifier m3/best_model_ft.pt --root D:\ogb-master\dataset
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
sys.path.insert(0, str(_repo_root))

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m4.teachers.base import Trajectory


# ---------------------------------------------------------------------------
# 候选图收集：仅读标签 CSV + 训练集 split，不加载任何图数据
# ---------------------------------------------------------------------------

def collect_candidates(root: str, target_labels: List[int],
                       max_per_label: int) -> Dict[int, List[int]]:
    """从原始 CSV 读取标签和训练集 split，按标签筛选候选 OGB index。

    完全绕过 DglGraphPropPredDataset —— 不加载 158k 张图，
    只读 graph-label.csv.gz（~2 MB）和 train.csv.gz。
    """
    data_dir = Path(root) / 'ogbg_ppa'

    labels_arr = pd.read_csv(
        data_dir / 'raw' / 'graph-label.csv.gz',
        compression='gzip', header=None,
    ).values.flatten()  # (158100,) int64

    train_idx = pd.read_csv(
        data_dir / 'split' / 'species' / 'train.csv.gz',
        compression='gzip', header=None,
    ).values.flatten()  # (78200,) int64

    candidates = {lbl: [] for lbl in target_labels}
    for ogb_idx in sorted(train_idx.tolist()):
        label = int(labels_arr[ogb_idx])
        if label in candidates and len(candidates[label]) < max_per_label:
            candidates[label].append(ogb_idx)

    for lbl in target_labels:
        print(f"  label {lbl}: {len(candidates[lbl])} candidate graphs")
    return candidates


# ---------------------------------------------------------------------------
# 图加载：仅从 DGL 预编译文件加载指定索引的图
# ---------------------------------------------------------------------------

def load_graphs_subset(root: str, indices: List[int],
                       device: torch.device) -> Dict[int, Tuple[dgl.DGLGraph, int]]:
    """通过 dgl.load_graphs(path, idx_list=indices) 仅加载需要的图。
    加自环后迁移到 GPU。返回 {ogb_idx: (gpu_graph, label)}。
    """
    proc_path = str(Path(root) / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    cpu_graphs, label_dict = dgl.load_graphs(proc_path, idx_list=indices)
    all_labels = label_dict['labels']  # 始终返回全量标签，不受 idx_list 影响

    cache: Dict[int, Tuple[dgl.DGLGraph, int]] = {}
    for i, ogb_idx in enumerate(indices):
        g = dgl.add_self_loop(cpu_graphs[i])
        g = g.to(device)
        label = int(all_labels[ogb_idx].item())
        cache[ogb_idx] = (g, label)

    del cpu_graphs
    return cache


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='v0 Reader trajectory generation')
    parser.add_argument('--labels', required=True,
                        help='Comma-separated target labels, e.g. 0,5,12')
    parser.add_argument('--reader-checkpoint', default='m5/checkpoint/best_reader.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='m4/data/reader_trajectories.pt')
    parser.add_argument('--seeds-per-graph', type=int, default=5)
    parser.add_argument('--max-graphs-per-label', type=int, default=50)
    parser.add_argument('--budget-ratios', default='0.05,0.10,0.15',
                        help='Comma-separated reveal ratios')
    args = parser.parse_args()

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    target_labels = [int(x) for x in args.labels.split(',')]
    budget_ratios = [float(x) for x in args.budget_ratios.split(',')]
    print(f"Device: {device}")
    print(f"Target labels: {target_labels}")
    print(f"Budget ratios: {budget_ratios}")

    # ---- 1. 仅读标签 CSV + 训练集 split，筛选候选图 ----
    print("\n=== Collecting candidate graphs (no graph data loaded) ===")
    candidates = collect_candidates(args.root, target_labels, args.max_graphs_per_label)
    all_candidates = [idx for idxs in candidates.values() for idx in idxs]
    print(f"Total candidate graphs: {len(all_candidates)}")
    if not all_candidates:
        print("No candidates found. Exiting.")
        return

    # ---- 2. 仅加载候选图，迁至 GPU ----
    print("\n=== Loading candidate graphs to GPU ===")
    graph_cache = load_graphs_subset(args.root, all_candidates, device)
    print(f"Loaded {len(graph_cache)} graphs to {device}")

    # ---- 3. 加载 reader + classifier ----
    print("\n=== Loading v0 LearnableReader ===")
    reader_ckpt = torch.load(str(repo_root / args.reader_checkpoint),
                             map_location='cpu', weights_only=False)
    sd = reader_ckpt['reader_state_dict']
    hidden_dim = sd['frontier_encoder.node_type_emb.weight'].shape[1]
    n_layers = 0
    for k in sd:
        if k.startswith('frontier_encoder.convs.'):
            n_layers = max(n_layers, int(k.split('.')[2]) + 1)
    print(f"  hidden_dim={hidden_dim}, n_layers={n_layers}")
    reader = LearnableReader(edge_feat_dim=7, hidden_dim=hidden_dim,
                             n_layers=n_layers, dropout=0.1).to(device)
    reader.load_state_dict(sd)
    reader.eval()

    print("\n=== Loading classifier ===")
    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=37, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    builder = SubgraphBuilder()

    # ---- 4. 生成轨迹 ----
    print("\n=== Generating trajectories ===")
    all_trajectories = []   # List[List[Trajectory]]
    stats = {r: {lbl: 0 for lbl in target_labels} for r in budget_ratios}

    for label in target_labels:
        graph_list = candidates[label]
        for ogb_idx in tqdm(graph_list, desc=f"  label {label}"):
            g_raw, lbl_val = graph_cache[ogb_idx]  # 已在 GPU
            n_nodes = g_raw.num_nodes()

            for ratio in budget_ratios:
                budget_abs = max(1, int(ratio * n_nodes))
                trajs = []

                for si in range(args.seeds_per_graph):
                    seed_node = (ogb_idx * 31 + si * 17 + 7) % n_nodes
                    reader.reset()
                    state = GraphState(g_raw, seed_node=seed_node)
                    reveal_order = []

                    while state.get_budget_used() < budget_abs:
                        nxt = reader.select(state)
                        if nxt is None:
                            break
                        reveal_order.append(nxt)
                        reveal_node(state, nxt)

                    sub_g, _ = builder.build(state)
                    sub_g = sub_g.to(device)
                    nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
                    logits = classifier(sub_g, nids)
                    pred = logits.argmax().item()

                    if pred == lbl_val:
                        trajs.append(Trajectory(
                            graph_idx=ogb_idx,
                            seed_node=seed_node,
                            reveal_ratio=ratio,
                            reveal_order=reveal_order,
                            final_pred=pred,
                            is_correct=True,
                            teacher_type='v0-reader',
                            ce=0.0,
                        ))

                if trajs:
                    all_trajectories.append(trajs)
                    stats[ratio][label] += len(trajs)

    # ---- 5. 保存 ----
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'trajectories': all_trajectories}, str(output_path))

    print(f"\nSaved {len(all_trajectories)} graph-entries to {output_path}")
    for ratio in budget_ratios:
        total = sum(stats[ratio].values())
        per_label = ', '.join(f'{lbl}:{stats[ratio][lbl]}' for lbl in target_labels)
        print(f"  r{ratio:.2f}: {total} trajectories ({per_label})")


if __name__ == '__main__':
    main()
