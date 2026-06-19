"""用 v0 Reader 在指定标签的训练图上生成轨迹。

使用方式：
  python code/m4/generate_reader_trajectories.py --labels 0,5,12
    --reader-checkpoint m5/checkpoint/best_reader.pt
    --classifier m3/best_model_ft.pt --root D:\ogb-master\dataset
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import dgl
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
sys.path.insert(0, str(_repo_root))

from ogb.graphproppred import DglGraphPropPredDataset
from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m4.teachers.base import Trajectory


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
    print(f"Target labels: {target_labels}")
    print(f"Budget ratios: {budget_ratios}")

    # ---- 加载 OGB 数据集 ----
    print("\n=== Loading OGB dataset ===")
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)
    train_idx = dataset.get_idx_split()['train'].tolist()

    # ---- 按标签收集候选图 ----
    print("\n=== Collecting candidate graphs ===")
    candidates = {label: [] for label in target_labels}
    for ogb_idx in tqdm(train_idx, desc="  Scanning labels"):
        _, lbl = dataset[ogb_idx]
        label = lbl.item()
        if label in candidates and len(candidates[label]) < args.max_graphs_per_label:
            candidates[label].append(ogb_idx)
    for label in target_labels:
        print(f"  label {label}: {len(candidates[label])} graphs")

    # ---- 加载 reader + classifier ----
    print("\n=== Loading v0 LearnableReader ===")
    reader_ckpt = torch.load(str(repo_root / args.reader_checkpoint),
                             map_location='cpu', weights_only=False)
    hidden_dim, n_layers = 0, 0
    sd = reader_ckpt['reader_state_dict']
    hidden_dim = sd['frontier_encoder.node_type_emb.weight'].shape[1]
    for k in sd:
        if k.startswith('frontier_encoder.convs.'):
            n_layers = max(n_layers, int(k.split('.')[2]) + 1)
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

    # ---- 生成轨迹 ----
    print("\n=== Generating trajectories ===")
    all_trajectories = []  # List[List[Trajectory]] — 每个元素是一张图的全部轨迹
    stats = {r: {l: 0 for l in target_labels} for r in budget_ratios}

    for label in target_labels:
        graph_list = candidates[label]
        for ogb_idx in tqdm(graph_list, desc=f"  label {label}"):
            g_raw, lbl = dataset[ogb_idx]
            g_raw = dgl.add_self_loop(g_raw)
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

                    if pred == lbl.item():
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

    # ---- 保存 ----
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'trajectories': all_trajectories}, str(output_path))

    print(f"\nSaved {len(all_trajectories)} graph-entries to {output_path}")
    for ratio in budget_ratios:
        total = sum(stats[ratio].values())
        per_label = ', '.join(f'{l}:{stats[ratio][l]}' for l in target_labels)
        print(f"  r{ratio:.2f}: {total} trajectories ({per_label})")


if __name__ == '__main__':
    main()
