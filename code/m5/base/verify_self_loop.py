"""验证：classifier 子图无自环条件下的准确率。"""

import argparse
import random
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
sys.path.insert(0, str(_repo_root))

import torch
import dgl
import numpy as np
import pandas as pd
from tqdm import tqdm

from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.classifier import GraphClassifier
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from m5.training_utils import patch_old_checkpoint

K = 37
BUDGETS = [0.05, 0.10, 0.15]


def pick_seeds_by_degree(g):
    indeg = g.in_degrees().float()
    outdeg = g.out_degrees().float()
    total_deg = indeg + outdeg
    return int(total_deg.argmax().item()), int(total_deg.argmin().item())


def remove_self_loops(g):
    src, dst = g.edges()
    mask = src != dst
    g2 = dgl.graph((src[mask], dst[mask]), num_nodes=g.num_nodes())
    if 'feat' in g.edata:
        g2.edata['feat'] = g.edata['feat'][mask]
    return g2


@torch.no_grad()
def evaluate(reader, classifier, builder, graphs, ogb_indices, labels, device):
    reader.eval()
    classifier.eval()
    correct = 0
    total = 0

    for i in tqdm(range(len(graphs)), desc="  eval"):
        g = graphs[i]
        gid = ogb_indices[i]
        label = int(labels[gid].item())
        n = g.num_nodes()

        for br in BUDGETS:
            B = max(1, int(br * n))
            best_seed, worst_seed = pick_seeds_by_degree(g)
            for seed_node in (best_seed, worst_seed):
                state = GraphState(g, seed_node=seed_node)
                state._budget_limit = B

                for _ in range(B):
                    if state.is_frontier_empty:
                        break
                    _, probs, fnodes = reader.forward(state)
                    if not fnodes:
                        break
                    reveal_node(state, fnodes[probs.argmax().item()])

                sub_g, _ = builder.build(state)
                sub_g = remove_self_loops(sub_g)
                sub_g = sub_g.to(device)
                nids = torch.zeros(sub_g.num_nodes(), dtype=torch.long, device=device)
                logits = classifier(sub_g, nids)
                if logits.argmax(dim=-1).item() == label:
                    correct += 1
                total += 1

    return correct / total, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='m5/base/rl/asymmetric/ex1/checkpoint_epoch10.pt')
    parser.add_argument('--classifier', default='m3/best_model_ft.pt')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-graphs', type=int, default=0, help='0=全量')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = _repo_root
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode: classifier subgraph WITHOUT self-loop")

    root = Path(args.root)
    proc = str(root / 'ogbg_ppa' / 'processed' / 'dgl_data_processed')
    csv = root / 'ogbg_ppa' / 'split' / 'species' / 'valid.csv.gz'
    indices = pd.read_csv(csv, compression='gzip', header=None).values.flatten().tolist()
    if args.n_graphs > 0:
        indices = random.sample(indices, args.n_graphs)

    print(f"Loading {len(indices)} graphs ...")
    graphs, labels = dgl.load_graphs(proc, idx_list=indices)
    graphs = [dgl.add_self_loop(g) for g in graphs]

    classifier = GraphClassifier(
        edge_feat_dim=7, n_classes=K, hidden_dim=300, n_layers=5, dropout=0.5,
    ).to(device)
    ckpt = torch.load(str(repo_root / args.classifier), map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'], strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    reader = LearnableReader(
        edge_feat_dim=7, hidden_dim=128, n_layers=2, dropout=0.1, stochastic=False,
    ).to(device)
    r_ckpt = torch.load(str(repo_root / args.checkpoint), map_location=device, weights_only=False)
    sd, _ = patch_old_checkpoint(r_ckpt['reader_state_dict'])
    reader.load_state_dict(sd)
    reader.eval()

    builder = SubgraphBuilder()

    acc, n = evaluate(reader, classifier, builder, graphs, indices, labels['labels'], device)
    print(f"Accuracy (no self-loop): {acc:.4f}  ({n} samples)")


if __name__ == '__main__':
    main()
