"""
Fine-tune pretrained GIN on reader-generated partial subgraphs
================================================================

Loads full-graph pretrained weights (best_model.pt), applies one of three
fine-tuning strategies, and trains on subgraphs produced by a random reader.

Strategies:
  A      Freeze early layers (0-1), fine-tune later layers (2-4) + head
  B1     Freeze all GIN layers, add bottleneck adapters after each layer
  A+B1   Freeze early layers (0-1), add adapters to later layers (2-4)

Usage:
  python finetune_reader.py --strategy A --pretrained best_model.pt
  python finetune_reader.py --strategy B1 --pretrained best_model.pt --epochs 30
  python finetune_reader.py --strategy A+B1 --pretrained best_model.pt
"""

import argparse
import logging
import random
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ogb.graphproppred import DglGraphPropPredDataset, Evaluator

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_loop import RevealLoop
from active_graph_reader.reader.fixed.registry import get_strategy
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from active_graph_reader.gnn.classifier import GraphClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
# Each strategy = (freeze_layers, adapter_layers, description)
# Layer indices: 0..4 for 5-layer GIN

STRATEGIES = {
    'A': {
        'freeze_layers': {0, 1},
        'adapter_layers': set(),
        'desc': 'Freeze layers 0-1, fine-tune layers 2-4 + head',
    },
    'B1': {
        'freeze_layers': {0, 1, 2, 3, 4},
        'adapter_layers': {0, 1, 2, 3, 4},
        'desc': 'Freeze all layers, bottleneck adapters on all layers',
    },
    'A+B1': {
        'freeze_layers': {0, 1},
        'adapter_layers': {2, 3, 4},
        'desc': 'Freeze layers 0-1, bottleneck adapters on layers 2-4',
    },
}

# Budget ratios sampled during training (per design doc 3.4)
TRAIN_BUDGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
# Fixed budget ratios for evaluation
EVAL_BUDGET_RATIOS = [0.05, 0.10, 0.20, 0.50]


def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune GIN on reader subgraphs')
    p.add_argument('--strategy', type=str, required=True,
                   choices=['A', 'B1', 'A+B1'],
                   help='Fine-tuning strategy')
    p.add_argument('--pretrained', type=str, required=True,
                   help='Path to best_model.pt from full-graph training')
    p.add_argument('--root', type=str, default='dataset',
                   help='OGB dataset root directory')
    p.add_argument('--epochs', type=int, default=20,
                   help='Fine-tuning epochs (default: 20)')
    p.add_argument('--batch-size', type=int, default=32,
                   help='Batch size in full graphs (default: 32)')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='Learning rate (default: 1e-4, lower than full-graph 1e-3)')
    p.add_argument('--adapter-ratio', type=int, default=16,
                   help='Adapter bottleneck ratio (default: 16; hidden//ratio)')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--output', type=str, default='',
                   help='Output path for best fine-tuned model')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data: load full graphs (no DataLoader — we process individually)
# ---------------------------------------------------------------------------

def load_dataset(root: str):
    """Load OGBG-PPA, add self-loops, return splits."""
    logger.info("Loading OGBG-PPA dataset (root=%s)...", root)
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
    split = dataset.get_idx_split()

    logger.info("Adding self-loops to all graphs...")
    for i in tqdm(range(len(dataset)), desc="self-loops"):
        dataset.graphs[i] = dgl.add_self_loop(dataset.graphs[i])

    logger.info("Dataset: %d graphs, %d classes",
                len(dataset), int(dataset.num_classes))
    return dataset, split


# ---------------------------------------------------------------------------
# Reader → Subgraph
# ---------------------------------------------------------------------------

def reveal_and_build(
    full_g: dgl.DGLGraph,
    budget: int,
    seed_node: int,
    builder: SubgraphBuilder,
    strategy,
    None,
):
    """Run reader on a full graph and return the subgraph.

    Returns (sub_g, original_ids) or None if the subgraph is empty.
    """
    state = GraphState(full_g, seed_node=seed_node)
    loop = RevealLoop(state, strategy)
    loop.run(budget=budget)

    sub_g, original_ids = builder.build(state)
    if sub_g.num_nodes() == 0:
        return None
    return sub_g, original_ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, dataset, split_idx, builder, reader_strategy,
    optimizer, device, budget_ratios, inner_batch=8,
):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    indices = split_idx.tolist()
    random.shuffle(indices)

    sub_batch, label_batch = [], []

    for idx in tqdm(indices, desc="train", leave=False):
        full_g, label = dataset[int(idx)]
        n_nodes = full_g.num_nodes()
        ratio = random.choice(budget_ratios)
        budget = max(1, int(ratio * n_nodes))
        seed_node = random.randint(0, n_nodes - 1)

        result = reveal_and_build(
            full_g, budget, seed_node, builder,
            reader_strategy,
        )
        if result is None:
            continue

        sub_g, _ = result
        sub_batch.append(sub_g)
        label_batch.append(label)

        # Forward + backward when batch is full (or at end)
        if len(sub_batch) >= inner_batch:
            loss = _forward_batch(model, sub_batch, label_batch, optimizer, device)
            total_loss += loss * len(sub_batch)
            total_graphs += len(sub_batch)
            sub_batch, label_batch = [], []

    # Remaining
    if sub_batch:
        loss = _forward_batch(model, sub_batch, label_batch, optimizer, device)
        total_loss += loss * len(sub_batch)
        total_graphs += len(sub_batch)

    return total_loss / max(total_graphs, 1)


def _forward_batch(model, sub_graphs, labels, optimizer, device):
    """Batch subgraphs together, forward, backward. Returns scalar loss."""
    batch_g = dgl.batch(sub_graphs).to(device)
    node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
    label_t = torch.tensor(labels, dtype=torch.long, device=device).view(-1)

    optimizer.zero_grad()
    logits = model(batch_g, node_ids)
    loss = nn.functional.cross_entropy(logits, label_t)
    loss.backward()
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model, dataset, split_idx, builder, reader_strategy,
    evaluator, device, budget_ratios, n_seeds=3,
):
    model.eval()
    y_true, y_pred = [], []

    indices = split_idx.tolist()

    for idx in tqdm(indices, desc="eval", leave=False):
        full_g, label = dataset[int(idx)]
        n_nodes = full_g.num_nodes()

        # Evaluate at each budget ratio with multiple seeds
        for ratio in budget_ratios:
            budget = max(1, int(ratio * n_nodes))
            for seed in range(n_seeds):
                seed_node = (idx * 31 + seed * 17 + 7) % max(n_nodes, 1)
                result = reveal_and_build(
                    full_g, budget, seed_node, builder,
                    reader_strategy,
                )
                if result is None:
                    continue

                sub_g, _ = result
                batch_g = dgl.batch([sub_g]).to(device)
                node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)

                logits = model(batch_g, node_ids)
                y_true.append(torch.tensor([label]).view(-1, 1))
                y_pred.append(logits.argmax(dim=1).view(-1, 1).cpu())

    if not y_true:
        return {'acc': 0.0}

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    strategy_cfg = STRATEGIES[args.strategy]
    logger.info("Strategy: %s — %s", args.strategy, strategy_cfg['desc'])

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Data
    dataset, split = load_dataset(args.root)
    n_classes = int(dataset.num_classes)
    edge_feat_dim = int(dataset.graphs[0].edata['feat'].shape[1])
    evaluator = Evaluator('ogbg-ppa')

    # Model — fresh architecture, then load pretrained weights
    logger.info("Creating GraphClassifier and loading pretrained weights...")
    model = GraphClassifier(
        edge_feat_dim=edge_feat_dim,
        n_classes=n_classes,
    ).to(device)

    model.load_pretrained(args.pretrained, device=str(device))

    # Apply fine-tuning strategy
    model.apply_finetune_strategy(
        freeze_layers=strategy_cfg['freeze_layers'],
        adapter_layers=strategy_cfg['adapter_layers'],
        adapter_bottleneck_ratio=args.adapter_ratio,
        freeze_node_emb=True,
    )
    model.log_trainable()

    # Reusable components
    builder = SubgraphBuilder()
    reader_strategy = get_strategy('random')

    # Optimizer — only trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.lr)

    # Output path
    output_path = args.output
    if not output_path:
        output_path = str(
            Path(__file__).parent / f"finetuned_{args.strategy}.pt"
        )

    # Fine-tuning loop
    best_val = -1.0
    train_idx = split['train']
    valid_idx = split['valid']
    test_idx = split['test']

    # Limit eval set size for speed (use subset for validation during tuning)
    valid_subset = valid_idx[:2000]
    test_subset = test_idx[:2000]

    for epoch in range(1, args.epochs + 1):
        logger.info("===== Epoch %d / %d =====", epoch, args.epochs)

        train_loss = train_one_epoch(
            model, dataset, train_idx, builder,
            reader_strategy, optimizer, device,
            TRAIN_BUDGET_RATIOS,
        )
        logger.info("train_loss=%.4f", train_loss)

        # Validate every epoch (on subset for speed)
        valid_perf = evaluate(
            model, dataset, valid_subset, builder,
            reader_strategy, evaluator, device,
            EVAL_BUDGET_RATIOS, n_seeds=1,
        )
        metric = evaluator.eval_metric
        logger.info("valid_%s=%.4f", metric, valid_perf[metric])

        if valid_perf[metric] > best_val:
            best_val = valid_perf[metric]
            torch.save({
                'model_state_dict': model.state_dict(),
                'strategy': args.strategy,
                'val_acc': best_val,
                'epoch': epoch,
            }, output_path)
            logger.info("Best model saved to %s (val=%.4f)", output_path, best_val)

    # Final test evaluation
    logger.info("Loading best model for test evaluation...")
    ckpt = torch.load(output_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_perf = evaluate(
        model, dataset, test_subset, builder,
        reader_strategy, evaluator, device,
        EVAL_BUDGET_RATIOS, n_seeds=3,
    )
    logger.info("Final test_%s=%.4f", metric, test_perf[metric])
    logger.info("Done. Best model: %s", output_path)


if __name__ == '__main__':
    main()
