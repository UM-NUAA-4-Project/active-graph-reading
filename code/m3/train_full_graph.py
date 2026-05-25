"""
Full-graph GIN training for OGBG-PPA upper bound
=================================================

Trains a GraphClassifier (GINEEncoder + Linear head) on complete PPA graphs
to establish the performance ceiling. Uses the OGB-official GIN architecture.

Output files:
  - checkpoint.pt    : overwritten every 5 epochs
  - best_model.pt    : saved on best validation accuracy
  - curves.pt        : train/val/test accuracy curves for final epoch

Usage:
  python train_full_graph.py                          # defaults
  python train_full_graph.py --epochs 50 --device cuda:0
  python train_full_graph.py --eval-only --load best_model.pt
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure mycode/ is on the path so that imports like
# "from active_graph_reader.encoder.classifier import GraphClassifier" work.
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import numpy as np
from tqdm import tqdm

from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl

from active_graph_reader.encoder.classifier import GraphClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (matching OGB reference)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    hidden_dim=300,
    n_layers=5,
    dropout=0.5,
    batch_size=32,
    epochs=100,
    lr=0.001,
    device='auto',
    output_dir=None,       # defaults to script directory
)


def parse_args():
    p = argparse.ArgumentParser(description='Full-graph GIN training (OGBG-PPA)')
    p.add_argument('--hidden-dim', type=int, default=DEFAULTS['hidden_dim'])
    p.add_argument('--n-layers', type=int, default=DEFAULTS['n_layers'])
    p.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
    p.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    p.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    p.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    p.add_argument('--device', type=str, default=DEFAULTS['device'])
    p.add_argument('--output-dir', type=str, default=DEFAULTS['output_dir'])
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training, evaluate saved model only')
    p.add_argument('--load', type=str, default='',
                   help='Path to saved model for eval-only mode')
    p.add_argument('--root', type=str, default='dataset',
                   help='OGB dataset root directory (default: dataset)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(batch_size: int, root: str):
    """Load dataset, add self-loops, create DataLoaders."""
    logger.info("Loading OGBG-PPA dataset (root=%s)...", root)
    dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
    split = dataset.get_idx_split()

    n_classes = int(dataset.num_classes)
    edge_feat_dim = int(dataset.graphs[0].edata['feat'].shape[1])

    logger.info("Adding self-loops to all graphs...")
    for i in tqdm(range(len(dataset)), desc="self-loops"):
        dataset.graphs[i] = dgl.add_self_loop(dataset.graphs[i])

    loaders = {}
    for name, idx in [('train', split['train']),
                      ('valid', split['valid']),
                      ('test', split['test'])]:
        subset = [dataset[int(i)] for i in idx]
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(name == 'train'),
            collate_fn=collate_dgl,
        )

    logger.info("Dataset ready: train=%d, valid=%d, test=%d",
                len(loaders['train'].dataset),
                len(loaders['valid'].dataset),
                len(loaders['test'].dataset))
    return loaders, n_classes, edge_feat_dim


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch_g, labels in tqdm(loader, desc="train", leave=False):
        batch_g = batch_g.to(device)
        labels = labels.to(device)

        if batch_g.num_nodes() <= 1:
            continue

        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        optimizer.zero_grad()
        logits = model(batch_g, node_ids)
        loss = nn.functional.cross_entropy(logits, labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_graphs += labels.size(0)

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []

    for batch_g, labels in tqdm(loader, desc="eval", leave=False):
        batch_g = batch_g.to(device)

        if batch_g.num_nodes() <= 1:
            continue

        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = model(batch_g, node_ids)
        y_true.append(labels.view(-1, 1).cpu())
        y_pred.append(logits.argmax(dim=1).view(-1, 1).cpu())

    if not y_true:
        return {'acc': 0.0}

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, path)
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)


def save_best_model(model, path, val_acc):
    torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc}, path)
    logger.info("Best model saved to %s (val_acc=%.4f)", path, val_acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'checkpoint.pt'
    best_model_path = output_dir / 'best_model.pt'

    # Data
    loaders, n_classes, edge_feat_dim = prepare_data(args.batch_size, args.root)
    evaluator = Evaluator('ogbg-ppa')

    # Model
    logger.info("Creating GraphClassifier (hidden=%d, layers=%d, dropout=%.1f)",
                args.hidden_dim, args.n_layers, args.dropout)
    model = GraphClassifier(
        edge_feat_dim=edge_feat_dim,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------------------------------------------------
    # Eval-only mode
    # -----------------------------------------------------------------------
    if args.eval_only:
        if not args.load:
            logger.error("--eval-only requires --load <path>")
            sys.exit(1)
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Loaded model from %s", args.load)

        for split in ['train', 'valid', 'test']:
            perf = evaluate(model, loaders[split], evaluator, device)
            logger.info("%s: %.4f", split.capitalize(), perf[evaluator.eval_metric])
        return

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val = -1.0
    train_curve, valid_curve, test_curve = [], [], []

    for epoch in range(1, args.epochs + 1):
        logger.info("===== Epoch %d / %d =====", epoch, args.epochs)

        train_loss = train_one_epoch(model, loaders['train'], optimizer, device)
        train_perf = evaluate(model, loaders['train'], evaluator, device)
        valid_perf = evaluate(model, loaders['valid'], evaluator, device)
        test_perf = evaluate(model, loaders['test'], evaluator, device)

        metric = evaluator.eval_metric
        logger.info("train_loss=%.4f | train_%s=%.4f | valid_%s=%.4f | test_%s=%.4f",
                    train_loss, metric, train_perf[metric], metric,
                    valid_perf[metric], metric, test_perf[metric])

        train_curve.append(train_perf[metric])
        valid_curve.append(valid_perf[metric])
        test_curve.append(test_perf[metric])

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Best model
        if valid_perf[metric] > best_val:
            best_val = valid_perf[metric]
            save_best_model(model, best_model_path, best_val)

    # Save curves
    torch.save({'train': train_curve, 'valid': valid_curve, 'test': test_curve},
               output_dir / 'curves.pt')

    best_epoch = int(np.argmax(valid_curve)) + 1
    logger.info("Training finished. Best val: %.4f at epoch %d, Test: %.4f",
                valid_curve[best_epoch - 1], best_epoch, test_curve[best_epoch - 1])


if __name__ == '__main__':
    main()
