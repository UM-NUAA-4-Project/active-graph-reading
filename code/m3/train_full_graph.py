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
  python train_full_graph.py --resume best_model.pt   # continue from best checkpoint
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure mycode/ is on the path so that imports like
# "from active_graph_reader.gnn.classifier import GraphClassifier" work.
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

from active_graph_reader.gnn.classifier import GraphClassifier

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
    weight_decay=5e-4,
    scheduler_patience=10,
    early_stop_patience=15,
    device='cuda:0',
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
    p.add_argument('--weight-decay', type=float, default=DEFAULTS['weight_decay'])
    p.add_argument('--scheduler-patience', type=int, default=DEFAULTS['scheduler_patience'])
    p.add_argument('--early-stop-patience', type=int, default=DEFAULTS['early_stop_patience'])
    p.add_argument('--device', type=str, default=DEFAULTS['device'])
    p.add_argument('--output-dir', type=str, default=DEFAULTS['output_dir'])
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training, evaluate saved model only')
    p.add_argument('--load', type=str, default='',
                   help='Path to saved model for eval-only mode')
    p.add_argument('--resume', type=str, default='',
                   help='Resume training from best_model.pt checkpoint')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root directory (default: /mnt/d/ogb-master/dataset)')
    p.add_argument('--label-smoothing', type=float, default=0.0,
                   help='Label smoothing for CrossEntropyLoss (default: 0.0)')
    p.add_argument('--class-weight', action=argparse.BooleanOptionalAction, default=True,
                   help='Use inverse-frequency class weights (default: True)')
    p.add_argument('--finetune-epochs', type=int, default=0,
                   help='Fine-tune best model for N epochs with lower LR (0=skip)')
    p.add_argument('--finetune-from', type=str, default='',
                   help='Skip main training, load this model and fine-tune directly')
    p.add_argument('--finetune-lr', type=float, default=1e-4,
                   help='Learning rate for fine-tuning phase (default: 1e-4)')
    p.add_argument('--finetune-patience', type=int, default=10,
                   help='Early-stop patience for fine-tuning (default: 10)')
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

def train_one_epoch(model, loader, optimizer, device, class_weights=None, label_smoothing=0.0):
    model.train()
    total_loss = 0.0
    correct, total_graphs = 0, 0

    for batch_g, labels in tqdm(loader, desc="train", leave=False):
        batch_g = batch_g.to(device)
        labels = labels.to(device)

        if batch_g.num_nodes() <= 1:
            continue

        node_ids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        optimizer.zero_grad()
        logits = model(batch_g, node_ids)
        loss = nn.functional.cross_entropy(logits, labels.view(-1),
                                              label_smoothing=label_smoothing,
                                              weight=class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels.view(-1)).sum().item()
        total_graphs += labels.size(0)

    return total_loss / max(total_graphs, 1), correct / max(total_graphs, 1)


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


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }, path)
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)


def save_best_model(model, optimizer, scheduler, epoch, path, val_acc):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
    }, path)
    logger.info("Best model saved to %s (epoch=%d, val_acc=%.4f)", path, epoch, val_acc)


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

    # Compute class weights from training labels (inverse frequency)
    if args.class_weight:
        train_labels = []
        for g, lbl in loaders['train'].dataset:
            train_labels.append(lbl.item() if hasattr(lbl, 'item') else int(lbl))
        label_counts = np.bincount(train_labels)
        n_samples = len(train_labels)
        n_classes = len(label_counts)
        # inverse-frequency weight: tail classes get higher weight
        class_weights = n_samples / (n_classes * np.maximum(label_counts, 1))
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
        logger.info("Class weights: min=%.3f max=%.3f",
                    class_weights.min().item(), class_weights.max().item())
    else:
        class_weights = None

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

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=args.scheduler_patience, min_lr=1e-6,
    )

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
    # Skip main training → directly fine-tune from a saved model
    # -----------------------------------------------------------------------
    metric = evaluator.eval_metric

    if args.finetune_from:
        ft_from_path = Path(args.finetune_from)
        if not ft_from_path.exists():
            logger.error("Finetune-from checkpoint not found: %s", ft_from_path)
            sys.exit(1)
        ckpt = torch.load(ft_from_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_val = ckpt.get('val_acc', -1.0)
        best_model_path = ft_from_path  # for reference in fine-tune phase
        logger.info("Loaded model from %s (val_acc=%.4f), skipping main training",
                    ft_from_path, best_val)
        # Jump directly to fine-tuning (skip main training loop + evaluation)
        if args.finetune_epochs <= 0:
            logger.warning("--finetune-from set but --finetune-epochs=0, nothing to do")
            return
    else:
        # -------------------------------------------------------------------
        # Training loop
        # -------------------------------------------------------------------
        start_epoch = 1
        best_val = -1.0
        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                logger.error("Resume checkpoint not found: %s", resume_path)
                sys.exit(1)
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val = ckpt.get('val_acc', -1.0)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            logger.info("Resumed from %s (epoch %d, best_val=%.4f)", resume_path, start_epoch - 1, best_val)

        valid_curve = []
        no_improve_steps = 0

        for epoch in range(start_epoch, args.epochs + 1):
            logger.info("===== Epoch %d / %d =====", epoch, args.epochs)

            train_loss, train_acc = train_one_epoch(
                model, loaders['train'], optimizer, device, class_weights, args.label_smoothing)
            valid_perf = evaluate(model, loaders['valid'], evaluator, device)
            val_acc = valid_perf[metric]

            logger.info("train_loss=%.4f train_acc=%.4f | valid_%s=%.4f | gap=%.4f",
                        train_loss, train_acc, metric, val_acc, train_acc - val_acc)

            valid_curve.append(val_acc)
            scheduler.step(val_acc)

            if epoch % 5 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

            if val_acc > best_val:
                best_val = val_acc
                no_improve_steps = 0
                save_best_model(model, optimizer, scheduler, epoch, best_model_path, best_val)
            else:
                no_improve_steps += 1

            if no_improve_steps >= args.early_stop_patience:
                logger.info("Early stopping at epoch %d — no improvement for %d epochs",
                            epoch, args.early_stop_patience)
                break

        # Final evaluation — load best model and evaluate once
        best_epoch = int(np.argmax(valid_curve)) + start_epoch
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        test_perf = evaluate(model, loaders['test'], evaluator, device)
        train_perf = evaluate(model, loaders['train'], evaluator, device)

        torch.save({
            'valid': valid_curve,
            'train_final': train_perf[metric],
            'test_final': test_perf[metric],
            'best_epoch': best_epoch,
            'best_val': best_val,
            'start_epoch': start_epoch,
        }, output_dir / 'curves.pt')

        logger.info("Training finished. Best val: %.4f at epoch %d | Train: %.4f | Test: %.4f | Gap: %.4f",
                    best_val, best_epoch,
                    train_perf[metric], test_perf[metric],
                    train_perf[metric] - test_perf[metric])

    # -----------------------------------------------------------------------
    # Fine-tuning phase: load best model, train with lower LR from scratch
    # -----------------------------------------------------------------------
    if args.finetune_epochs > 0:
        logger.info("=" * 60)
        logger.info("Starting fine-tuning phase (%d epochs, lr=%.1e)",
                    args.finetune_epochs, args.finetune_lr)

        ft_best_path = output_dir / 'best_model_ft.pt'

        # Load best model weights (discard old optimizer/scheduler state)
        best_ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        logger.info("Loaded best model (val_acc=%.4f) for fine-tuning", best_ckpt.get('val_acc', -1))

        ft_optimizer = optim.AdamW(model.parameters(), lr=args.finetune_lr,
                                    weight_decay=args.weight_decay)
        ft_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            ft_optimizer, mode='max', factor=0.5,
            patience=args.finetune_patience // 2, min_lr=1e-7,
        )

        ft_best_val = best_val
        ft_no_improve = 0
        ft_valid_curve = []

        for epoch in range(1, args.finetune_epochs + 1):
            logger.info("===== Fine-tune Epoch %d / %d =====", epoch, args.finetune_epochs)

            train_loss, train_acc = train_one_epoch(
                model, loaders['train'], ft_optimizer, device,
                class_weights, args.label_smoothing)
            valid_perf = evaluate(model, loaders['valid'], evaluator, device)
            val_acc = valid_perf[metric]

            logger.info("ft_train_loss=%.4f ft_train_acc=%.4f | ft_valid_%s=%.4f | gap=%.4f",
                        train_loss, train_acc, metric, val_acc, train_acc - val_acc)

            ft_valid_curve.append(val_acc)
            ft_scheduler.step(val_acc)

            if val_acc > ft_best_val:
                ft_best_val = val_acc
                ft_no_improve = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': ft_best_val,
                }, ft_best_path)
                logger.info("Fine-tune best saved (epoch=%d, val_acc=%.4f)", epoch, ft_best_val)
            else:
                ft_no_improve += 1

            if ft_no_improve >= args.finetune_patience:
                logger.info("Fine-tune early stop at epoch %d", epoch)
                break

        # Final evaluation — load best fine-tuned model
        if ft_best_path.exists():
            ft_ckpt = torch.load(ft_best_path, map_location=device)
            model.load_state_dict(ft_ckpt['model_state_dict'])
            ft_test_perf = evaluate(model, loaders['test'], evaluator, device)
            ft_train_perf = evaluate(model, loaders['train'], evaluator, device)

            ft_epoch = ft_ckpt['epoch']
            torch.save({
                'valid': ft_valid_curve,
                'train_final': ft_train_perf[metric],
                'test_final': ft_test_perf[metric],
                'best_epoch': ft_epoch,
                'best_val': ft_best_val,
            }, output_dir / 'curves_ft.pt')

            logger.info("Fine-tuning finished. Best val: %.4f (Δ=%.4f) at epoch %d | Train: %.4f | Test: %.4f",
                        ft_best_val, ft_best_val - best_val, ft_epoch,
                        ft_train_perf[metric], ft_test_perf[metric])


if __name__ == '__main__':
    main()
