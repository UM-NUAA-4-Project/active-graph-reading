#!/usr/bin/env python3
"""
小数据集 GIN 训练脚本
=====================

在 sample_small_dataset.py 生成的小数据集上训练 GraphClassifier,
用于快速验证 pipeline 是否跑通, 以及初步 benchmark 信号。

用法:
  python split-tools-analyze/train_small.py
  python split-tools-analyze/train_small.py --virtual_node --epochs 50
  python split-tools-analyze/train_small.py --data_dir dataset/ogbg_ppa/small

输出:
  small_checkpoints/best_model.pt  — 与 train_full_graph.py 格式一致
"""

import os
import sys
import gc
import json
import random
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from dgl.data.utils import load_graphs

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from ogb.graphproppred import collate_dgl
from active_graph_reader.encoder.classifier import GraphClassifier


# ============================================================
# 数据
# ============================================================

class SmallDataset:
    def __init__(self, path, num_classes=None):
        self.graphs, label_dict = load_graphs(path)
        self.labels = label_dict["labels"]
        self._num_classes = num_classes

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def load_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        print(f"[ERROR] 找不到 {path}, 请先运行 sample_small_dataset.py")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 批处理 / 训练 / 评估
# ============================================================

def _prepare_batch(bg, labels, device):
    bg = bg.to(device)
    labels = labels.to(device)
    node_ids = torch.zeros(bg.num_nodes(), dtype=torch.long, device=device)
    return bg, node_ids, labels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for bg, labels in loader:
        bg, node_ids, labels = _prepare_batch(bg, labels, device)
        optimizer.zero_grad()
        out = model(bg, node_ids)
        loss = criterion(out, labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bg.batch_size
        correct += (out.argmax(-1) == labels.view(-1)).sum().item()
        total += bg.batch_size
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for bg, labels in loader:
        bg, node_ids, labels = _prepare_batch(bg, labels, device)
        out = model(bg, node_ids)
        loss = criterion(out, labels.view(-1))
        total_loss += loss.item() * bg.batch_size
        correct += (out.argmax(-1) == labels.view(-1)).sum().item()
        total += bg.batch_size
    return total_loss / total, correct / total


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="小数据集 GIN 训练")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--virtual_node", action="store_true", default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ckpt_dir", type=str, default="small_checkpoints")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--load", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.ckpt_dir, exist_ok=True)

    meta = load_meta(args.data_dir)
    num_classes = meta["num_classes"]
    train_size = meta["train_size"]
    val_size = meta["val_size"]
    test_size = meta["test_size"]

    # ---- 加载数据 ----
    print("=" * 60)
    print("  小数据集 GIN 训练")
    print(f"  {args.data_dir}/  ({train_size+val_size+test_size:,} 图)")
    print(f"  train={train_size:,}  val={val_size:,}  test={test_size:,}")
    print("=" * 60)

    print(f"\n[1] 加载数据 ...")
    train_ds = SmallDataset(os.path.join(args.data_dir, "train.bin"), num_classes)
    val_ds = SmallDataset(os.path.join(args.data_dir, "val.bin"), num_classes)
    test_ds = SmallDataset(os.path.join(args.data_dir, "test.bin"), num_classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               collate_fn=collate_dgl)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             collate_fn=collate_dgl)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              collate_fn=collate_dgl)
    print(f"  已加载, 当前内存正常")

    # ---- 构建模型 ----
    vn_label = "ON" if args.virtual_node else "OFF"
    print(f"\n[2] 构建模型 (emb_dim={args.emb_dim}, layers={args.num_layers}, "
          f"VN={vn_label}, residual={args.residual})")
    model = GraphClassifier(
        edge_feat_dim=7, n_classes=num_classes,
        hidden_dim=args.emb_dim, n_layers=args.num_layers,
        dropout=args.dropout, virtual_node=args.virtual_node,
        residual=args.residual,
    ).to(device)
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    best_path = os.path.join(args.ckpt_dir, "best_model.pt")

    # ---- eval only ----
    if args.eval_only:
        load_path = args.load or best_path
        print(f"\n[eval only] 加载: {load_path}")
        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  记录 val_acc = {ckpt.get('val_acc', 'N/A')}")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        print(f"  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")
        return

    # ---- 训练 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
    )

    print(f"\n[3] 训练 ({args.epochs} epochs, patience={args.patience})")
    best_val_acc = 0.0
    patience_cnt = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        is_best = val_acc > best_val_acc
        marker = " *" if is_best else ""

        print(f"  [{ep:>3d}]  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={lr:.2e}{marker}")

        if is_best:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save({
                "model_state_dict": {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()},
                "val_acc": best_val_acc,
                "args": vars(args),
            }, best_path)
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            print(f"\n  [早停] {args.patience} epoch 未提升, 结束")
            break

    elapsed = time.time() - t0
    print(f"\n  训练完成: {elapsed:.0f}s, 最佳 val_acc={best_val_acc:.4f}")

    # ---- 测试 ----
    print(f"\n[4] 测试评估 ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    print(f"\n  模型: {best_path}")
    print(f"  可用于: python m3/benchmark_m3.py --pretrained {best_path} --quick")

    del train_ds, val_ds, test_ds, model
    gc.collect()


if __name__ == "__main__":
    main()
