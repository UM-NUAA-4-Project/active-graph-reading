"""Sample 1 graph per class from each split for M4 smoke testing.

Produces a .pt file containing ~111 graphs (37 classes x 3 splits)
that can be used for fast smoke testing of teacher trajectory generation.

Usage:
  python -m m4.sample_test_graphs --root D:\\ogb-master\\dataset
  python -m m4.sample_test_graphs --root D:\\ogb-master\\dataset --output /tmp/smoke.pt
"""

import argparse
import sys

from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from active_graph_reader.dataloader.loader import PPADataset


def parse_args():
    p = argparse.ArgumentParser(description='Sample test graphs for M4 smoke test')
    p.add_argument('--root', type=str, default=r'D:\ogb-master\dataset',
                   help='OGB dataset root directory')
    p.add_argument('--output', type=str, default=None,
                   help='Output file path (default: smoke_test_graphs.pt in script dir)')
    return p.parse_args()


def main():
    args = parse_args()

    output = Path(args.output) if args.output else Path(__file__).parent / 'smoke_test_graphs.pt'

    print(f"Loading OGBG-PPA from {args.root}...")
    dataset = PPADataset(root=args.root)
    n_classes = int(dataset.num_classes)

    samples: dict = {}

    for split_name, indices in [
        ('train', dataset.train_indices),
        ('valid', dataset.valid_indices),
        ('test', dataset.test_indices),
    ]:
        class_to_entry: dict = {}
        for idx in indices:
            g, label = dataset[idx]
            if label not in class_to_entry:
                class_to_entry[label] = {
                    'graph_idx': idx,
                    'graph': g,
                    'label': label,
                }
            if len(class_to_entry) == n_classes:
                break

        samples[split_name] = [
            class_to_entry[c] for c in sorted(class_to_entry)
        ]

        missing = sorted(set(range(n_classes)) - set(class_to_entry))
        print(f"  {split_name}: {len(class_to_entry)}/{n_classes} classes"
              + (f", missing: {missing}" if missing else ""))

    torch.save(samples, str(output))
    print(f"\nSaved {sum(len(v) for v in samples.values())} graphs to {output}")


if __name__ == '__main__':
    main()
