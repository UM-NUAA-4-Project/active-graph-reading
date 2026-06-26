"""Fix traj.graph_idx in existing trajectory .pt files.

Remaps local positional indices (0, 1, 2, ...) to original OGB dataset indices,
so train_bc.py loads the correct graph for each trajectory.

Usage:
  # Dry-run to preview changes
  python ./code/m4/process-data/fix_graph_idx.py \
    --data-dir ./code/m4/data/stage4_balanced --root /mnt/data --dry-run

  # Apply fixes
  python ./code/m4/process-data/fix_graph_idx.py \
    --data-dir ./code/m4/data/stage4_balanced --root /mnt/data

  # With max-graphs (if generation used --max-graphs N)
  python ./code/m4/process-data/fix_graph_idx.py \
    --data-dir ./code/m4/data/stage4_balanced --root /mnt/data --max-graphs 1000
"""

import argparse
import logging
import sys
from pathlib import Path

# pickle 反序列化 Trajectory 需要 m4 包可导入
_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent.parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

import torch
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Fix graph_idx in trajectory files')
    p.add_argument('--data-dir', type=str, required=True,
                   help='Directory containing trajectory .pt files')
    p.add_argument('--root', type=str, required=True,
                   help='OGB dataset root (e.g., D:\\ogb-master\\dataset)')
    p.add_argument('--max-graphs', type=int, default=0,
                   help='Limit to first N training graphs (0 = use all)')
    p.add_argument('--dry-run', action='store_true',
                   help='Preview changes without modifying files')
    return p.parse_args()


def build_mapping(root: str, max_graphs: int):
    """Build local_idx → ogb_idx mapping from raw training split CSV."""
    logger.info("Loading OGB training split (root=%s)...", root)
    train_path = Path(root) / 'ogbg_ppa' / 'split' / 'species' / 'train.csv.gz'
    train_idx = pd.read_csv(train_path, compression='gzip', header=None).values.flatten().tolist()
    if max_graphs > 0:
        train_idx = train_idx[:max_graphs]
        logger.info("Limited to %d training graphs", max_graphs)
    logger.info("Mapping: local [0..%d) → OGB indices", len(train_idx))
    return train_idx


def fix_file(filepath: Path, mapping: list, dry_run: bool):
    """Remap graph_idx for all trajectories in a .pt file."""
    data = torch.load(str(filepath), map_location='cpu', weights_only=False)
    trajs_dict = data['trajectories']

    total_trajs = 0
    changes = []

    for teacher_name, per_graph_list in trajs_dict.items():
        for g_list_idx, traj_list in enumerate(per_graph_list):
            for t_idx, traj in enumerate(traj_list):
                total_trajs += 1
                old_idx = traj.graph_idx
                if old_idx < 0 or old_idx >= len(mapping):
                    logger.error(
                        "  [%s] graph_idx=%d out of mapping range [0, %d) in %s, graph_list[%d], traj[%d]",
                        filepath.name, old_idx, len(mapping), teacher_name, g_list_idx, t_idx,
                    )
                    continue
                new_idx = mapping[old_idx]
                if old_idx != new_idx:
                    changes.append((old_idx, new_idx))
                    traj.graph_idx = new_idx

    if changes:
        unique_mappings = set(changes)
        logger.info("  %s: %d/%d trajectories remapped (%d unique mappings)",
                     filepath.name, len(changes), total_trajs, len(unique_mappings))
    else:
        logger.info("  %s: %d trajectories, no changes needed", filepath.name, total_trajs)

    if not dry_run and changes:
        # Create backup before overwriting
        bak_path = filepath.with_suffix(filepath.suffix + '.bak')
        if not bak_path.exists():
            import shutil
            shutil.copy2(filepath, bak_path)
            logger.info("  Backup: %s", bak_path.name)

        torch.save(data, filepath)
        logger.info("  Saved: %s", filepath.name)

    return len(changes), total_trajs


def main():
    args = parse_args()

    mapping = build_mapping(args.root, args.max_graphs)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    pt_files = sorted(data_dir.glob('trajectories*.pt'))
    if not pt_files:
        logger.error("No trajectory .pt files found in %s", data_dir)
        sys.exit(1)

    logger.info("Found %d trajectory file(s) in %s", len(pt_files), data_dir)
    if args.dry_run:
        logger.info("=== DRY RUN — no files will be modified ===")

    total_changes = 0
    total_trajs = 0
    for f in pt_files:
        changes, n_trajs = fix_file(f, mapping, args.dry_run)
        total_changes += changes
        total_trajs += n_trajs

    logger.info("=== Summary: %d/%d trajectories remapped across %d files ===",
                 total_changes, total_trajs, len(pt_files))

    if args.dry_run:
        logger.info("Re-run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
