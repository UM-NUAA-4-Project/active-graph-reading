"""M4 轨迹分片数据整合脚本。

将 m4/output/ 下的分片 .pt 文件按 reveal_ratio 合并为完整文件。

用法:
  python ./code/m4/merge_shards.py --input-dir ./code/m4/output --output-dir ./code/m4/output
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='M4 Trajectory Shard Merger')
    p.add_argument('--input-dir', type=str, default=None,
                   help='Directory containing shard .pt files (default: m4/data/stage1_merged)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory for merged files (default: same as input-dir)')
    return p.parse_args()


def extract_ratio(filename: str) -> str:
    """从文件名提取 ratio 字符串, 如 trajectories_r0_100_shard01_of_10.pt → 0_100."""
    m = re.match(r'trajectories_r(\d+_\d+)_shard', filename)
    return m.group(1) if m else None


def main():
    args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else _script_dir / 'data' / 'stage1_merged'
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_dir.glob('trajectories_r*_shard*_of_*.pt'))
    if not pt_files:
        logger.error("No shard files found in %s", input_dir)
        sys.exit(1)

    ratio_groups = defaultdict(list)
    for f in pt_files:
        ratio_str = extract_ratio(f.name)
        if ratio_str is None:
            logger.warning("Cannot parse ratio from %s, skipping", f.name)
            continue
        ratio_groups[ratio_str].append(f)

    logger.info("Found %d files in %d ratio groups", len(pt_files), len(ratio_groups))
    for ratio_str, files in sorted(ratio_groups.items()):
        logger.info("  ratio=%s: %d shards", ratio_str, len(files))

    for ratio_str, files in sorted(ratio_groups.items()):
        logger.info("Merging ratio=%s (%d files)...", ratio_str, len(files))

        merged_trajs = None
        merged_config = None
        total_graphs = 0

        for f in sorted(files):
            data = torch.load(f, map_location='cpu', weights_only=False)
            trajs = data['trajectories']
            config = data['config']

            if merged_trajs is None:
                merged_trajs = {tn: [] for tn in config['teacher_names']}
                merged_config = {
                    'reveal_ratio': config['reveal_ratio'],
                    'n_seeds': config['n_seeds'],
                    'n_graphs': 0,
                    'teacher_names': config['teacher_names'],
                }

            for teacher_name in merged_config['teacher_names']:
                if teacher_name in trajs:
                    merged_trajs[teacher_name].extend(trajs[teacher_name])
                else:
                    logger.warning("  Teacher '%s' missing in %s", teacher_name, f.name)

            total_graphs += config['n_graphs']

        merged_config['n_graphs'] = total_graphs

        total_trajs = sum(
            sum(len(trajs) for trajs in per_teacher)
            for per_teacher in merged_trajs.values()
        )
        logger.info("  Merged: %d graphs, %d trajectories", total_graphs, total_trajs)
        for tn in merged_config['teacher_names']:
            non_empty = sum(1 for trajs in merged_trajs[tn] if len(trajs) > 0)
            total_t = sum(len(trajs) for trajs in merged_trajs[tn])
            logger.info("    %25s: %d/%d graphs non-empty, %d trajectories",
                        tn, non_empty, total_graphs, total_t)

        output_path = output_dir / f'trajectories_merged_r{ratio_str}.pt'
        torch.save({
            'trajectories': merged_trajs,
            'config': merged_config,
        }, output_path)
        logger.info("  Saved to %s", output_path)


if __name__ == '__main__':
    main()
