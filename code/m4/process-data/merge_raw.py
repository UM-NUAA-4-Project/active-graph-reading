"""M4 原始轨迹合并脚本 — 将任意来源的 trajectory .pt 文件按 budget ratio 合并。

支持的文件命名格式:
  - 分片文件:  trajectories_r{ratio}_shard{NN}_of_{NN}.pt
  - 补充文件:  trajectories_r{ratio}_of_{label}.pt
  - 已合并文件: trajectories_merged_r{ratio}.pt
  - 筛选文件:   trajectories_filtered_r{ratio}.pt

用法:
  python code/m4/process-data/merge_raw.py \
      --input-dirs code/m4/data/stage1_merged code/m4/data/stage3_trimmed \
      --output-dir code/m4/process-data
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent.parent  # process-data/ → m4/ → code/
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='M4 Raw Trajectory Merger')
    p.add_argument('--input-dirs', type=str, nargs='+', required=True,
                   help='Directories containing trajectory .pt files (shards, supplement, merged)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: code/m4/process-data)')
    return p.parse_args()


def extract_ratio(filename: str) -> str | None:
    """从文件名提取 ratio 字符串, 如 '0_050'。"""
    # 匹配 _r{ratio}_ 模式，ratio = digits_digits
    m = re.search(r'_r(\d+_\d+)[_.]', filename)
    if m:
        return m.group(1)
    return None


def extract_label(filename: str) -> int | None:
    """从补充文件名提取 target_label, 如 trajectories_r0_050_of_13.pt → 13。"""
    m = re.search(r'_of_(\d+)\.pt$', filename)
    if m:
        return int(m.group(1))
    return None


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 trajectory .pt 文件
    all_files: list[Path] = []
    for dir_str in args.input_dirs:
        d = Path(dir_str)
        if not d.is_dir():
            logger.warning("Directory not found, skipping: %s", d)
            continue
        pt_files = sorted(d.glob('trajectories*.pt'))
        logger.info("Found %d .pt files in %s", len(pt_files), d)
        all_files.extend(pt_files)

    if not all_files:
        logger.error("No trajectory .pt files found in %s", args.input_dirs)
        sys.exit(1)

    # 按 ratio 分组
    ratio_groups: dict[str, list[Path]] = defaultdict(list)
    skipped = 0
    for f in all_files:
        ratio_str = extract_ratio(f.name)
        if ratio_str is None:
            logger.warning("Cannot extract ratio from '%s', skipping", f.name)
            skipped += 1
            continue
        ratio_groups[ratio_str].append(f)

    logger.info("Total: %d files in %d ratio groups (%d skipped)",
                 len(all_files), len(ratio_groups), skipped)
    for ratio_str, files in sorted(ratio_groups.items()):
        logger.info("  ratio=%s: %d files", ratio_str, len(files))

    # 逐 ratio 合并
    for ratio_str, files in sorted(ratio_groups.items()):
        logger.info("=== Merging ratio=%s (%d files) ===", ratio_str, len(files))

        merged_trajs: dict[str, list] | None = None
        merged_config: dict | None = None
        total_graphs = 0
        supplement_labels: list[int] = []

        for f in sorted(files):
            data = torch.load(f, map_location='cpu', weights_only=False)
            trajs = data['trajectories']
            config = data['config']

            if merged_trajs is None:
                teacher_names = config['teacher_names']
                merged_trajs = {tn: [] for tn in teacher_names}
                merged_config = {
                    'reveal_ratio': config['reveal_ratio'],
                    'n_seeds': config.get('n_seeds', 0),
                    'n_graphs': 0,
                    'teacher_names': teacher_names,
                }

            # 检查 teacher 一致
            if set(config['teacher_names']) != set(merged_config['teacher_names']):
                logger.warning(
                    "  Teacher mismatch in %s: file has %s, merged has %s. Skipping.",
                    f.name, config['teacher_names'], merged_config['teacher_names'])
                continue

            # 合并 trajectories
            for tn in merged_config['teacher_names']:
                if tn in trajs:
                    merged_trajs[tn].extend(trajs[tn])
                else:
                    logger.warning("  Teacher '%s' missing in %s", tn, f.name)

            total_graphs += config.get('n_graphs', 0)

            label = extract_label(f.name)
            if label is not None:
                supplement_labels.append(label)

        if merged_trajs is None:
            logger.error("No valid data for ratio=%s", ratio_str)
            continue

        merged_config['n_graphs'] = total_graphs
        if supplement_labels:
            merged_config['supplement_labels'] = sorted(supplement_labels)

        # 统计
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
        if supplement_labels:
            logger.info("  Supplement labels: %s", supplement_labels)

        ratio_float = float(ratio_str.replace('_', '.'))
        output_path = output_dir / f'trajectories_merged_r{ratio_str}.pt'
        torch.save({
            'trajectories': merged_trajs,
            'config': merged_config,
        }, output_path)
        logger.info("  Saved to %s", output_path)


if __name__ == '__main__':
    main()
