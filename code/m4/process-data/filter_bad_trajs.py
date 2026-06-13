"""过滤轨迹文件中 graph_idx 指向不一致图的坏轨迹。

检查:
  1. seed_node 是否在图中存在
  2. final_pred 是否与 dataset[graph_idx] 的 label 一致

不一致说明该轨迹的 graph_idx 指向了错误的图，予以丢弃。

用法:
  python ./code/m4/process-data/filter_bad_trajs.py \
    --input-dir ./code/m4/data/stage4_balanced \
    --output-dir ./code/m4/data/stage5_filtered \
    --root /mnt/data
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from ogb.graphproppred import DglGraphPropPredDataset

_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent.parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Filter bad trajectories')
    p.add_argument('--input-dir', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--root', type=str, required=True,
                   help='OGB dataset root')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = DglGraphPropPredDataset(name='ogbg-ppa', root=args.root)

    pt_files = sorted(input_dir.glob('trajectories*.pt'))
    if not pt_files:
        logger.error("No trajectory files found in %s", input_dir)
        sys.exit(1)

    pt_files = [f for f in pt_files if not f.name.endswith('.bak')]

    grand_before = 0
    grand_after = 0
    grand_bad = 0

    for fpath in pt_files:
        data = torch.load(str(fpath), map_location='cpu', weights_only=False)
        trajs_dict = data['trajectories']
        config = data.get('config', {})
        teacher_names = config.get('teacher_names', list(trajs_dict.keys()))
        n_graphs = len(trajs_dict[teacher_names[0]]) if teacher_names else 0

        filtered = {tn: [[] for _ in range(n_graphs)] for tn in teacher_names}
        before = 0
        after = 0
        bad = 0

        # 缓存图 label，避免重复加载
        label_cache = {}

        for tn in teacher_names:
            for g_list_idx, traj_list in enumerate(trajs_dict[tn]):
                for traj in traj_list:
                    before += 1
                    gid = traj.graph_idx

                    # 获取图 label（缓存避免重复加载）
                    if gid not in label_cache:
                        try:
                            g, lbl = ds[int(gid)]
                            label_cache[gid] = (
                                g.num_nodes(),
                                int(lbl.item()) if hasattr(lbl, 'item') else int(lbl),
                            )
                        except Exception as e:
                            logger.warning("  Load error graph_idx=%d: %s", gid, e)
                            label_cache[gid] = (0, -1)

                    n_nodes, label = label_cache[gid]

                    # 检查 seed_node 和 label
                    if traj.seed_node >= n_nodes:
                        bad += 1
                        continue
                    if label >= 0 and traj.final_pred != label:
                        bad += 1
                        continue

                    filtered[tn][g_list_idx].append(traj)
                    after += 1

        grand_before += before
        grand_after += after
        grand_bad += bad

        fname = os.path.basename(fpath)
        logger.info("%s: %d -> %d (removed %d)", fname, before, after, bad)

        out_path = output_dir / fname
        torch.save({'trajectories': filtered, 'config': config}, out_path)
        logger.info("  Saved to %s", out_path)

    logger.info("=== Total: %d -> %d (removed %d, %.2f%%) ===",
                 grand_before, grand_after, grand_bad,
                 grand_bad / grand_before * 100 if grand_before > 0 else 0)


if __name__ == '__main__':
    main()
