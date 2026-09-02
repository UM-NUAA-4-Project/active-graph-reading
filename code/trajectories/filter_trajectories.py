"""轨迹过滤：相似去重 → 每图 top-k + 教师多样性（供 train_bc.py 使用）。

自 m7/filter_trajectories.py 迁入（逻辑在 trajectories/core/filtering.py，
本脚本为薄 CLI；旧 m5/m4 目录名 --input-dir 直接兼容）。

用法:
  python code/trajectories/filter_trajectories.py --input-dir data/cifar10/trajectories
  python code/trajectories/filter_trajectories.py --sim-threshold 0.90 --top-k 5 --no-diversity
"""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tools.paths import CODE_DIR  # noqa: E402
from trajectories.core.filtering import filter_file  # noqa: E402


def ratio_key_from_name(fname: str) -> str:
    """'trajectories_r0_050.pt' / 'trajectories_r0_050_shard1of4.pt' -> '0_050'。"""
    stem = Path(fname).stem
    tail = stem.rsplit('_r', 1)[1]
    return tail.split('_shard')[0]


def resolve(p: str) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def main():
    parser = argparse.ArgumentParser(
        description='轨迹过滤（相似去重 → 每图 top-k + 教师多样性）')
    parser.add_argument('--input-dir', required=True,
                        help='输入目录（generate_trajectories.py 输出）')
    parser.add_argument('--output-dir', required=True,
                        help='输出目录（train_bc.py --data-dir 指向这里）')
    parser.add_argument('--sim-threshold', type=float, default=0.85,
                        help='同图去重的余弦相似度阈值')
    parser.add_argument('--top-k', type=int, default=3,
                        help='去重后每图保留最大轨迹数')
    parser.add_argument('--no-diversity', action='store_true',
                        help='关闭教师类型多样性保证（默认开启）')
    parser.add_argument('--teachers', nargs='+', default=None,
                        help='教师名列表（默认取各文件 config.teachers）')
    args = parser.parse_args()

    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}（先运行 generate_trajectories.py）")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('trajectories_r*.pt'))
    if not files:
        raise SystemExit(f"输入目录无 trajectories_r*.pt：{input_dir}")
    require_diversity = not args.no_diversity

    report = {}
    for fpath in tqdm(files, desc="Filtering"):
        key = ratio_key_from_name(fpath.name)
        out = filter_file(fpath, args.teachers, args.top_k, require_diversity,
                          args.sim_threshold)
        report.setdefault(key, []).append(out['report'])
        save_path = output_dir / fpath.name
        torch.save(out['data'], str(save_path))
        r = out['report']
        print(f"  {fpath.name}: {r['n_before']} -> dedup {r['n_after_dedup']} "
              f"(removed {r['dedup_removed']}) -> top-k {r['n_after']} "
              f"(graphs {r['n_graphs_after']}, diverse {r['diverse_graphs']}, "
              f"single {r['single_teacher_graphs']})")

    report_path = output_dir / 'filter_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")
    print(f"下一步: python code/train_reader/train_bc.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    main()
