"""跨 checkpoint 权重迁移：生成任意数据集 reader 的初始化 checkpoint。

自 m7/transfer.py 的 CLI 用法独立化（transfer_weights 函数已收编至
tools.ckpt_io，m5 patch_old_checkpoint 的泛化版）。

机制：按源 checkpoint 结构构建目标 reader：
  - 同键同形 → 复制（transferred）；
  - 仅最后一维更窄 → 零填充后复制（padded，旧 v0/v1 mlp 输入宽度）；
  - 其余（维度不匹配、新增层）→ 保留随机初始化（skipped）。

用法:
  python code/train_reader/transfer.py --dataset cifar10 --root D:\\superpixels \
      --source result/ppa/experiments/rl_asymmetric_ex1/base/rl/checkpoint_epoch4.pt \
      --reader-kind base --output result/cifar10/experiments/ppa_transfer/base/rl/init.pt
"""

import argparse
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from datasets import get_dataset  # noqa: E402
from tools import cli as cli_utils  # noqa: E402
from tools.ckpt_io import save_epoch_checkpoint  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from train_reader.core import bootstrap_reader  # noqa: E402


def resolve(p) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def main():
    parser = argparse.ArgumentParser(description='跨 checkpoint 权重迁移（初始化产物）')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--source', required=True, help='源 reader checkpoint')
    cli_utils.add_reader_args(parser, reader_kind_default='base',
                              temperature_default=0.2)
    parser.add_argument('--output', required=True, help='输出 checkpoint 路径')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cli_utils.attach_defaults(args, parser)

    device = torch.device(args.device)
    src_path = resolve(args.source)
    src_ckpt = torch.load(str(src_path), map_location='cpu', weights_only=False)

    reader, (n_transferred, n_padded, skipped) = bootstrap_reader(
        src_ckpt, device, stochastic=True, temperature=args.temperature)

    out_path = resolve(args.output)
    config = {
        'reader': reader.config.to_dict(),
        'dataset': args.dataset,
        'transferred_from': str(src_path),
        'n_transferred': n_transferred,
        'n_padded': n_padded,
        'n_skipped': len(skipped),
    }
    save_epoch_checkpoint(out_path, epoch=0, reader=reader, optimizer=None,
                          config=config, training_log=[])
    print(f"Transferred: copied={n_transferred}, padded={n_padded}, "
          f"skipped={len(skipped)}")
    if skipped:
        print(f"  skipped: {skipped[:6]}{'...' if len(skipped) > 6 else ''}")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
