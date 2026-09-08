"""Reader checkpoint 维度扩展（128 → 256 等）：生成"桥接 checkpoint"。

问题：`--resume` 按 checkpoint config 严格重建（改不了维度）；vanilla
`--bootstrap` 按**源** config 重建 reader（CLI 的 --hidden-dim 在 bootstrap
路径被忽略）；而 tools.ckpt_io.transfer_weights 的零填充规则只覆盖"末维
变宽"（输入维变化），对 Linear 进出维同时变化的宽度扩展会整条跳过。

方案（本脚本）：加载源 ckpt → 复制其 ReaderConfig 并改宽度 → 按 256 重建
reader（得到目标 shape 与默认初始化）→ **左上子块复制**（src 块写入目标
张量的 [:s0, :s1, ...]，扩展区保留目标自身默认初始化；BN 统计新通道用默认
mean=0/var=1）→ 存成 256 形状的桥接 ckpt（epoch=0、无 optimizer）。
之后照常：

  python code/train_reader/train_rl.py --bootstrap <bridge.pt> ...
  # bootstrap 按 bridge 的 config 建 256 reader，transfer_weights 全同形、
  # 全量迁移（transferred=全部，padded/skipped≈0）

语义注记：左上复制保留一个可用的 128 子网络（权重连续性），扩展通道为
新初始化、靠后续训练分化；这不是信息无损的维度提升——隐藏宽度扩展本质
上只能"热启动"而非等价迁移。

用法:
  python code/tools/expand_reader_ckpt.py \
      --checkpoint code/result/ppa/experiments/ev/evidence/rl/checkpoint_epoch4.pt \
      --dim 256 \
      --output code/result/ppa/experiments/ev/evidence/rl/bridge_256.pt
"""

import argparse
import datetime
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import torch  # noqa: E402

from active_graph_reader import build_reader as _build_reader  # noqa: E402
from active_graph_reader.config import reader_config_from_checkpoint  # noqa: E402
from tools.ckpt_io import strip_classifier_keys  # noqa: E402
from tools.paths import resolve  # noqa: E402


def expand_state_dict(src_sd: dict, tgt_sd: dict):
    """src → 目标形状：同形复制；各维皆不增 → 左上子块复制；否则保留默认。

    Returns: (n_same, n_block, n_default)
    """
    n_same = n_block = n_default = 0
    out = {}
    for k, tgt in tgt_sd.items():
        src = src_sd.get(k)
        if src is None or not torch.is_tensor(src) or not torch.is_tensor(tgt):
            out[k] = tgt
            n_default += 1
            continue
        if src.shape == tgt.shape:
            out[k] = src.clone()
            n_same += 1
            continue
        if src.dim() == tgt.dim() and all(s <= t for s, t in zip(src.shape, tgt.shape)):
            block = tgt.clone()                       # 扩展区 = 目标默认初始化
            sl = tuple(slice(0, s) for s in src.shape)
            block[sl] = src                           # 左上子块 = 源权重
            out[k] = block
            n_block += 1
        else:                                         # 形状不兼容（收缩等）：保留默认
            out[k] = tgt
            n_default += 1
    return out, (n_same, n_block, n_default)


def main():
    parser = argparse.ArgumentParser(description='Reader ckpt 维度扩展 → 桥接 checkpoint')
    parser.add_argument('--checkpoint', required=True, help='源 reader ckpt（如 128 宽）')
    parser.add_argument('--dim', type=int, required=True, help='新 hidden_dim（如 256）')
    parser.add_argument('--scorer-dim', type=int, default=0,
                        help='新 scorer_hidden_dim（默认跟随 --dim）')
    parser.add_argument('--output', required=True, help='桥接 ckpt 输出路径')
    args = parser.parse_args()

    src_path = resolve(args.checkpoint)
    src_ckpt = torch.load(str(src_path), map_location='cpu', weights_only=False)
    sd = strip_classifier_keys(dict(
        src_ckpt.get('reader_state_dict', src_ckpt.get('state_dict'))))
    cfg = reader_config_from_checkpoint(src_ckpt, sd)

    old_h = list(cfg.encoder_hidden_dims)
    old_s = list(cfg.scorer_hidden_dims)
    cfg.encoder_hidden_dims = [args.dim] * len(old_h)
    cfg.scorer_hidden_dims = [args.scorer_dim or args.dim] * len(old_s)
    print(f"source: {src_path}")
    print(f"width:  encoder {old_h}->{cfg.encoder_hidden_dims} | "
          f"scorer {old_s}->{cfg.scorer_hidden_dims} (层数不变)")

    tgt_reader = _build_reader(cfg)
    tgt_sd = tgt_reader.state_dict()
    expanded_sd, (n_same, n_block, n_default) = expand_state_dict(sd, tgt_sd)
    tgt_reader.load_state_dict(expanded_sd, strict=True)
    total = n_same + n_block + n_default
    print(f"transfer: same={n_same}  block-copied={n_block}  default-init={n_default}"
          f"  (of {total})")

    bridge = {
        'reader_state_dict': expanded_sd,
        'config': {**(src_ckpt.get('config') or {}),
                   'reader': cfg.to_dict(),
                   'expanded_from': str(src_path),
                   'expansion': {'encoder_hidden_dims': [old_h, cfg.encoder_hidden_dims],
                                 'scorer_hidden_dims': [old_s, cfg.scorer_hidden_dims],
                                 'rule': 'top-left block copy, extended region default-init',
                                 'timestamp': datetime.datetime.now()
                                 .isoformat(timespec='seconds')}},
        'epoch': 0,
        'training_log': [],
    }
    out_path = Path(args.output)
    if out_path.parts and out_path.parts[0] == 'code':
        out_path = Path(*out_path.parts[1:])   # 写文件约定：result/ 在 code/ 下
    out_path = _CODE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge, str(out_path))
    print(f"bridge ckpt saved: {out_path}")
    print("后续照常: train_rl.py --bootstrap <该文件> ...（bootstrap 会按其 config "
          "建 256 reader，transfer_weights 全同形全量迁移）")


if __name__ == '__main__':
    main()
