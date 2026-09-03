"""一次性迁移工具：给无内嵌 config 的旧分类器 checkpoint 补写 config。

背景（2026-09-03 起）：classifier 重建一律以 checkpoint 内嵌 config 为准
（load_state_dict 的 shape 校验负责一致性），不再从权重 shape 反推结构。
m3 时代的 checkpoint（如 result/ppa/classifier/best_model_ft.pt）没有
内嵌 config，用本脚本一次性补写后即可被 load_classifier 加载。

用法：
  # 预览推断结果（不写盘）
  python code/tools/migrate_legacy_classifier_ckpt.py result/ppa/classifier/best_model_ft.pt
  # 确认无误后写回
  python code/tools/migrate_legacy_classifier_ckpt.py <ckpt...> --write
  # 可选：顺手记录数据集名
  python code/tools/migrate_legacy_classifier_ckpt.py <ckpt...> --write --dataset ppa

推断逻辑仅存在于本脚本（legacy-only）：按 state_dict 键名区分 backbone
（GIN: convs.i.mlp.0.weight；GAT: convs.i.fc.weight + convs.i.attn_l），
hidden/edge_feat/n_classes 从权重 shape 读取；dropout 无法推断，取 0.5
（如与实际训练值不同，请手改写回后的 config）。
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from classifier import ClassifierConfig

CONV_RE = re.compile(r'^encoder\.convs\.(\d+)\.mlp\.0\.weight$')   # GIN
GAT_FC_RE = re.compile(r'^encoder\.convs\.(\d+)\.fc\.weight$')     # GAT
GAT_ATTN_RE = re.compile(r'^encoder\.convs\.(\d+)\.attn_l$')


def infer_classifier_config(state_dict: dict) -> ClassifierConfig:
    """旧 checkpoint 权重 shape → ClassifierConfig（仅迁移脚本使用）。"""
    gin, gat, attn = {}, {}, {}
    for key, tensor in state_dict.items():
        if (m := CONV_RE.match(key)):
            gin[int(m.group(1))] = int(tensor.shape[0]) // 2   # mlp.0: hidden*2
        elif (m := GAT_FC_RE.match(key)):
            gat[int(m.group(1))] = int(tensor.shape[0])
        elif (m := GAT_ATTN_RE.match(key)):
            attn[int(m.group(1))] = int(tensor.shape[1])

    if gat:
        backbone, hidden_dims = 'gat', [gat[i] for i in sorted(gat)]
        n_heads = ([attn[i] for i in sorted(attn)]
                   if len(attn) == len(hidden_dims) else None)
    elif gin:
        backbone, hidden_dims, n_heads = 'gin', [gin[i] for i in sorted(gin)], None
    else:
        raise ValueError("state_dict 中未找到 encoder.convs.* 权重，无法推断结构")

    node_feat_dim = 0
    if 'encoder.node_proj.weight' in state_dict:
        node_feat_dim = int(state_dict['encoder.node_proj.weight'].shape[1])

    return ClassifierConfig(
        backbone=backbone,
        edge_feat_dim=int(state_dict['encoder.convs.0.edge_encoder.weight'].shape[1]),
        node_feat_dim=node_feat_dim,
        n_classes=int(state_dict['head.weight'].shape[0]),
        hidden_dims=hidden_dims,
        dropouts=[0.5] * len(hidden_dims),
        n_heads=n_heads,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('ckpts', nargs='+', help='旧分类器 checkpoint 路径')
    parser.add_argument('--write', action='store_true',
                        help='写回（默认仅预览推断结果）')
    parser.add_argument('--dataset', default='',
                        help='可选，写入 config.dataset 便于溯源')
    args = parser.parse_args()

    for path in args.ckpts:
        path = Path(path)
        ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
        embedded = ckpt.get('config')
        if isinstance(embedded, dict) and (
                'classifier' in embedded or 'hidden_dim' in embedded
                or 'hidden_dims' in embedded):
            print(f'{path.name}: 已有内嵌 config，跳过')
            continue
        cfg = infer_classifier_config(ckpt.get('model_state_dict',
                                               ckpt.get('state_dict', {})))
        print(f'{path.name}: 推断 → {cfg.to_dict()}')
        if not args.write:
            continue
        new_config = {'classifier': cfg.to_dict()}
        if args.dataset:
            new_config['dataset'] = args.dataset
        ckpt['config'] = new_config
        torch.save(ckpt, str(path))
        print(f'{path.name}: 已写回（dropout 默认 0.5，如与实际不符请手改）')


if __name__ == '__main__':
    main()
