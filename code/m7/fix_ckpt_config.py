"""修复 classifier checkpoint 的 config：从权重实际形状修正 hidden_dim/n_layers 等。

背景：train_classifier.py 微调继承 bug——config 保存了 args 默认值而非实际结构，
导致下游（train_rl/eval）按错误 config 构建模型、加载失败。
用法: python fix_ckpt_config.py <path1.pt> [path2.pt ...]
"""
import re
import sys

import torch

for p in sys.argv[1:]:
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    cfg = dict(ckpt.get('config', {}))
    if 'encoder.node_proj.weight' in sd:
        cfg['hidden_dim'] = sd['encoder.node_proj.weight'].shape[0]
        cfg['node_feat_dim'] = sd['encoder.node_proj.weight'].shape[1]
    conv_idx = [int(m.group(1)) for k in sd
                if (m := re.match(r'encoder\.convs\.(\d+)\.mlp\.0\.weight', k))]
    if conv_idx:
        cfg['n_layers'] = max(conv_idx) + 1
    cfg['edge_feat_dim'] = sd['encoder.convs.0.edge_encoder.weight'].shape[1]
    cfg['n_classes'] = sd['head.weight'].shape[0]
    cfg.setdefault('dropout', 0.5)
    ckpt['config'] = cfg
    torch.save(ckpt, p)
    print(f'fixed {p}: {cfg}')
