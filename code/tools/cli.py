"""CLI 公共参数与 config 合并的胶水。

所有训练/生成脚本共享的参数组定义 + argparse defaults 附加
（merge_args_with_config 需要 parser 默认值做"CLI 是否被显式修改"的判断）。
"""

import argparse
from typing import Dict, Optional


def attach_defaults(args, parser: argparse.ArgumentParser) -> None:
    """把 parser 默认值附到 args._defaults（供 tools.config.merge 使用）。"""
    args._defaults = {a.dest: parser.get_default(a.dest)
                      for a in parser._actions if a.dest != 'help'}


def add_dataset_args(parser: argparse.ArgumentParser,
                     default_dataset: str = 'cifar10') -> None:
    parser.add_argument('--dataset', default=default_dataset,
                        help='数据集注册名（datasets/registry）')
    parser.add_argument('--root', required=True,
                        help='数据根目录（数据集加载器解释）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='每 split 最多加载的图数（0=不限制；PPA 必配）')
    parser.add_argument('--device', default='cuda:0')


def add_reader_args(parser: argparse.ArgumentParser,
                    reader_kind_default: str = 'base',
                    temperature_default: float = 1.0) -> None:
    """reader 结构参数（键名与 ReaderConfig.to_dict 一致，config 合并无需映射）。"""
    parser.add_argument('--reader-kind', choices=['base', 'attention', 'evidence'],
                        default=reader_kind_default)
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='旧式统一隐层宽度；与 --hidden-dims 二选一')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='encoder GNN 层数（uniform 形式）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout（uniform 形式）')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=None,
                        help='encoder 逐层宽度列表（覆盖 --hidden-dim/--n-layers）')
    parser.add_argument('--encoder-dropouts', type=float, nargs='+', default=None,
                        help='encoder 逐层 dropout 列表')
    parser.add_argument('--scorer-hidden-dims', type=int, nargs='+', default=None,
                        help='base reader scorer MLP 逐层宽度（默认 2 个隐层）')
    parser.add_argument('--m', type=int, default=4, help='belief token 数（attention）')
    parser.add_argument('--n-heads', type=int, default=4, help='注意力头数（attention）')
    parser.add_argument('--temperature', type=float, default=temperature_default,
                        help='采样温度（RL/评估用；BC 训练恒 1.0）')


def reader_config_from_args(args) -> Dict:
    """CLI reader 参数 → ReaderConfig 兼容 dict（供 active_graph_reader.ReaderConfig）。"""
    cfg: Dict[str, object] = {
        'kind': getattr(args, 'reader_kind', 'base'),
        'edge_feat_dim': getattr(args, 'edge_feat_dim', 7),
        'node_feat_dim': getattr(args, 'node_feat_dim', 0),
        'm': getattr(args, 'm', 4),
        'n_heads': getattr(args, 'n_heads', 4),
        'temperature': getattr(args, 'temperature', 1.0),
    }
    hidden_dims = getattr(args, 'hidden_dims', None)
    dropouts = getattr(args, 'encoder_dropouts', None)
    if hidden_dims:
        cfg['hidden_dims'] = list(hidden_dims)
        cfg['dropouts'] = list(dropouts) if dropouts else \
            [getattr(args, 'dropout', 0.1)] * len(hidden_dims)
    else:
        cfg['hidden_dim'] = getattr(args, 'hidden_dim', 128)
        cfg['n_hidden_dims'] = getattr(args, 'n_layers', 2)
        cfg['dropout'] = getattr(args, 'dropout', 0.1)
    scorer = getattr(args, 'scorer_hidden_dims', None)
    if scorer and cfg['kind'] in ('base', 'evidence'):
        cfg['scorer_hidden_dims'] = list(scorer)
    if cfg['kind'] == 'evidence':
        # 证据头类别数：train 脚本拿到 DatasetSpec 后覆写为 spec.n_classes
        cfg['n_classes'] = getattr(args, 'n_classes', 0)
    return cfg
