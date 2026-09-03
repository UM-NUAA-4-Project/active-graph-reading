"""ReaderConfig：reader 结构配置（逐层可调）+ 工厂 + checkpoint 推断。

解决旧脚本的痛点：
- 模型参数只有 hidden_dim/n_layers/dropout 粗粒度标量 → 现在支持逐层列表
- 三代脚本各自手写 build_reader → 收敛为 build_reader(config)
- checkpoint config 键名混乱（n_layers vs n_layers_reader 等）→ from_dict 统一兼容

序列化约定与 classifier.ClassifierConfig 一致：层间全同时折叠为标量。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


def _fold(d: Dict, values: List, base: str) -> None:
    """全同 → {base: v, n_{base}s: len}；否则 → {base+'s': list}。"""
    if len(set(values)) == 1:
        d[base] = values[0]
        d['n_' + base + 's'] = len(values)
    else:
        d[base + 's'] = list(values)


def _unfold(d: Dict, base: str, default: List) -> List:
    """from_dict 的反向：接受 {base, n_bases} / {base+'s'} / 缺省；None 视同缺省
    （实验 config 会把未提供的 nargs='+' 参数序列化为 null）。"""
    vals = d.get(base + 's')
    if vals is not None:
        return list(vals)
    scalar = d.get(base)
    if scalar is not None:
        n = _to_int(d.get('n_' + base + 's'), len(default))
        return [scalar] * n
    return list(default)


def _to_int(v, default):
    return default if v is None else int(v)


def _to_float(v, default):
    return default if v is None else float(v)


@dataclass
class ReaderConfig:
    """reader 结构配置。

    encoder_*  ：FrontierNodesEncoder 的 GIN 层（逐层 hidden/dropout）
    scorer_*   ：base/evidence reader 的 ReaderMLP（逐层 hidden/dropout）
                 attention reader 要求 encoder 各层同宽（= d_model），
                 scorer_* 忽略
    m/n_heads  ：attention 专用
    n_classes  ：evidence 专用（证据头类别数，须 >0）
    """

    kind: str = 'base'                    # 'base' | 'attention' | 'evidence'
    edge_feat_dim: int = 7
    node_feat_dim: int = 0
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    encoder_dropouts: List[float] = field(default_factory=lambda: [0.1, 0.1])
    # 旧 ReaderMLP = Linear(in,h)→ReLU→Drop→Linear(h,h)→ReLU→Drop→Linear(h,1)
    # 即 2 个隐层条目 + 1 个输出层
    scorer_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    scorer_dropouts: List[float] = field(default_factory=lambda: [0.1, 0.1])
    m: int = 4
    n_heads: int = 4
    stochastic: bool = False
    temperature: float = 1.0
    n_classes: int = 0

    # ---- 便捷属性 ----
    @property
    def hidden_dim(self) -> int:
        return self.encoder_hidden_dims[-1]

    @property
    def n_layers(self) -> int:
        return len(self.encoder_hidden_dims)

    @property
    def dropout(self) -> float:
        return self.encoder_dropouts[0]

    def validate(self) -> None:
        if self.kind not in ('base', 'attention', 'evidence'):
            raise ValueError(f"kind 必须是 base/attention/evidence，得到 '{self.kind}'")
        if len(self.encoder_dropouts) != len(self.encoder_hidden_dims):
            raise ValueError("encoder_dropouts 与 encoder_hidden_dims 长度不一致")
        if len(self.scorer_dropouts) != len(self.scorer_hidden_dims):
            raise ValueError("scorer_dropouts 与 scorer_hidden_dims 长度不一致")
        if self.kind == 'attention' and len(set(self.encoder_hidden_dims)) > 1:
            raise ValueError("attention reader 要求 encoder 各层同宽（attention 模块"
                             "与 belief 的 d_model 需要统一）")
        if self.kind == 'evidence' and self.n_classes <= 0:
            raise ValueError("evidence reader 需要 n_classes > 0（证据头类别数）")

    # ---- 序列化 ----
    def to_dict(self) -> Dict:
        self.validate()
        d: Dict = {'kind': self.kind,
                   'edge_feat_dim': self.edge_feat_dim,
                   'node_feat_dim': self.node_feat_dim,
                   'm': self.m, 'n_heads': self.n_heads,
                   'stochastic': self.stochastic,
                   'temperature': self.temperature}
        _fold(d, self.encoder_hidden_dims, 'hidden_dim')
        _fold(d, self.encoder_dropouts, 'dropout')
        if self.kind in ('base', 'evidence'):
            _fold(d, self.scorer_hidden_dims, 'scorer_hidden_dim')
            _fold(d, self.scorer_dropouts, 'scorer_dropout')
        if self.kind == 'evidence':
            d['n_classes'] = self.n_classes
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'ReaderConfig':
        kind = d.get('kind', d.get('reader_kind', 'base'))
        encoder_hidden = _unfold(d, 'hidden_dim', [128, 128])
        encoder_drop = _unfold(d, 'dropout', [0.1] * len(encoder_hidden))
        if len(encoder_drop) != len(encoder_hidden):
            encoder_drop = (encoder_drop * len(encoder_hidden))[:len(encoder_hidden)]
        cfg = ReaderConfig(
            kind=kind,
            edge_feat_dim=_to_int(d.get('edge_feat_dim'), 7),
            node_feat_dim=_to_int(d.get('node_feat_dim'), 0),
            encoder_hidden_dims=[int(x) for x in encoder_hidden],
            encoder_dropouts=[float(x) for x in encoder_drop],
            scorer_hidden_dims=[int(x) for x in
                                _unfold(d, 'scorer_hidden_dim',
                                        [encoder_hidden[-1]] * 2)],
            scorer_dropouts=[float(x) for x in
                             _unfold(d, 'scorer_dropout',
                                     encoder_drop[:1] * 2)],
            m=_to_int(d.get('m'), 4),
            n_heads=_to_int(d.get('n_heads'), 4),
            stochastic=bool(d.get('stochastic', False)),
            temperature=_to_float(d.get('temperature'), 1.0),
            n_classes=_to_int(d.get('n_classes'), 0),
        )
        cfg.validate()
        return cfg

    @staticmethod
    def from_legacy(kind: str = 'base', edge_feat_dim: int = 7,
                    node_feat_dim: int = 0, hidden_dim: int = 128,
                    n_layers: int = 2, dropout: float = 0.1,
                    m: int = 4, n_heads: int = 4,
                    stochastic: bool = False, temperature: float = 1.0,
                    n_classes: int = 0,
                    ) -> 'ReaderConfig':
        """旧式标量构造（与旧 LearnableReader/AttentionReader 签名对齐）。"""
        cfg = ReaderConfig(
            kind=kind, edge_feat_dim=edge_feat_dim, node_feat_dim=node_feat_dim,
            encoder_hidden_dims=[hidden_dim] * n_layers,
            encoder_dropouts=[dropout] * n_layers,
            scorer_hidden_dims=[hidden_dim] * 2,
            scorer_dropouts=[dropout] * 2,
            m=m, n_heads=n_heads, stochastic=stochastic, temperature=temperature,
            n_classes=n_classes,
        )
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# 从 state_dict 推断（老 checkpoint 无内嵌 config 时）
# ---------------------------------------------------------------------------

_CONV_RE = re.compile(r'^frontier_encoder\.convs\.(\d+)\.mlp\.0\.weight$')
_SCORER_HIDDEN_RE = re.compile(r'^reader_mlp\.mlp\.(\d+)\.weight$')


def infer_reader_config(state_dict: Dict[str, torch.Tensor]) -> ReaderConfig:
    """从权重 shape 推断 reader 结构（结构事实以权重为准）。

    注意：与 m5/training_utils.infer_reader_config 同名不同职——本函数返回
    完整 ReaderConfig，供 build_reader 直接重建模型。
    """
    conv_idx = {}
    for key, tensor in state_dict.items():
        m = _CONV_RE.match(key)
        if m:
            conv_idx[int(m.group(1))] = int(tensor.shape[0] // 2)  # bottleneck=2h
    if not conv_idx:
        raise ValueError("state_dict 中未找到 frontier_encoder.convs.*，无法推断")
    encoder_hidden = [conv_idx[i] for i in sorted(conv_idx)]

    if 'frontier_encoder.node_feat_proj.weight' in state_dict:
        node_feat_dim = int(state_dict['frontier_encoder.node_feat_proj.weight'].shape[1])
    else:
        node_feat_dim = 0

    edge_feat_dim = int(
        state_dict['frontier_encoder.convs.0.edge_encoder.weight'].shape[1])

    kind = 'base'
    if any('belief_state' in k for k in state_dict):
        kind = 'attention'
    elif any(k.startswith('evidence_lstm') for k in state_dict):
        kind = 'evidence'

    m = 4
    if kind == 'attention' and 'belief_state.tokens' in state_dict:
        m = int(state_dict['belief_state.tokens'].shape[0])

    # evidence 证据头类别数：Sequential 里最后一个 Linear 的输出宽度
    n_classes = 0
    if kind == 'evidence':
        head_layers = [(int(k.split('.')[1]), int(t.shape[0]))
                       for k, t in state_dict.items()
                       if k.startswith('evidence_head.') and
                       k.endswith('.weight')]
        if head_layers:
            n_classes = max(head_layers)[1]

    scorer_hidden = [128, 128]
    if kind in ('base', 'evidence'):
        linears = []                             # (block_idx, out_dim)
        for key, tensor in state_dict.items():
            m = _SCORER_HIDDEN_RE.match(key)
            if m and int(m.group(1)) % 3 == 0:   # Linear 位于 block 起始 3i
                linears.append((int(m.group(1)) // 3, int(tensor.shape[0])))
        if linears:
            linears.sort()
            # 最后一个 Linear 是输出层（out=1），其余为隐层
            scorer_hidden = [h for _, h in linears[:-1]]

    cfg = ReaderConfig(
        kind=kind,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        encoder_hidden_dims=encoder_hidden,
        encoder_dropouts=[0.1] * len(encoder_hidden),
        scorer_hidden_dims=scorer_hidden,
        scorer_dropouts=[0.1] * len(scorer_hidden),
        m=m,
        n_classes=n_classes,
    )
    return cfg


def reader_config_from_checkpoint(ckpt: dict,
                                  state_dict: Optional[dict] = None) -> ReaderConfig:
    """checkpoint → ReaderConfig：结构取自权重 shape，非结构超参（m/n_heads/
    temperature/stochastic）取内嵌 config（如有）。"""
    sd = state_dict if state_dict is not None else ckpt.get('reader_state_dict', {})
    cfg = infer_reader_config(sd)
    embedded = ckpt.get('config')
    if isinstance(embedded, dict):
        embedded = dict(embedded)
        # None = CLI 未提供（build_experiment_config 原样序列化），不得覆盖推断值
        embedded = {k: v for k, v in embedded.items() if v is not None}
        # 结构字段以权重为准，防止内嵌 config 漂移（fix_ckpt_config 的教训）；
        # 标量折叠与逐层列表两种形式都要剔掉
        for k in ('hidden_dim', 'hidden_dims', 'n_hidden_dims',
                  'n_layers', 'n_layers_reader',
                  'n_dropouts', 'n_scorer_hidden_dims', 'n_scorer_dropouts',
                  'edge_feat_dim', 'node_feat_dim'):
            embedded.pop(k, None)
        merged = cfg.to_dict()
        merged.update(embedded)
        cfg = ReaderConfig.from_dict(merged)
    return cfg


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def build_reader(config: ReaderConfig):
    """ReaderConfig → reader 实例（统一替代各脚本手写的 build_*_reader）。"""
    config.validate()
    if config.kind == 'attention':
        from .reader.attention.attention_reader import AttentionReader
        return AttentionReader(config=config)
    if config.kind == 'evidence':
        from .reader.evidence import EvidenceReader
        return EvidenceReader(config=config)
    from .reader.learnable_reader import LearnableReader
    return LearnableReader(config=config)
