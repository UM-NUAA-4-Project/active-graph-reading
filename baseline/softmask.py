"""Soft-Mask GNN 基线（AGR 协议适配版）。

复现自 Soft-Mask 官方实现（WWW 2021 "Soft-Mask: Adaptive Substructure
Extractions for GNNs"，用户本地 soft-mask-gnn-main，PyG 版）SMG 变体：
3×(WeightConv1 掩码生成 → SparseConv 掩码调制卷积)、add-pool 读出、
2 层 MLP 头；超参对齐原版默认（hidden 128 / dropout 0.5 / lr 5e-4 /
wd 1e-4 / seed 777）。

协议适配决策（详见 protocol-adaptation.md §6）：
1. 掩码生成的邻域聚合在完整图上进行（训练）；评测时只在
   "V_revealed ∪ frontier" 可见子图上重算第一层掩码分数（对齐 SAGPool
   适配的重打分口径），取分数最高的 frontier 节点揭示（平分取小 id）；
2. 原版分类管线弃用边特征（TU 无边特征）；本仓 cifar10/ppa 有边特征，
   用小 MLP 将边特征编码为每边标量权重，走 SparseConv 原版自带的
   edge_weight 消息通道（消息 = w_e · h_j）；
3. 聚合排除自环（原版 WeightConv1/SparseConv 均 remove_self_loops），
   用 not_self 边权实现；frontier 节点特征置零（build_visible_graph）；
4. 无节点特征数据集用共享 token（对齐 sagpool.py 约定）。

训练（run_softmask.py）在完整图上标准监督；eval 时 scorer 只见可见
子图——分布漂移是基线设定的一部分（与 SAGPool 相同）。
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

_CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from active_graph_reader.core.base_reader import BaseReader  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402


def _not_self_weight(g: dgl.DGLGraph, feat_weight: torch.Tensor = None) -> torch.Tensor:
    """每边标量消息权重 = 边特征编码 ⊗ 非自环指示（排除自环聚合）。"""
    src, dst = g.edges()
    w = (src != dst).float()
    if feat_weight is not None:
        w = w * feat_weight
    return w


class MaskGen(nn.Module):
    """WeightConv1 的 DGL 移植：邻域聚合 + 自身特征 → 每节点掩码 logit。"""

    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(in_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim * 2, hid_dim)
        self.lin4 = nn.Linear(hid_dim, 1)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        with g.local_scope():
            xs = x if mask is None else x * mask.unsqueeze(-1)
            g.ndata['_h'] = self.lin1(xs)
            g.update_all(fn.copy_u('_h', '_m'), fn.sum('_m', '_agg'))
            agg = g.ndata['_agg']
            w = self.lin4(F.relu(self.lin3(torch.cat([agg, self.lin2(xs)], dim=-1))))
            return w.squeeze(-1)          # pre-sigmoid logit


class MaskedConv(nn.Module):
    """SparseConv 的 DGL 移植：(x⊙m)W 消息（边权调制）+ lin(x) 残差，输出 ⊙m。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, mask: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            g.ndata['_x'] = x
            g.ndata['_h'] = self.weight(x * mask.unsqueeze(-1))
            g.edata['_w'] = edge_weight
            g.update_all(fn.u_mul_e('_h', '_w', '_m'), fn.sum('_m', '_agg'))
            out = g.ndata['_agg'] + self.lin(x)
        return out * mask.unsqueeze(-1)


class SoftMaskModel(nn.Module):
    """层级软掩码（单图前向；自环需已加，ndata/edata['feat'] 按需提供）。"""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, n_classes: int,
                 hidden: int = 128, n_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout

        if node_feat_dim > 0:
            self.node_enc = nn.Linear(node_feat_dim, hidden)
        else:
            self.node_enc = None
            self.node_token = nn.Parameter(torch.zeros(hidden))
        # 边特征 → 每边标量权重（SparseConv edge_weight 通道的实例化）。
        # 末层零初始化：初始 w_e ≡ 0 → 边权 = not_self 指示，行为与原版
        # 无边权 SparseConv 逐位一致，调制能力由训练习得
        if edge_feat_dim > 0:
            self.edge_enc = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))
            nn.init.zeros_(self.edge_enc[-1].weight)
            nn.init.zeros_(self.edge_enc[-1].bias)
        else:
            self.edge_enc = None
        self.convs = nn.ModuleList(MaskedConv(hidden, hidden)
                                   for _ in range(n_layers))
        self.masks = nn.ModuleList(MaskGen(hidden, hidden)
                                   for _ in range(n_layers))
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, n_classes)

    def init_nodes(self, g: dgl.DGLGraph) -> torch.Tensor:
        if self.node_enc is not None:
            return self.node_enc(g.ndata['feat'])
        return self.node_token.unsqueeze(0).expand(g.num_nodes(), -1)

    def edge_weights(self, g: dgl.DGLGraph) -> torch.Tensor:
        feat_w = None
        if self.edge_enc is not None and 'feat' in g.edata:
            feat_w = self.edge_enc(g.edata['feat']).squeeze(-1)
        return _not_self_weight(g, feat_w)

    def first_stage_scores(self, g: dgl.DGLGraph) -> torch.Tensor:
        """协议评测入口：第一层掩码的每节点 logit（sigmoid 单调，排序一致）。"""
        h = self.init_nodes(g)
        return self.masks[0](g, h, mask=None)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """完整 SMG 前向（单图；掩码逐层链式传递 + 掩码调制卷积）。"""
        h = self.init_nodes(g)
        e = self.edge_weights(g)
        mask_val = None
        for i in range(self.n_layers):
            mask_logit = self.masks[i](g, h, mask_val)
            mask_val = torch.sigmoid(mask_logit)
            h = F.relu(self.convs[i](g, h, mask_val, e))
        with g.local_scope():
            g.ndata['_p'] = h
            out = dgl.sum_nodes(g, '_p')          # global add pool（单图）
        x = F.relu(self.lin1(out))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)                        # [1, n_classes]


class SoftmaskStrategy(BaseReader):
    """Soft-Mask 第一层掩码分数 → 贪心 frontier 选点（协议合规评测形态）。"""

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return 'softmask'

    def load(self, ckpt_path: str, device='cpu') -> None:
        """加载 run_softmask.py 训出的 scorer checkpoint（结构以内嵌 config 为准）。"""
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        model = SoftMaskModel(
            node_feat_dim=cfg['node_feat_dim'],
            edge_feat_dim=cfg['edge_feat_dim'],
            n_classes=cfg['n_classes'],
            hidden=cfg['hidden'], n_layers=cfg['n_layers'],
            dropout=cfg['dropout'])
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()

    def reset(self) -> None:
        pass   # 无跨 rollout 状态（模型常驻）

    def select(self, graph_state: GraphState) -> 'int | None':
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._model is None:
            raise RuntimeError("softmask 策略需先 load(checkpoint) "
                               "（eval.py --strategy softmask 自动完成）")
        frontier_nodes = sorted({v for (_u, v) in frontier})
        from baseline.sagpool import build_visible_graph
        g, ids = build_visible_graph(graph_state)
        with torch.no_grad():
            scores = self._model.first_stage_scores(g.to(self._device))
        score_of = {old: float(s) for old, s in zip(ids, scores.tolist())}
        # 分数最高者优先；平分取小 id（确定性）
        return max(frontier_nodes, key=lambda v: (score_of[v], -v))
