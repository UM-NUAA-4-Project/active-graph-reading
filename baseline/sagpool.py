"""SAGPool 基线（AGR 协议适配版）。

复现自 SAGPool 官方实现（ICML 2019 "Self-Attention Graph Pooling"，
用户本地 SAGPool-master，PyG 版）：结构骨架保持一致——
3×(conv → gate → top-k 门控池化) 层级结构、逐层 mean+max 读出求和、
3 层 MLP 头；超参对齐原默认（hidden 128 / pool_ratio 0.5 / dropout 0.5 /
lr 5e-4 / weight-decay 1e-4）。

协议适配决策：
1. conv 用 GINE 式边特征卷积替换 GCNConv——边特征是本协议的前沿可见信息，
   且 PPA/COLLAB 无节点特征，消息传递必须吃边特征；
2. 评分器评测形态：原版 top-k 在完整图上一次性选点，违反 AGR 协议（使用
   了未观测的全局结构，且选点不保证与 seed 连通）。适配版每一步只在
   "V_revealed ∪ frontier" 的可见子图上重算第一级 gate 分数，取分数最高
   的 frontier 节点揭示（平分取小 id，确定性）。
3. 可见子图中 frontier 节点的身份与边特征是协议可见信息（one-ended），
   其节点特征未揭示 → 置零。
4. 训练（run_sagpool.py）在完整图上进行（标准 SAGPool 监督，单图 batch，
   与原实现单图处理一致）；eval 时 scorer 只见可见子图——分布漂移是
   该基线设定的一部分。
"""

import math
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


class EGINConv(nn.Module):
    """GIN 式边特征卷积：h_v' = MLP((1+eps)·h_v + Σ_{u∈N(v)} relu(h_u + e_uv))。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())

    def forward(self, g, h, e):
        with g.local_scope():
            src, _dst = g.edges()
            g.edata['m'] = F.relu(h[src] + e)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'agg'))
            agg = g.ndata['agg']
        return self.mlp((1.0 + self.eps) * h + agg)


class GateConv(nn.Module):
    """SAGPool score_layer（原版 Conv(in,1)）的 GINE 版：输出每节点标量分。"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, g, h, e):
        with g.local_scope():
            src, _dst = g.edges()
            g.edata['m'] = F.relu(h[src] + e)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'agg'))
            agg = g.ndata['agg']
        return self.lin((1.0 + self.eps) * h + agg).squeeze(-1)


class SAGPoolModel(nn.Module):
    """层级 SAGPool（单图前向；自环需已加，ndata/edata['feat'] 按需提供）。"""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, n_classes: int,
                 hidden: int = 128, n_layers: int = 3, pool_ratio: float = 0.5,
                 dropout: float = 0.5):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.pool_ratio = pool_ratio
        self.dropout = dropout

        if node_feat_dim > 0:
            self.node_enc = nn.Linear(node_feat_dim, hidden)
        else:
            self.node_enc = None
            self.node_token = nn.Parameter(torch.zeros(hidden))
        self.edge_proj = (nn.Linear(edge_feat_dim, hidden)
                          if edge_feat_dim > 0 else None)
        self.convs = nn.ModuleList(EGINConv(hidden, hidden)
                                   for _ in range(n_layers))
        self.gates = nn.ModuleList(GateConv(hidden) for _ in range(n_layers))
        self.lin1 = nn.Linear(2 * hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden // 2)
        self.lin3 = nn.Linear(hidden // 2, n_classes)

    def init_nodes(self, g):
        if self.node_enc is not None:
            return self.node_enc(g.ndata['feat'])
        return self.node_token.unsqueeze(0).expand(g.num_nodes(), -1)

    def init_edges(self, g):
        if self.edge_proj is not None:
            return F.relu(self.edge_proj(g.edata['feat']))
        return g.edata['feat'].new_zeros(g.num_edges(), self.hidden)

    def first_stage_scores(self, g):
        """协议评测入口：第一级 gate 的每节点分数（对齐原版 stage-1 score_layer）。"""
        h = self.init_nodes(g)
        e = self.init_edges(g)
        h = F.relu(self.convs[0](g, h, e))
        return self.gates[0](g, h, e)

    def forward(self, g):
        """完整 SAGPool 前向（单图；层级 top-k 池化 + 逐层读出求和）。"""
        h = self.init_nodes(g)
        e = self.init_edges(g)
        out = 0
        for i, (conv, gate) in enumerate(zip(self.convs, self.gates)):
            h = F.relu(conv(g, h, e))
            s = gate(g, h, e)
            k = max(1, math.ceil(self.pool_ratio * g.num_nodes()))
            idx = torch.topk(s, k, largest=True).indices
            h = h[idx] * torch.tanh(s[idx]).unsqueeze(-1)
            out = out + torch.cat([h.max(dim=0).values, h.mean(dim=0)])
            if i < self.n_layers - 1:
                g = dgl.node_subgraph(g, idx)
                e = e[g.edata[dgl.EID]]
        x = F.relu(self.lin1(out))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        return self.lin3(x).unsqueeze(0)   # [1, n_classes]（单图 batch）


def build_visible_graph(state: GraphState):
    """可见子图：V_revealed ∪ frontier_nodes + 已揭示边（双向）+ frontier 边（双向）。

    - frontier 节点的身份与边特征是协议可见信息（one-ended）；其节点特征
      未揭示 → 置零；
    - 自环归一（remove+add，与训练图约定一致）。

    Returns
    -------
    g : dgl.DGLGraph（ndata['feat']/edata['feat'] 按原图是否提供）
    ids : List[int]  新图节点 i → 原图节点 id（revealed 升序在前，frontier 升序在后）
    """
    revealed = sorted(state.V_revealed)
    frontier_nodes = sorted({v for (_u, v) in state.frontier_edges})
    ids = revealed + frontier_nodes
    n = len(ids)
    old_to_new = {old: i for i, old in enumerate(ids)}

    src, dst, ekeys = [], [], []
    for u, v in list(state.revealed_edge_keys) + list(state.frontier_edges):
        key_u, key_v = min(u, v), max(u, v)
        nu, nv = old_to_new[key_u], old_to_new[key_v]
        src += [nu, nv]
        dst += [nv, nu]
        ekeys += [(key_u, key_v), (key_u, key_v)]

    g = dgl.graph((torch.tensor(src, dtype=torch.long),
                   torch.tensor(dst, dtype=torch.long)), num_nodes=n)
    if state.has_edge_feat:
        eids = [state._edge_key_to_id[k] for k in ekeys]
        g.edata['feat'] = state.edge_feat[eids]
    if state.has_node_feat:
        x = torch.zeros(n, state.node_feat_dim)
        x[:len(revealed)] = state.node_feat[revealed]
        g.ndata['feat'] = x
    g = dgl.add_self_loop(dgl.remove_self_loop(g))
    return g, ids


class SagpoolStrategy(BaseReader):
    """SAGPool 第一级 gate 分数 → 贪心 frontier 选点（协议合规评测形态）。"""

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return 'sagpool'

    def load(self, ckpt_path: str, device='cpu') -> None:
        """加载 run_sagpool.py 训出的 scorer checkpoint（结构以内嵌 config 为准）。"""
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        model = SAGPoolModel(
            node_feat_dim=cfg['node_feat_dim'],
            edge_feat_dim=cfg['edge_feat_dim'],
            n_classes=cfg['n_classes'],
            hidden=cfg['hidden'], n_layers=cfg['n_layers'],
            pool_ratio=cfg['pool_ratio'], dropout=cfg['dropout'])
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
            raise RuntimeError("sagpool 策略需先 load(checkpoint) "
                               "（eval.py --strategy sagpool 自动完成）")
        frontier_nodes = sorted({v for (_u, v) in frontier})
        g, ids = build_visible_graph(graph_state)
        with torch.no_grad():
            scores = self._model.first_stage_scores(g.to(self._device))
        score_of = {old: float(s) for old, s in zip(ids, scores.tolist())}
        # 分数最高者优先；平分取小 id（确定性）
        return max(frontier_nodes, key=lambda v: (score_of[v], -v))

    @torch.no_grad()
    def own_predict_batch(self, subgraphs, seed_rows, steps):
        """自带头终判：SAGPool 分类头对最终揭示子图逐图前向（单图 BN 约定）。

        供 evaluate_fixed 的 acc_own_*（judge='clf'）或 judge='own' 主判；
        seed_rows/steps 为统一批量接口的占位参数（本头不依赖）。
        """
        outs = []
        for g in subgraphs:
            g = dgl.add_self_loop(dgl.remove_self_loop(g))
            if self._model.edge_feat_dim > 0 and 'feat' not in g.edata:
                g.edata['feat'] = torch.zeros(g.num_edges(),
                                              self._model.edge_feat_dim)
            if self._model.node_feat_dim > 0 and 'feat' not in g.ndata:
                g.ndata['feat'] = torch.zeros(g.num_nodes(),
                                              self._model.node_feat_dim)
            outs.append(self._model(g.to(self._device)))
        return torch.cat(outs, dim=0)
