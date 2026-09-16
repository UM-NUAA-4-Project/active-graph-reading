"""AgentNet 基线（AGR 协议适配版，单 agent）。

复现自 AgentNet 官方实现（ICLR 2023 "Agent-based Graph Neural Networks"，
arXiv 2206.11010，用户本地 AgentNet-main，PyG 版）的单 agent 移植：
移动头 = Q·K 注意力（[邻居 ‖ 当前 ‖ 边] 编码）+ BSEU 探索偏置
（back/stay/explored/unexplored 可学习标量，visited 衰减 0.9），
gumbel-softmax(hard, straight-through) 选边；节点/agent 残差更新 +
逐步读出累加平均 → 分类。超参对齐原版 README（hidden 128 / steps 16 /
lr 1e-4 / wd 0.01 / clip 1.0 / visited-decay 0.9 / gumbel τ=2/3）。

协议适配决策（详见 protocol-adaptation.md §7）：
1. num_agents=1（原版 K=18~26）：训练/评测同构，预算语义逐字节对齐
   （每步揭示 1 个节点）；
2. 去掉 time conditioning：原版 time embedding 表按固定 num_steps 建，
   评测预算步数可变会越界；单 agent 下时间条件收益有限，裁剪；
3. 训练（run_agentnet.py）在完整图上端到端分类监督（原则 2）；评测时
   agent 的移动候选限缩为**当前位置的 frontier 邻居**（深度型，与
   random-walk 候选结构等价），frontier 节点特征置零（P4），one-ended
   边特征进移动头 K（模型 edge_feat 通道现成）；
4. 评测死角处理：rollout 契约下前沿空即结束（rollout_strategy 不瞬移），
   agent 在**已揭示区内用 BSEU 打分导航**（不消耗预算，步数上限 4n）
   到有 frontier 候选的位置再继续；导航耗尽返回 None；
5. 评测确定性：argmax + 平分取小 id（无 gumbel 噪声）。
"""

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import dgl

_CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from active_graph_reader.core.base_reader import BaseReader  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402


def symmetrize_with_feat(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """无向化并保留 ndata/edata['feat']（dgl.to_bidirected 会丢边特征）。

    按 (min,max) 去重后每条无向边补齐双向（平行边/单向边输入均归一）；
    边特征复制到两个方向（cifar10/ppa 边特征本身对称）。
    """
    src, dst = g.edges()
    seen = set()
    new_src, new_dst, feats = [], [], []
    has_efeat = 'feat' in g.edata
    for e in range(g.num_edges()):
        u, v = int(src[e]), int(dst[e])
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        new_src += [u, v]
        new_dst += [v, u]
        if has_efeat:
            feats.append(g.edata['feat'][e])
    g2 = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)),
                   num_nodes=g.num_nodes())
    if has_efeat:
        g2.edata['feat'] = torch.stack(feats).repeat_interleave(2, dim=0)
    if 'feat' in g.ndata:
        g2.ndata['feat'] = g.ndata['feat']
    return g2


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    """原版 conv_mlp/node_mlp/agent_mlp 形态：pre-LN → Linear → act → Linear。"""
    layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.LeakyReLU(0.01)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AgentNetModel(nn.Module):
    """单 agent AgentNet（单图前向；自环需已加——自环即 stay 动作）。"""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, n_classes: int,
                 hidden: int = 128, num_steps: int = 16, dropout: float = 0.0,
                 visited_decay: float = 0.9, input_mlp: bool = True,
                 gumbel_tau: float = 2.0 / 3.0):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.hidden = hidden
        self.num_steps = num_steps
        self.dropout = dropout
        self.visited_decay = visited_decay
        self.gumbel_tau = gumbel_tau
        self.with_edge = edge_feat_dim > 0

        act = nn.LeakyReLU(0.01)
        if node_feat_dim > 0:
            if input_mlp:
                self.input_proj = nn.Sequential(
                    nn.Linear(node_feat_dim, hidden * 2), act,
                    nn.Linear(hidden * 2, hidden))
            else:
                self.input_proj = nn.Linear(node_feat_dim, hidden)
        else:
            self.input_proj = None
            self.node_token = nn.Parameter(torch.zeros(hidden))
        if self.with_edge:
            self.edge_input_proj = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden * 2), act,
                nn.Linear(hidden * 2, hidden))

        self.agent_emb = nn.Embedding(1, hidden)
        # 移动头：Q·K 注意力（key 输入 = [邻居 ‖ (边) ‖ 当前]）
        key_in = hidden * 3 if self.with_edge else hidden * 2
        self.key = nn.Sequential(nn.LayerNorm(key_in), nn.Linear(key_in, hidden))
        self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        # BSEU 偏置（bias_attention 初始化：向探索倾斜）
        self.back_param = nn.Parameter(torch.tensor(0.0))
        self.stay_param = nn.Parameter(torch.tensor(-1.0))
        self.explored_param = nn.Parameter(torch.tensor(0.0))
        self.unexplored_param = nn.Parameter(torch.tensor(5.0))
        # 更新 MLP（pre-LN，对齐原版 post_ln=False）
        ed = hidden if self.with_edge else 0
        self.agent_node_lin = (nn.Sequential(
            nn.LayerNorm(hidden + ed), nn.Linear(hidden + ed, hidden), act)
            if self.with_edge else nn.Identity())
        self.node_mlp = _mlp(hidden * 2, hidden * 2, hidden, dropout)
        self.message_val = (nn.Sequential(
            nn.LayerNorm(hidden + ed), nn.Linear(hidden + ed, hidden), act)
            if self.with_edge else None)
        self.conv_mlp = _mlp(hidden * 2, hidden * 2, hidden, dropout)
        self.agent_mlp = _mlp(hidden * 2 + ed, hidden * 2, hidden, dropout)
        self.step_readout = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2),
            nn.Dropout(dropout))
        self.fc = nn.Linear(hidden * 2, n_classes)

    # ------------------------------------------------------------------
    # 表征初始化
    # ------------------------------------------------------------------
    def init_hidden(self, g: dgl.DGLGraph) -> torch.Tensor:
        if self.input_proj is not None:
            return self.input_proj(g.ndata['feat'])
        return self.node_token.unsqueeze(0).expand(g.num_nodes(), -1)

    def init_edge_emb(self, g: dgl.DGLGraph):
        if not self.with_edge:
            return None
        return self.edge_input_proj(g.edata['feat'])

    def new_agent(self, device) -> torch.Tensor:
        return self.agent_emb(torch.zeros(1, dtype=torch.long, device=device))[0]

    # ------------------------------------------------------------------
    # 移动头（训练 forward 与策略评测共享）
    # ------------------------------------------------------------------
    def move_logits(self, h: torch.Tensor, agent: torch.Tensor, cur: int,
                    cand: torch.Tensor, cand_edge: torch.Tensor = None,
                    cand_visit: torch.Tensor = None,
                    cand_is_prev: torch.Tensor = None) -> torch.Tensor:
        """候选移动打分：注意力 + BSEU 偏置。cand 含 cur（自环 stay）时
        stay_param 生效；cand_is_prev 标记上一步位置（back_param）。"""
        q = self.query(agent)
        cur_h = h[cur].unsqueeze(0).expand(len(cand), -1)
        k_in = torch.cat([h[cand], cur_h], dim=-1) if not self.with_edge \
            else torch.cat([h[cand], cand_edge, cur_h], dim=-1)
        k = self.key(k_in)
        attn = (q.unsqueeze(0) * k).sum(-1) / math.sqrt(self.hidden)
        bias = torch.zeros_like(attn)
        if cand_visit is not None:
            is_cur = cand == cur
            mask_old = ~is_cur
            s = cand_visit[mask_old]
            bias[mask_old] += (s / self.visited_decay) * self.explored_param \
                + (1.0 - s / self.visited_decay) * self.unexplored_param
            bias[is_cur] += self.stay_param
            if cand_is_prev is not None:
                bias[cand_is_prev] += self.back_param
        return attn + bias

    @torch.no_grad()
    def move_choice_eval(self, logits: torch.Tensor) -> int:
        """评测：argmax（确定性，无 gumbel 噪声）。"""
        return int(logits.argmax().item())

    def move_choice_train(self, logits: torch.Tensor):
        """gumbel-softmax(hard, straight-through)：返回 (idx, attn_val)。"""
        gumbels = -torch.log(torch.rand_like(logits).clamp_min(1e-10))
        y_soft = torch.softmax((logits + gumbels) / self.gumbel_tau, dim=-1)
        idx = int(y_soft.argmax().item())
        onehot = torch.zeros_like(y_soft)
        onehot[idx] = 1.0
        ret = onehot - y_soft.detach() + y_soft      # straight-through
        return idx, ret[idx]

    # ------------------------------------------------------------------
    # 步进更新：agent→node 消息 + 邻域卷积（边感知消息）+ agent 更新
    # ------------------------------------------------------------------
    def step_update(self, h: torch.Tensor, agent: torch.Tensor, pos: int,
                    conv_nbr: torch.Tensor, conv_edge: torch.Tensor = None,
                    moved_edge: torch.Tensor = None,
                    attn_val=1.0):
        """返回更新后的 (h, agent)；functional 更新（index_copy，无 in-place），
        跨步 autograd 安全。"""
        if self.with_edge:
            me = moved_edge * attn_val
            a_msg = self.agent_node_lin(torch.cat([agent, me], dim=-1))
        else:
            a_msg = agent
        new_h_pos = h[pos] + self.node_mlp(torch.cat([h[pos], a_msg]))
        if conv_nbr.numel() > 0:
            if self.with_edge:
                agg = self.message_val(
                    torch.cat([h[conv_nbr], conv_edge], dim=-1)).sum(dim=0)
            else:
                agg = h[conv_nbr].sum(dim=0)
        else:
            agg = torch.zeros(self.hidden, device=h.device)
        new_h_pos = new_h_pos + self.conv_mlp(torch.cat([new_h_pos, agg]))
        h = h.index_copy(0, torch.tensor([pos], dtype=torch.long,
                                          device=h.device),
                         new_h_pos.unsqueeze(0))
        if self.with_edge:
            agent_cat = torch.cat([agent, new_h_pos * attn_val, me])
        else:
            agent_cat = torch.cat([agent, new_h_pos * attn_val])
        return h, agent + self.agent_mlp(agent_cat)

    # ------------------------------------------------------------------
    # 训练前向（完整图；start=None 时随机起点，random 已播种）
    # ------------------------------------------------------------------
    def forward(self, g: dgl.DGLGraph, start: int = None) -> torch.Tensor:
        h = self.init_hidden(g)
        edge_emb = self.init_edge_emb(g)
        n = g.num_nodes()
        device = h.device

        src, dst = g.edges()
        edge_map = {}
        nbrs = {}
        for e in range(g.num_edges()):
            u, v = int(src[e]), int(dst[e])
            edge_map.setdefault((u, v), e)
            nbrs.setdefault(u, set()).add(v)
            nbrs.setdefault(v, set()).add(u)

        agent = self.new_agent(device)
        cur = random.randrange(n) if start is None else int(start)
        visit = torch.zeros(n, device=device)
        visit[cur] = 1.0
        prev = None
        attn_val = 1.0
        moved_edge = torch.zeros(self.hidden, device=device) \
            if self.with_edge else None
        out = None

        for i in range(self.num_steps + 1):
            if i > 0:
                cand = sorted(nbrs.get(cur, set()))      # 含自环 stay
                cand_t = torch.tensor(cand, dtype=torch.long, device=device)
                cand_e = None
                if self.with_edge:
                    cand_e = torch.stack([
                        edge_emb[edge_map[(cur, u)]] for u in cand])
                is_prev = (cand_t == prev) if prev is not None else None
                logits = self.move_logits(
                    h, agent, cur, cand_t, cand_e, visit[cand_t], is_prev)
                idx, attn_val = self.move_choice_train(logits)
                new_cur = cand[idx]
                if self.with_edge:
                    moved_edge = edge_emb[edge_map[(cur, new_cur)]]
                prev, cur = cur, new_cur
                visit = visit * self.visited_decay
                visit[cur] = 1.0
            conv_nbr = sorted(nbrs.get(cur, set()) - {cur})
            conv_nbr_t = torch.tensor(conv_nbr, dtype=torch.long,
                                      device=device)
            conv_e = None
            if self.with_edge:
                conv_e = torch.stack([
                    edge_emb[edge_map[(cur, u)]] for u in conv_nbr]) \
                    if conv_nbr else torch.zeros(0, self.hidden, device=device)
            h, agent = self.step_update(h, agent, cur, conv_nbr_t, conv_e,
                                        moved_edge, attn_val)
            layer_out = self.fc(self.step_readout(agent))
            out = layer_out if out is None else out + layer_out
        return (out / (self.num_steps + 1)).unsqueeze(0)   # [1, n_classes]


class AgentnetStrategy(BaseReader):
    """单 agent 游走 → 逐点揭示（协议合规评测形态；有状态，reset 重建）。"""

    NAV_CAP_FACTOR = 4   # 已揭示区内导航步数上限 = 4 × n_nodes（对齐 random-walk）

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None
        self._reset_walk()

    def _reset_walk(self):
        self._pos = None          # agent 当前位置（原图 id）
        self._prev = None
        self._visit = {}          # orig_id → 衰减访问强度
        self._h_cache = {}        # orig_id → 更新后的节点嵌入
        self._agent = None
        self._nav_steps = 0

    @property
    def name(self) -> str:
        return 'agentnet'

    def load(self, ckpt_path: str, device='cpu') -> None:
        """加载 run_agentnet.py 训出的 checkpoint（结构以内嵌 config 为准）。"""
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        model = AgentNetModel(
            node_feat_dim=cfg['node_feat_dim'],
            edge_feat_dim=cfg['edge_feat_dim'],
            n_classes=cfg['n_classes'],
            hidden=cfg['hidden'], num_steps=cfg['num_steps'],
            dropout=cfg['dropout'], visited_decay=cfg['visited_decay'])
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()

    def reset(self) -> None:
        self._reset_walk()

    def own_predict(self, sub_g):
        """自身读出头对揭示子图的分类（评测双口径记录，对齐 evidence 的
        acc_* / acc_clf_* 惯例：协议终判仍为冻结分类器，本头为参照）。

        输入 SubgraphBuilder 产出的揭示子图（已双向+自环+特征）；按训练
        约定重整构图后从头走一遍 agent 游走读出 → [1, n_classes] logits。
        """
        g = dgl.add_self_loop(symmetrize_with_feat(dgl.remove_self_loop(sub_g)))
        return self._model(g.to(self._device))

    def on_new_frontier(self, edges) -> None:
        """rollout 开始时经 initial_frontier 推断 seed，初始化 walk 状态。"""
        if self._pos is None and edges:
            self._pos = edges[0][0]
            self._visit[self._pos] = 1.0
            self._agent = self._model.new_agent(self._device)

    def _edge_feat_of(self, state: GraphState, u: int, v: int):
        """(u,v) 边的特征（原图 id；frontier 边 one-ended 可见）。"""
        if not state.has_edge_feat:
            return None
        key = (min(u, v), max(u, v))
        return state.edge_feat[state._edge_key_to_id[key]]

    def _cand_from_frontier(self, state: GraphState):
        return sorted({v for (u, v) in state.frontier_edges if u == self._pos})

    def _visible_adjacency(self, state: GraphState, node: int):
        """node 在原图中的无向邻居（去自身）。"""
        nbrs = (set(state.g.successors(node).tolist())
                | set(state.g.predecessors(node).tolist()))
        nbrs.discard(node)
        return nbrs

    def select(self, graph_state: GraphState) -> 'int | None':
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._model is None:
            raise RuntimeError("agentnet 策略需先 load(checkpoint) "
                               "（eval.py --strategy agentnet 自动完成）")
        if self._pos is None:
            self.on_new_frontier(frontier)

        from baseline.sagpool import build_visible_graph
        g, ids = build_visible_graph(graph_state)
        g = g.to(self._device)
        h = self._model.init_hidden(g)      # frontier 特征已置零（P4）
        edge_emb = self._model.init_edge_emb(g)
        id_to_row = {old: i for i, old in enumerate(ids)}
        # 游走记忆：已访问节点用更新后的嵌入覆盖初始嵌入
        for old, hv in self._h_cache.items():
            h[id_to_row[old]] = hv
        src, dst = g.edges()
        edge_map = {}
        for e in range(g.num_edges()):
            edge_map.setdefault((int(src[e]), int(dst[e])), e)

        cand = self._cand_from_frontier(graph_state)
        # 可见图邻接（agent 只知道可见边：已揭示边 + frontier one-ended 边；
        # frontier–frontier 的原图边不可见，不得进入卷积/移动）
        g_nbrs = {}
        gs, gd = g.edges()
        for e in range(g.num_edges()):
            u, v = int(gs[e]), int(gd[e])
            g_nbrs.setdefault(u, set()).add(v)
            g_nbrs.setdefault(v, set()).add(u)
        with torch.no_grad():
            # ---- 死角：在已揭示区内用 BSEU 打分导航（不消耗预算）----
            nav_cap = self.NAV_CAP_FACTOR * graph_state.n_nodes_total
            while not cand and self._nav_steps < nav_cap:
                self._nav_steps += 1
                nbrs = self._visible_adjacency(graph_state, self._pos) \
                    & set(graph_state.V_revealed)
                if not nbrs:
                    return None
                nbrs = sorted(nbrs)
                nbr_t = torch.tensor([id_to_row[v] for v in nbrs],
                                     dtype=torch.long, device=self._device)
                nbr_e = None
                if self._model.with_edge:
                    nbr_e = torch.stack([
                        edge_emb[edge_map[(id_to_row[self._pos],
                                           id_to_row[v])]]
                        for v in nbrs])
                visit = torch.tensor([self._visit.get(v, 0.0) for v in nbrs],
                                     device=self._device)
                is_prev = torch.tensor([v == self._prev for v in nbrs],
                                       dtype=torch.bool, device=self._device) \
                    if self._prev is not None else None
                logits = self._model.move_logits(
                    h, self._agent, id_to_row[self._pos], nbr_t, nbr_e,
                    visit, is_prev)
                nxt = nbrs[self._model.move_choice_eval(logits)]
                h = self._apply_move(graph_state, g, ids, h, edge_emb, edge_map,
                                     id_to_row, nxt, g_nbrs)
                cand = self._cand_from_frontier(graph_state)
            if not cand:
                return None

            # ---- 揭示步：frontier 候选打分 → argmax（平分取小 id）----
            cand_t = torch.tensor([id_to_row[v] for v in cand],
                                  dtype=torch.long, device=self._device)
            cand_e = None
            if self._model.with_edge:
                cand_e = torch.stack([
                    edge_emb[edge_map[(id_to_row[self._pos],
                                       id_to_row[v])]]
                    for v in cand])
            visit = torch.tensor([self._visit.get(v, 0.0) for v in cand],
                                 device=self._device)
            is_prev = torch.tensor([v == self._prev for v in cand],
                                   dtype=torch.bool, device=self._device) \
                if self._prev is not None else None
            logits = self._model.move_logits(
                h, self._agent, id_to_row[self._pos], cand_t, cand_e,
                visit, is_prev)
            chosen = cand[self._model.move_choice_eval(logits)]
            h = self._apply_move(graph_state, g, ids, h, edge_emb, edge_map,
                                 id_to_row, chosen, g_nbrs)
        return chosen

    def _apply_move(self, state: GraphState, g, ids, h, edge_emb, edge_map,
                    id_to_row, target: int, g_nbrs: dict):
        """移动到 target：agent/节点更新 + visit 衰减（揭示由 reveal_node 完成）。

        返回更新后的 h（functional 更新）。卷积邻居取自可见图 g_nbrs。"""
        with torch.no_grad():
            moved_edge = None
            if self._model.with_edge:
                moved_edge = edge_emb[edge_map[(id_to_row[self._pos],
                                                id_to_row[target])]]
            conv_nbr = sorted(g_nbrs.get(id_to_row[target], set())
                              - {id_to_row[target]})          # 行号集合
            conv_nbr_t = torch.tensor(conv_nbr, dtype=torch.long,
                                      device=self._device)
            conv_e = None
            if self._model.with_edge:
                conv_e = torch.stack([
                    edge_emb[edge_map[(id_to_row[target], v)]]
                    for v in conv_nbr]) \
                    if conv_nbr else torch.zeros(
                        0, self._model.hidden, device=self._device)
            h, self._agent = self._model.step_update(
                h, self._agent, id_to_row[target], conv_nbr_t, conv_e,
                moved_edge)
            self._h_cache[target] = h[id_to_row[target]].clone()
            self._prev, self._pos = self._pos, target
            for k in self._visit:
                self._visit[k] *= self._model.visited_decay
            self._visit[target] = 1.0
        return h
