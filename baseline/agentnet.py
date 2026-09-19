"""AgentNet 基线（AGR 协议适配版，单 agent）。

复现自 AgentNet 官方实现（ICLR 2023 "Agent-based Graph Neural Networks"，
arXiv 2206.11010，本地 AgentNet-main 逐行对照）的**单 agent** 移植（AGR
接入约束：协议每步揭示 1 个节点）：
移动头 = Q·K 注意力（[邻居 ‖ 当前 ‖ 边] 编码）+ BSEU 探索偏置
（back/stay/explored/unexplored 可学习标量，visited 衰减 0.9），
gumbel-softmax(hard, straight-through) 选边；节点/agent 残差更新 +
逐步读出累加平均 → 分类。

2026-09-19 源码/论文审查后恢复的原版机制（此前未声明丢失，导致分类头
训练坍缩——见 docs/decisions.md）：
- mlp_width_mult：更新 MLP 隐层宽度倍数（原版 README 2）；
- reduce='log'：邻域卷积聚合 sum/log2(deg+1) 度归一化（原版 README）；
- 训练配置：batch 32（原版）/ epochs 100（原版 350）/ Adam 保留；
- 游走步数按预算比例现场计算 steps=max(1,int(ratio×n))（ratio 逐样本采样，
  训练与 own_predict 读出游走同式）——模型无 time conditioning、读出按
  1/(T+1) 归一，天然支持变步数。
明确接受的单 agent 代价：论文 Theorem 12 证明存在 1 agent 任意倍步数
无法解决、2 agent 可解的结构——多 agent 表达力损失为协议约束下的已知
局限；global_agent_pool（agent 间全局通信，附录 C.2）在 K=1 下退化为
自环广播，无信息增益，不实现。

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

评测性能（2026-09-19）：策略侧 per-graph 静态缓存（_GraphCache：全图
input_proj/edge_input_proj 编码 + 无向邻接）+ 行 gather 构造可见表征，
去掉每步 build_visible_graph/DGL 构图/全量重编码/全边 Python 循环；
own_predict 走 forward_batch 批图 lockstep 游走（eval_loop 按 512 批调用）。
选点行为与旧实现逐位一致（legacy ckpt 回归 12/12 + logits 逐位）；
own_predict 确定性化（argmax 移动 + seed 行起点，对齐 §7 test_argmax/
原则 4——旧实现走训练 forward，gumbel + 未播种随机起点，acc_own 不可
复现；另注：原版 test_argmax 的选择仍含 gumbel 噪声，我们更严格）。
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

    张量化实现：按 (min,max) 规范键去重（每键取首见边的特征），每条无向
    边补齐双向（平行边/单向边输入均归一）；边特征复制到两个方向
    （cifar10/ppa 边特征本身对称）。输出边的排列为 (min,max) 字典序——
    模型对边序置换不变（逐边独立编码 + 按节点查边聚合）。
    """
    src, dst = g.edges()
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    keys = torch.stack([lo, hi], dim=1)
    uniq, inv = torch.unique(keys, dim=0, return_inverse=True)
    # 每 key 首见边 id（对齐旧实现的 seen-set 首见语义）
    first = torch.empty(uniq.shape[0], dtype=torch.long, device=keys.device)
    first.scatter_reduce_(0, inv, torch.arange(keys.shape[0], device=keys.device),
                          reduce='amin', include_self=False)
    u, v = uniq[:, 0], uniq[:, 1]
    g2 = dgl.graph((torch.cat([u, v]), torch.cat([v, u])),
                   num_nodes=g.num_nodes())
    if 'feat' in g.edata:
        # 注意：dgl.graph 不保证按传入顺序存边（连续性不同的输入可能触发
        # 内部排序），特征须按输出边的规范键回查，不能按传入顺序赋值
        s2, d2 = g2.edges()
        code_u = uniq[:, 0].long() * g.num_nodes() + uniq[:, 1].long()
        code_e = (torch.minimum(s2, d2) * g.num_nodes()
                  + torch.maximum(s2, d2))
        rows = torch.searchsorted(code_u, code_e)   # 输出边键必在 uniq 中
        g2.edata['feat'] = g.edata['feat'][first[rows]]
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
                 gumbel_tau: float = 2.0 / 3.0, mlp_width_mult: int = 1,
                 reduce: str = 'sum'):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.hidden = hidden
        self.num_steps = num_steps
        self.dropout = dropout
        self.visited_decay = visited_decay
        self.gumbel_tau = gumbel_tau
        self.with_edge = edge_feat_dim > 0
        # 原版对齐项（2026-09-19 审查后恢复）：MLP 隐层宽度倍数与聚合归约。
        # 默认值 = 旧版单 agent 结构（旧 checkpoint 严格加载兼容）；新训练
        # 配方经 run_agentnet.py 传 width=2 / reduce='log'（原版 README）。
        self.mlp_width_mult = int(mlp_width_mult)
        self.reduce = reduce
        assert self.reduce in ('sum', 'log'), f"未知 reduce: {reduce}"

        act = nn.LeakyReLU(0.01)
        w = self.mlp_width_mult
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
        # 更新 MLP（pre-LN，对齐原版 post_ln=False；隐层宽度 = hidden*2*w）
        ed = hidden if self.with_edge else 0
        self.agent_node_lin = (nn.Sequential(
            nn.LayerNorm(hidden + ed), nn.Linear(hidden + ed, hidden), act)
            if self.with_edge else nn.Identity())
        self.node_mlp = _mlp(hidden * 2, hidden * 2 * w, hidden, dropout)
        self.message_val = (nn.Sequential(
            nn.LayerNorm(hidden + ed), nn.Linear(hidden + ed, hidden), act)
            if self.with_edge else None)
        self.conv_mlp = _mlp(hidden * 2, hidden * 2 * w, hidden, dropout)
        self.agent_mlp = _mlp(hidden * 2 + ed, hidden * 2 * w, hidden, dropout)
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

    def _reduce(self, agg: torch.Tensor, count):
        """聚合归约（原版 util.scatter reduce 语义）。'log' = sum/log2(count+1)
        （度归一化；count≤1 时除数为 1，恒等）——count 为聚合进该节点的
        消息条数（卷积 = 邻居数）。'sum' 恒等（旧版行为）。"""
        if self.reduce != 'log':
            return agg
        if isinstance(count, int):
            return agg / (math.log2(count + 1) if count > 0 else 1.0)
        cnt = count.clamp_min(1).float().unsqueeze(-1)
        return agg / torch.log2(cnt + 1.0)

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
            agg = self._reduce(agg, int(conv_nbr.numel()))
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
    def forward(self, g: dgl.DGLGraph, start: int = None,
                steps: int = None) -> torch.Tensor:
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

        T = self.num_steps if steps is None else int(steps)
        for i in range(T + 1):
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
        return (out / (T + 1)).unsqueeze(0)   # [1, n_classes]

    # ------------------------------------------------------------------
    # 批图 lockstep 游走前向（own_predict_batch 评测用）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_batch(self, g: dgl.DGLGraph, starts=None,
                      steps: int = None) -> torch.Tensor:
        """块对角批图（dgl.batch 产物，自环需已加）的确定性评测读出。

        对齐协议 test_argmax(τ→1e-6)/原则 4（protocol-adaptation.md §7）：
        移动 = argmax + 平分取小 id（无 gumbel 噪声），起点 = 每图首节点，
        或调用方给定的各图起点行号（AGR 传 seed 在揭示子图中的行号），
        attn_val ≡ 1（τ→0 硬 one-hot 的 straight-through 极限，与训练路径
        ret[idx]≈1 一致）。
        与 forward 单图数学逐式对齐，差异仅是批内按图 padding——pad 位经
        -inf 掩蔽不参与 argmax、经 mask 乘零不参与邻域聚合。同一输入必产
        出同一 logits（旧 own_predict 走训练 forward，gumbel+未播种随机
        起点不可复现）。返回 [B, n_classes]。
        """
        h = self.init_hidden(g)
        edge_emb = self.init_edge_emb(g)
        device = h.device
        B = g.batch_size

        # 邻接 u → [(v, eid)] 按 v 升序；eid 取首见（对齐 forward 的
        # edge_map.setdefault 首见语义）——每批一次 O(E) 构建
        src, dst = g.edges()
        nbr_eid = {}
        for e, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            nbr_eid.setdefault(u, {}).setdefault(v, e)
            nbr_eid.setdefault(v, {}).setdefault(u, e)
        nbr_lists = {u: sorted(d.items()) for u, d in nbr_eid.items()}

        def _pad(seqs, width):
            """[(v, eid)] 序列 → (节点 id [B,W], 边 id [B,W], 有效 mask [B,W])。"""
            rows, pos, vals, eids = [], [], [], []
            for r, seq in enumerate(seqs):
                rows += [r] * len(seq)
                pos += list(range(len(seq)))
                vals += [v for v, _ in seq]
                eids += [e for _, e in seq]
            t_v = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
            t_e = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
            t_m = torch.zeros(len(seqs), width, dtype=torch.bool, device=device)
            if rows:
                r_t = torch.tensor(rows, device=device)
                p_t = torch.tensor(pos, device=device)
                t_v[r_t, p_t] = torch.tensor(vals, device=device)
                t_e[r_t, p_t] = torch.tensor(eids, device=device)
                t_m[r_t, p_t] = True
            return t_v, t_e, t_m

        agent = self.agent_emb(torch.zeros(B, dtype=torch.long, device=device))
        moved_edge = (torch.zeros(B, self.hidden, device=device)
                      if self.with_edge else None)
        visit = torch.zeros(g.num_nodes(), device=device)
        bnn = g.batch_num_nodes()
        cur_t = torch.zeros(B, dtype=torch.long, device=device)
        cur_t[1:] = torch.cumsum(bnn, 0)[:-1]     # 每图首节点（确定性起点）
        if starts is not None:
            cur_t = cur_t + torch.as_tensor(starts, dtype=torch.long,
                                            device=device)
        visit[cur_t] = 1.0
        cur = cur_t.tolist()
        prev_t = None
        attn_val = torch.ones(B, device=device)
        out = None

        T = self.num_steps if steps is None else int(steps)
        for i in range(T + 1):
            if i > 0:
                seqs = [nbr_lists[c] for c in cur]           # 含自环 stay
                D = max(len(s) for s in seqs)
                cand_t, cand_eid, valid = _pad(seqs, D)
                q = self.query(agent)
                cur_feat = h[cur_t]
                if self.with_edge:
                    k_in = torch.cat([h[cand_t], edge_emb[cand_eid],
                                      cur_feat.unsqueeze(1).expand(-1, D, -1)],
                                     dim=-1)
                else:
                    k_in = torch.cat([h[cand_t],
                                      cur_feat.unsqueeze(1).expand(-1, D, -1)],
                                     dim=-1)
                k = self.key(k_in)
                attn = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(self.hidden)
                bias = torch.zeros_like(attn)
                is_cur = cand_t == cur_t.unsqueeze(1)
                s = visit[cand_t] / self.visited_decay
                term = s * self.explored_param \
                    + (1.0 - s) * self.unexplored_param
                bias = bias + term * (~is_cur & valid)
                bias = bias + self.stay_param * is_cur
                if prev_t is not None:
                    bias = bias + self.back_param * \
                        ((cand_t == prev_t.unsqueeze(1)) & valid)
                logits = (attn + bias).masked_fill(~valid, float('-inf'))
                # test_argmax：pad 已 -inf 不会被选中；argmax 平分取最小
                # 行号（行序 = 候选 v 升序），attn_val = τ→0 硬 one-hot 极限
                idx = logits.argmax(dim=-1)
                attn_val = torch.ones(B, device=device)
                sel = idx.unsqueeze(1)
                prev_t = cur_t
                cur_t = cand_t.gather(1, sel).squeeze(1)
                cur = cur_t.tolist()
                if self.with_edge:
                    moved_edge = edge_emb[cand_eid.gather(1, sel).squeeze(1)]
                visit = visit * self.visited_decay
                visit[cur_t] = 1.0

            # 邻域卷积（去 stay 自环）+ agent/节点更新 + 读出累加
            conv_seqs = [[(v, e) for v, e in nbr_lists[c] if v != c]
                         for c in cur]
            Dc = max(len(s) for s in conv_seqs)
            if Dc > 0:
                conv_t, conv_eid, conv_valid = _pad(conv_seqs, Dc)
                if self.with_edge:
                    agg = (self.message_val(
                        torch.cat([h[conv_t], edge_emb[conv_eid]], dim=-1))
                        * conv_valid.unsqueeze(-1)).sum(dim=1)
                else:
                    agg = (h[conv_t] * conv_valid.unsqueeze(-1)).sum(dim=1)
                agg = self._reduce(agg, conv_valid.sum(dim=1))
            else:
                agg = torch.zeros(B, self.hidden, device=device)
            if self.with_edge:
                me = moved_edge * attn_val.unsqueeze(-1)
                a_msg = self.agent_node_lin(torch.cat([agent, me], dim=-1))
            else:
                a_msg = agent
            cur_feat = h[cur_t]
            new_h_pos = cur_feat + self.node_mlp(
                torch.cat([cur_feat, a_msg], dim=-1))
            new_h_pos = new_h_pos + self.conv_mlp(
                torch.cat([new_h_pos, agg], dim=-1))
            h = h.index_copy(0, cur_t, new_h_pos)
            if self.with_edge:
                agent_cat = torch.cat(
                    [agent, new_h_pos * attn_val.unsqueeze(-1), me], dim=-1)
            else:
                agent_cat = torch.cat(
                    [agent, new_h_pos * attn_val.unsqueeze(-1)], dim=-1)
            agent = agent + self.agent_mlp(agent_cat)
            layer_out = self.fc(self.step_readout(agent))
            out = layer_out if out is None else out + layer_out
        return out / (T + 1)                  # [B, n_classes]


class _GraphCache:
    """单图静态量缓存（跨 rollout 共享）：全图编码 + 无向邻接。

    select 逐步重建可见图 + 全量重编码的等价替代：节点/边特征是图的静态
    属性，input_proj/edge_input_proj 逐行独立 → 全图一次前向后按行 gather，
    与逐步重前向一致；frontier 节点特征置零的编码是全节点同一常数行
    （input_proj(0)）。换图重建（eval 逐图连续跑全部 rollout，天然命中）。
    """

    __slots__ = ('adj', 'x_proj', 'e_proj', 'zero_row')

    def __init__(self, model: 'AgentNetModel', state: GraphState, device):
        g = state.g
        # 无向邻接（去自身/并行边去重；原图 id 升序）——一次 O(E) 构建
        # 替代 select 内逐步的全边 Python 循环
        src, dst = g.edges()
        adj = {}
        for u, v in zip(src.tolist(), dst.tolist()):
            if u == v:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        self.adj = {k: sorted(s) for k, s in adj.items()}
        if model.input_proj is not None:
            self.x_proj = model.input_proj(g.ndata['feat'].to(device))
            self.zero_row = model.input_proj(
                torch.zeros(1, state.node_feat_dim, device=device))
        else:
            self.x_proj = None
            self.zero_row = None
        self.e_proj = (model.edge_input_proj(g.edata['feat'].to(device))
                       if model.with_edge else None)

    def edge_id(self, state: GraphState, u: int, v: int) -> int:
        """(u,v) 的原图边 id（键与 GraphState._edge_key_to_id 同为 min/max 规范）。"""
        return state._edge_key_to_id[(min(u, v), max(u, v))]


class AgentnetStrategy(BaseReader):
    """单 agent 游走 → 逐点揭示（协议合规评测形态；有状态，reset 重建）。

    评测快速路径：per-graph 静态缓存（_GraphCache）+ 行 gather 构造可见
    表征，不再每步经 build_visible_graph 重建 DGL 图/重编码/重建邻接，
    选点行为与旧实现逐位一致（见各方法内等价性说明）。
    """

    NAV_CAP_FACTOR = 4   # 已揭示区内导航步数上限 = 4 × n_nodes（对齐 random-walk）

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None
        self._gc_graph = None    # 缓存对应的图对象（identity 判定换图）
        self._gc = None          # _GraphCache
        self._reset_walk()

    def _reset_walk(self):
        self._pos = None          # agent 当前位置（原图 id）
        self._prev = None
        self._visit = {}          # orig_id → 衰减访问强度
        self._h_cache = {}        # orig_id → 更新后的节点嵌入
        self._agent = None
        self._nav_steps = 0
        # 注意：不清 _gc/_gc_graph——同图多个 rollout 共享静态缓存

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
            dropout=cfg['dropout'], visited_decay=cfg['visited_decay'],
            mlp_width_mult=cfg.get('mlp_width_mult', 1),
            reduce=cfg.get('reduce', 'sum'))
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._gc_graph = None   # 权重变了，静态缓存失效
        self._gc = None

    def reset(self) -> None:
        self._reset_walk()

    def own_predict(self, sub_g, seed_row=None, steps=None):
        """自身读出头对揭示子图的分类（评测双口径记录，对齐 evidence 的
        acc_* / acc_clf_* 惯例：协议终判仍为冻结分类器，本头为参照）。

        输入 SubgraphBuilder 产出的揭示子图（已双向+自环+特征）；按训练
        约定重整构图后确定性 agent 游走读出（argmax + seed 行起点）→
        [1, n_classes] logits。同一子图必产出同一 logits。seed_row 为
        seed 在子图中的行号（builder original_ids 内的下标），缺省 0。
        """
        return self.own_predict_batch([sub_g],
                                      [seed_row] if seed_row is not None else None,
                                      [steps] if steps is not None else None)

    def own_predict_batch(self, subgraphs, seed_rows=None, steps=None):
        """own_predict 批量版：symmetrize → dgl.batch → lockstep 游走前向。

        seed_rows: 各子图内 seed 的行号（缺省 None=每图首节点）；
        steps: 各子图读出游走步数（缺省 None=模型 num_steps）。步数不均
        时按步数分组分别批前向（图间独立，结果与逐图一致），避免 pad 浪费。
        返回 [B, n_classes]。
        """
        gs = [dgl.add_self_loop(symmetrize_with_feat(dgl.remove_self_loop(s)))
              for s in subgraphs]
        B = len(gs)
        out = torch.zeros(B, self._model.fc.weight.shape[0])
        if steps is None:
            groups = [(None, list(range(B)))]
        else:
            order = sorted(range(B), key=lambda i: steps[i])
            groups, i = [], 0
            while i < B:
                j = i
                while j < B and steps[order[j]] == steps[order[i]]:
                    j += 1
                groups.append((int(steps[order[i]]), order[i:j]))
                i = j
        for st_val, idx in groups:
            bg = dgl.batch([gs[k] for k in idx]).to(self._device)
            sr = ([seed_rows[k] for k in idx]
                  if seed_rows is not None else None)
            out[torch.tensor(idx, dtype=torch.long)] =                 self._model.forward_batch(bg, starts=sr, steps=st_val)
        return out

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

    @torch.no_grad()
    def select(self, graph_state: GraphState) -> 'int | None':
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._model is None:
            raise RuntimeError("agentnet 策略需先 load(checkpoint) "
                               "（eval.py --strategy agentnet 自动完成）")
        if self._pos is None:
            self.on_new_frontier(sorted(frontier))

        m = self._model
        if self._gc_graph is not graph_state.g:
            self._gc_graph = graph_state.g
            self._gc = _GraphCache(m, graph_state, self._device)
        cache = self._gc

        # 可见表征行序与 build_visible_graph 一致：revealed 升序在前、
        # frontier 升序在后。h 初始行 = 全图编码按行 gather（frontier 行
        # 用置零特征编码 zero_row）+ 游走记忆覆盖 —— 与旧实现"重建可见图
        # → init_hidden 全量重前向 → 覆盖 _h_cache"逐位一致（同输入同权重
        # 的 Linear 逐行独立），但无任何逐步前向/DGL 构图。
        revealed = graph_state.get_revealed_sorted()
        frontier_nodes = graph_state.get_frontier_nodes()
        ids = revealed + frontier_nodes
        row_of = {old: i for i, old in enumerate(ids)}
        if cache.x_proj is not None:
            h = torch.cat([
                cache.x_proj[revealed] if revealed else
                torch.zeros(0, m.hidden, device=self._device),
                cache.zero_row.expand(len(frontier_nodes), -1)])
        else:
            h = m.node_token.repeat(len(ids), 1)
        for old, hv in self._h_cache.items():
            h[row_of[old]] = hv

        cand = self._cand_from_frontier(graph_state)
        frontier_set = set(frontier_nodes)
        # ---- 死角：在已揭示区内用 BSEU 打分导航（不消耗预算）----
        nav_cap = self.NAV_CAP_FACTOR * graph_state.n_nodes_total
        while not cand and self._nav_steps < nav_cap:
            self._nav_steps += 1
            nbrs = set(cache.adj.get(self._pos, ())) & graph_state.V_revealed
            if not nbrs:
                return None
            nbrs = sorted(nbrs)
            nbr_t = torch.tensor([row_of[v] for v in nbrs],
                                 dtype=torch.long, device=self._device)
            nbr_e = None
            if m.with_edge:
                nbr_e = torch.stack([
                    cache.e_proj[cache.edge_id(graph_state, self._pos, v)]
                    for v in nbrs])
            visit = torch.tensor([self._visit.get(v, 0.0) for v in nbrs],
                                 device=self._device)
            is_prev = torch.tensor([v == self._prev for v in nbrs],
                                   dtype=torch.bool, device=self._device) \
                if self._prev is not None else None
            logits = m.move_logits(
                h, self._agent, row_of[self._pos], nbr_t, nbr_e,
                visit, is_prev)
            nxt = nbrs[m.move_choice_eval(logits)]
            h = self._apply_move(graph_state, cache, row_of, h, nxt,
                                 frontier_set)
            cand = self._cand_from_frontier(graph_state)
        if not cand:
            return None

        # ---- 揭示步：frontier 候选打分 → argmax（平分取小 id）----
        cand_t = torch.tensor([row_of[v] for v in cand],
                              dtype=torch.long, device=self._device)
        cand_e = None
        if m.with_edge:
            cand_e = torch.stack([
                cache.e_proj[cache.edge_id(graph_state, self._pos, v)]
                for v in cand])
        visit = torch.tensor([self._visit.get(v, 0.0) for v in cand],
                             device=self._device)
        is_prev = torch.tensor([v == self._prev for v in cand],
                               dtype=torch.bool, device=self._device) \
            if self._prev is not None else None
        logits = m.move_logits(
            h, self._agent, row_of[self._pos], cand_t, cand_e,
            visit, is_prev)
        chosen = cand[m.move_choice_eval(logits)]
        h = self._apply_move(graph_state, cache, row_of, h, chosen,
                             frontier_set)
        return chosen

    def _apply_move(self, state: GraphState, cache: _GraphCache,
                    row_of: dict, h: torch.Tensor, target: int,
                    frontier_set: set) -> torch.Tensor:
        """移动到 target（原图 id）：agent/节点更新 + visit 衰减（揭示由
        reveal_node 完成）。返回更新后的 h（functional 更新）。

        卷积邻居 = target 的可见邻居（原图 id），行序 = 已揭示（升序）在
        前、frontier（升序）在后 —— 与旧实现按可见图行号升序的聚合序列
        逐位一致（可见图行 0..R-1 = revealed 升序，R.. = frontier 升序）：
        - target 已揭示（导航步）：与 revealed 的边都在 revealed_edge_keys、
          与 frontier 的真实边都是 frontier 边（一端已揭示），全部可见；
        - target 未揭示（揭示步）：只有 frontier 边可见，即 revealed 邻居；
          frontier–frontier 的原图边不可见，不得进入卷积。
        无边特征图的历史行为（revealed_edge_keys 恒空 → 可见图缺
        revealed-revealed 边）按原样保留。"""
        with torch.no_grad():
            m = self._model
            moved_edge = None
            if m.with_edge:
                moved_edge = cache.e_proj[
                    cache.edge_id(state, self._pos, target)]
            nbrs = cache.adj.get(target, ())
            if target in state.V_revealed:
                rev = ([v for v in nbrs if v in state.V_revealed]
                       if state.has_edge_feat else [])
                conv_orig = sorted(rev) + sorted(v for v in nbrs
                                                 if v in frontier_set)
            else:
                conv_orig = sorted(v for v in nbrs if v in state.V_revealed)
            conv_nbr = torch.tensor([row_of[v] for v in conv_orig],
                                    dtype=torch.long, device=self._device)
            conv_e = None
            if m.with_edge:
                conv_e = torch.stack([
                    cache.e_proj[cache.edge_id(state, target, v)]
                    for v in conv_orig]) \
                    if conv_orig else torch.zeros(
                        0, m.hidden, device=self._device)
            h, self._agent = m.step_update(
                h, self._agent, row_of[target], conv_nbr, conv_e, moved_edge)
            self._h_cache[target] = h[row_of[target]].clone()
            self._prev, self._pos = self._pos, target
            for k in self._visit:
                self._visit[k] *= m.visited_decay
            self._visit[target] = 1.0
        return h
