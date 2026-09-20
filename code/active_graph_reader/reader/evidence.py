"""EvidenceReader：base 选点路径 + LSTM 证据积分器 + 历史引导注意力读出。

设计：docs/design/evidence-reader.md（v2）。三条硬护栏：
1. 单通路——唯一 GNN 是共享的 FrontierNodesEncoder，Δ_t（新揭示节点嵌入）
   取自同一次前向的输出，无第二编码器；
2. 推理路径零改动——选点打分与 base 逐字节一致（无状态），证据流不参与
   选点，rollout/评估可直接复用 base 机制；
3. 递归只住在预测头——LSTM/注意力的梯度仅来自监督 CE（L_ev），
   不进策略梯度。

每步证据流（state 独立）：
  Δ_t    = 编码器对"上一步所揭示节点"的嵌入（step 0 为 seed）
  h_t    = LSTM(Δ_t, h_{t-1})                       证据积分器
  z_t    = AttnPool(query=h_t, K/V=已揭示节点嵌入)   历史引导读出
  logits = MLP([z_t ; h_t])                          逐步图类别预测
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ReaderConfig
from .learnable_reader import LearnableReader
from .common import stable_probs as _stable_probs


class HistoryPool(nn.Module):
    """单头缩放点积注意力池化：query=(d,) 历史态，K/V=(n,d) 已揭示嵌入 → (d,)。"""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """query (d,)；kv (n, d) → (d,)。n=0 时返回零向量（无可读证据）。"""
        if kv.shape[0] == 0:
            return torch.zeros(self.d, device=query.device, dtype=query.dtype)
        q = self.q_proj(query)                                    # (d,)
        K = self.k_proj(kv)                                       # (n, d)
        V = self.v_proj(kv)                                       # (n, d)
        attn = torch.softmax(K @ q / (self.d ** 0.5), dim=0)      # (n,)
        return attn @ V                                           # (d,)


class EvidenceReader(LearnableReader):
    """证据 reader：选点 = base（无状态）；预测 = 证据累积流（带历史）。

    与 base 的唯一结构差异是三个证据模块（evidence_lstm / evidence_pool /
    evidence_head），参数 ≈ +200K（h128、10 类），前向开销 ≈ base。
    """

    def __init__(
        self,
        edge_feat_dim: int = 7,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        stochastic: bool = False,
        temperature: float = 1.0,
        node_feat_dim: int = 0,
        n_classes: int = 0,
        config: Optional[ReaderConfig] = None,
    ):
        if config is None:
            config = ReaderConfig.from_legacy(
                kind='evidence', edge_feat_dim=edge_feat_dim,
                node_feat_dim=node_feat_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, dropout=dropout,
                stochastic=stochastic, temperature=temperature,
                n_classes=n_classes)
        config.validate()
        super().__init__(config=config)
        d = self.config.hidden_dim   # 证据流工作在编码器输出宽度上
        # 消融变体（docs/design/evidence-reader-ablation.md）：被去掉的模块
        # 不创建（state_dict 键随之收敛），evidence_head 恒在
        self.evidence_variant = self.config.evidence_variant
        if self.evidence_variant not in ('no_lstm', 'v1'):
            self.evidence_lstm = nn.LSTM(d, d)            # seq=1，batch_first=False
        if self.evidence_variant not in ('no_pool', 'v1'):
            self.evidence_pool = HistoryPool(d)
        self.evidence_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, self.config.n_classes),
        )

    # ------------------------------------------------------------------
    # 证据流
    # ------------------------------------------------------------------

    def init_evidence(self, n: int, device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """n 个 state 的初始 (h, c)（零）。锁步循环按 state 独立维护、
        rollout 边界重建。"""
        return [(torch.zeros(1, self.config.hidden_dim, device=device),
                 torch.zeros(1, self.config.hidden_dim, device=device))
                for _ in range(n)]

    def forward_states_evidence(
        self, states, temperature: float,
        hc_list: List[Tuple[torch.Tensor, torch.Tensor]],
        last_revealed: List[int],
        return_states: bool = False,
    ):
        """base 选点 + 证据流，单次共享 GNN 前向。

        states/temperature 语义同 forward_states；hc_list 为各 state 当前
        (h, c)；last_revealed 为各 state 上一步所揭示节点（step 0 = seed），
        其嵌入取自本次前向（lazy Δ，无第二编码器）。

        return_states=True 时额外返回各 state 本步的 (z_t, h_t)——供冻结
        reader 的离线聚合头（train_state_attention.py）采集轨迹特征；只增
        不改，默认 False 时返回值与原签名逐字节一致。

        Returns:
            results   : List[(scores, probs, frontier_nodes)]——与 base
                        forward_states 逐字节一致；
            new_hc    : List[(h, c)] 更新后的积分器状态；
            ev_logits : List[Tensor (n_classes,)] 各 state 逐步预测；
            step_states（仅 return_states=True）: List[(z_t, h_t)]，各
                        (d,)/ (d,)。
        """
        if not states:
            if return_states:
                return [], [], [], []
            return [], [], []

        subgraphs = [self.frontier_builder.build(s) for s in states]
        all_embeds = self.frontier_encoder.forward_all_batch(subgraphs)

        d = self.config.hidden_dim
        deltas, hs, cs = [], [], []
        for k, sg in enumerate(subgraphs):
            h, c = hc_list[k]
            revealed_sorted = states[k].get_revealed_sorted()
            lr = last_revealed[k]
            if lr is not None and lr in revealed_sorted:
                pos = revealed_sorted.index(lr)
                deltas.append(all_embeds[k][pos])             # (d,)
            else:
                deltas.append(torch.zeros(d, device=h.device, dtype=h.dtype))
            hs.append(h.squeeze(0))                            # (d,)
            cs.append(c.squeeze(0))

        # 积分器：full/no_pool 走 LSTM；no_lstm/v1 递归关闭，h_t = Δ_t 直通，
        # (h, c) 原样占位返回（循环侧语义不变，状态不被消费）
        if self.evidence_variant in ('no_lstm', 'v1'):
            h_new = torch.stack(deltas)                            # (N, d)
            new_hc = list(hc_list)
        else:
            # 一次批量 LSTM 步：输入 (1, N, d)，状态 (1, N, d)
            _, (h_seq, c_seq) = self.evidence_lstm(
                torch.stack(deltas).unsqueeze(0),
                (torch.stack(hs).unsqueeze(0), torch.stack(cs).unsqueeze(0)),
            )
            h_new = h_seq[0]                                       # (N, d)
            c_new = c_seq[0]                                       # (N, d)
            new_hc = [(h_new[i:i + 1], c_new[i:i + 1]) for i in range(len(states))]

        results: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
        ev_logits: List[torch.Tensor] = []
        step_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k, sg in enumerate(subgraphs):
            h_all = all_embeds[k]
            n_rev = sg.n_revealed
            frontier_embeds = h_all[n_rev:]
            frontier_nodes = sg.frontier_original_ids

            if frontier_nodes:
                scores = self._policy_scores(frontier_embeds, sg, states[k])
                probs = _stable_probs(scores, temperature)
                results.append((scores, probs, frontier_nodes))
            else:
                results.append((torch.empty(0), torch.empty(0), []))

            h_t = h_new[k]                                     # (d,)
            # 读出：full/no_lstm 用历史引导注意力；no_pool/v1 退化为对已揭示
            # 嵌入的非引导均值池化（无可读证据时零向量）
            if self.evidence_variant in ('no_pool', 'v1'):
                r_t = h_all[:n_rev]
                z_t = (r_t.mean(dim=0) if r_t.shape[0] > 0 else
                       torch.zeros(d, device=h_t.device, dtype=h_t.dtype))
            else:
                z_t = self.evidence_pool(h_t, h_all[:n_rev])   # 历史引导读出
            ev_logits.append(self.evidence_head(torch.cat([z_t, h_t])))
            if return_states:
                step_states.append((z_t, h_t))

        if return_states:
            return results, new_hc, ev_logits, step_states
        return results, new_hc, ev_logits

    @property
    def name(self) -> str:
        return 'evidence'
