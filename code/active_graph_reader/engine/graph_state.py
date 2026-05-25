"""
图状态管理模块
==============

管理揭示过程中图的部分可观测状态。是协议中 G_t 的程序化表示：

    G_t = (V_revealed^(t), E_known^(t), X_E^(t), X_V^(t))

状态跟踪:
- V_revealed        : 已揭示节点集合
- frontier_edges    : 前沿边集合 {(revealed, unrevealed), ...}
- revealed_edge_feats: 已揭示边特征 {(min(u,v), max(u,v)) → tensor}
- frontier_degree   : 每个前沿节点与已揭示子图的连接边数

所有查询方法为纯读取，修改通过 reveal_action 模块进行。
"""

import logging
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

import torch
import dgl

logger = logging.getLogger(__name__)


class GraphState:
    """图的部分可观测状态。

    封装揭示过程中所有可知信息，并对外提供查询接口供策略和成本模型使用。

    Attributes
    ----------
    g : dgl.DGLGraph
        原始完整图（用于查询节点邻居、边特征等"真值"）。
    seed_node : int
        种子节点 ID。
    V_revealed : Set[int]
        已揭示节点集合。
    frontier_edges : Set[Tuple[int, int]]
        前沿边集合，每条为 (revealed, unrevealed)。
    revealed_edge_feats : Dict[Tuple[int, int], torch.Tensor]
        已揭示边特征，key 为无序对 (min, max)。
    n_nodes_total : int
        图中总节点数（用于统计，不暴露给策略）。
    has_edge_feat : bool
        原始图是否有边特征。
    """

    __slots__ = (
        'g', 'seed_node', 'V_revealed', 'frontier_edges',
        'revealed_edge_feats', '_frontier_degree',
        'n_nodes_total', 'n_edges_total',
        'has_edge_feat', 'edge_feat', 'edge_feat_dim',
        'reveal_history', '_budget_used', '_initial_frontier',
    )

    def __init__(self, dgl_graph: dgl.DGLGraph, seed_node: int):
        """初始化 G_0 状态（方案 D：k-Seed，k=1）。

        Parameters
        ----------
        dgl_graph : dgl.DGLGraph
            原始完整图。
        seed_node : int
            种子节点 ID。
        """
        self.g = dgl_graph
        self.seed_node = seed_node
        self.n_nodes_total = dgl_graph.num_nodes()
        self.n_edges_total = dgl_graph.num_edges()

        # 边特征
        self.has_edge_feat = 'feat' in dgl_graph.edata
        self.edge_feat = dgl_graph.edata['feat'] if self.has_edge_feat else None
        self.edge_feat_dim = (
            self.edge_feat.shape[1] if self.has_edge_feat else 0
        )

        # ---- 揭示状态 ----
        self.V_revealed: Set[int] = set()
        self.frontier_edges: Set[Tuple[int, int]] = set()
        self.revealed_edge_feats: Dict[Tuple[int, int], torch.Tensor] = {}
        self._frontier_degree: Dict[int, int] = defaultdict(int)

        self.reveal_history: List[int] = []
        self._budget_used: int = 0

        # 初始前沿边（按发现顺序排列，供 RevealLoop 通知策略）
        self._initial_frontier: List[Tuple[int, int]] = []

        # ---- 初始化 G_0 ----
        self._initial_frontier = self._init_g0()

    # ===================================================================
    # G_0 初始化
    # ===================================================================

    def _init_g0(self) -> List[Tuple[int, int]]:
        """从种子节点初始化 G_0。

        Returns
        -------
        List[Tuple[int, int]]
            初始前沿边列表（按邻居 ID 升序排列，保证确定性）。
        """
        s = self.seed_node
        self.V_revealed.add(s)
        self.reveal_history.append(s)

        new_frontier: List[Tuple[int, int]] = []
        neighbors = self._get_neighbors(s)

        for nbr in neighbors:
            if nbr == s:
                continue
            edge = (s, nbr)
            self.frontier_edges.add(edge)
            self._frontier_degree[nbr] += 1
            new_frontier.append(edge)

        logger.debug(
            "G_0 初始化完成: 种子=%d, 邻居=%d, 前沿边=%d",
            s, len(neighbors), len(new_frontier),
        )
        return new_frontier

    # ===================================================================
    # 查询接口（供策略和成本模型使用）
    # ===================================================================

    @property
    def is_frontier_empty(self) -> bool:
        """前沿边集合是否为空。"""
        return len(self.frontier_edges) == 0

    def get_frontier_degree(self, node: int) -> int:
        """获取前沿节点 node 与已揭示子图之间的连接边数（局部度）。

        仅统计 node 与 V_revealed 中节点的边数，非原始完整图的度。
        """
        return self._frontier_degree.get(node, 0)

    def get_revealed_ratio(self) -> float:
        """已揭示节点占总节点数的比例（含种子）。"""
        if self.n_nodes_total == 0:
            return 0.0
        return len(self.V_revealed) / self.n_nodes_total

    def get_budget_used(self) -> int:
        """已使用的预算（揭示节点数，不含种子）。"""
        return self._budget_used

    @property
    def n_revealed(self) -> int:
        """已揭示节点数（含种子）。"""
        return len(self.V_revealed)

    # ===================================================================
    # 内部辅助方法
    # ===================================================================

    def _get_neighbors(self, node: int) -> List[int]:
        """获取节点的所有邻居（去重、排序，兼容无向/有向图）。

        返回值按节点 ID 升序排列，保证对同一图的重复运行产生
        完全相同的邻居序列——这对 BFS 策略的确定性至关重要。

        对于无向图（DGL 中通常存储为双向边），successors 与
        predecessors 返回相同集合；取并集可兼容有向图场景。
        """
        succ = set(self.g.successors(node).tolist())
        pred = set(self.g.predecessors(node).tolist())
        return sorted(succ | pred)

    def _get_edge_feat(self, u: int, v: int) -> torch.Tensor:
        """获取无向边 {u, v} 的特征张量。

        先尝试 u→v 方向，若不存在则尝试 v→u。
        对于双向存储的图，仅返回一个方向的特征（两者相同）。
        返回值已 squeeze 掉 batch 维，shape = (feat_dim,)。
        """
        if not self.has_edge_feat:
            raise RuntimeError("图无边特征，无法获取边特征")

        u_t = torch.tensor([u])
        v_t = torch.tensor([v])

        if self.g.has_edges_between(u_t, v_t).item():
            eid = self.g.edge_ids(u_t, v_t)
            # edge_ids 返回 shape=(k,) 的 tensor；对 OGBG-PPA 双向存储，
            # k=1，故 squeeze 到 (dim,) 便于下游消费。
            return self.edge_feat[eid].squeeze(0).clone()

        if self.g.has_edges_between(v_t, u_t).item():
            eid = self.g.edge_ids(v_t, u_t)
            return self.edge_feat[eid].squeeze(0).clone()

        raise ValueError(f"边 ({u}, {v}) 在图中不存在")

    def _increment_budget(self) -> None:
        """预算计数器 +1。"""
        self._budget_used += 1

    # ===================================================================
    # 统计信息
    # ===================================================================

    def stats(self) -> dict:
        """返回当前揭示状态的统计字典。"""
        return {
            'seed': self.seed_node,
            'n_total': self.n_nodes_total,
            'n_edges_total': self.n_edges_total,
            'n_revealed': len(self.V_revealed),
            'n_frontier_remaining': len(self.frontier_edges),
            'n_revealed_edge_feats': len(self.revealed_edge_feats),
            'budget_used': self._budget_used,
            'reveal_ratio': round(self.get_revealed_ratio(), 4),
            'reveal_history': self.reveal_history.copy(),
            'has_edge_feat': self.has_edge_feat,
            'edge_feat_dim': self.edge_feat_dim,
        }
