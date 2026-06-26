"""图状态管理模块：管理揭示过程中图的部分可观测状态 G_t。"""

import bisect
import logging
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

import torch
import dgl

logger = logging.getLogger(__name__)


class GraphState:
    """图的部分可观测状态 G_t = (V_revealed, E_known, X_E, X_V)。

    封装揭示过程中所有可知信息，提供查询接口供策略和成本模型使用。
    修改通过 reveal_action 模块进行。
    """

    __slots__ = (
        'g', 'seed_node', 'V_revealed', 'frontier_edges',
        'revealed_edge_keys', '_edge_key_to_id',
        '_frontier_degree', '_frontier_by_unrevealed',
        'n_nodes_total', 'n_edges_total',
        'has_edge_feat', 'edge_feat', 'edge_feat_dim',
        'has_node_feat', 'node_feat', 'node_feat_dim',
        'reveal_history', '_budget_used', '_initial_frontier',
        '_sorted_revealed', '_sorted_frontier_nodes', '_revealed_edge_ids',
    )

    def __init__(
        self,
        dgl_graph: dgl.DGLGraph,
        seed_node: int,
        edge_key_to_id: Optional[Dict[Tuple[int, int], int]] = None,
    ):
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

        # 节点特征（PPA 无节点特征，预留接口适配其他数据集）
        self.has_node_feat = 'feat' in dgl_graph.ndata
        self.node_feat = dgl_graph.ndata['feat'] if self.has_node_feat else None
        self.node_feat_dim = self.node_feat.shape[1] if self.has_node_feat else 0

        # 预计算 canonical key → edge_id 映射（O(1) 边特征查找）
        # Accept pre-built dict for reuse across trajectories (M4 optimization)
        if edge_key_to_id is not None:
            self._edge_key_to_id = edge_key_to_id
        else:
            self._edge_key_to_id: Dict[Tuple[int, int], int] = {}
            if self.has_edge_feat:
                src = dgl_graph.edges()[0]
                dst = dgl_graph.edges()[1]
                for eid in range(dgl_graph.num_edges()):
                    u, v = int(src[eid]), int(dst[eid])
                    key = (min(u, v), max(u, v))
                    if key not in self._edge_key_to_id:
                        self._edge_key_to_id[key] = eid

        self.V_revealed: Set[int] = set()
        self.frontier_edges: Set[Tuple[int, int]] = set()
        self.revealed_edge_keys: Set[Tuple[int, int]] = set()
        self._frontier_degree: Dict[int, int] = defaultdict(int)
        self._frontier_by_unrevealed: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

        self.reveal_history: List[int] = []
        self._budget_used: int = 0
        self._initial_frontier: List[Tuple[int, int]] = []

        # 增量排序维护（避免 FrontierEncoder 每步重新排序）
        self._sorted_revealed: List[int] = []
        self._sorted_frontier_nodes: List[int] = []
        self._revealed_edge_ids: List[int] = []

        self._initial_frontier = self._init_g0()

    def _init_g0(self) -> List[Tuple[int, int]]:
        s = self.seed_node
        self.V_revealed.add(s)
        self.reveal_history.append(s)
        self._sorted_revealed.append(s)  # single element, already sorted

        new_frontier: List[Tuple[int, int]] = []
        neighbors = self._get_neighbors(s)

        for nbr in neighbors:
            if nbr == s:
                continue
            edge = (s, nbr)
            self.frontier_edges.add(edge)
            self._frontier_degree[nbr] += 1
            self._frontier_by_unrevealed[nbr].add(edge)
            new_frontier.append(edge)
            bisect.insort(self._sorted_frontier_nodes, nbr)

        logger.debug("G_0 初始化完成: 种子=%d, 邻居=%d, 前沿边=%d", s, len(neighbors), len(new_frontier))
        return new_frontier

    @property
    def is_frontier_empty(self) -> bool:
        return len(self.frontier_edges) == 0

    def get_frontier_edge_feat(self, u: int, v: int):
        """获取连接已揭示节点和前沿节点的边特征。要求至少一端在 V_revealed 中。"""
        if not self.has_edge_feat:
            raise ValueError("图中无边特征")
        if u not in self.V_revealed and v not in self.V_revealed:
            raise ValueError(f"frontier edge ({u}, {v}) 两端点都不在 V_revealed 中")
        return self._get_edge_feat(u, v)

    def get_frontier_degree(self, node: int) -> int:
        return self._frontier_degree.get(node, 0)

    def get_frontier_nodes(self) -> List[int]:
        return self._sorted_frontier_nodes  # already a list, sorted by construction

    def get_revealed_sorted(self) -> List[int]:
        return self._sorted_revealed  # already a list, sorted by construction

    def get_frontier_edges_of(self, node: int) -> List[Tuple[int, int]]:
        return list(self._frontier_by_unrevealed.get(node, []))

    def get_revealed_ratio(self) -> float:
        if self.n_nodes_total == 0:
            return 0.0
        return len(self.V_revealed) / self.n_nodes_total

    def get_budget_used(self) -> int:
        return self._budget_used

    @property
    def n_revealed(self) -> int:
        return len(self.V_revealed)

    def _add_to_revealed(self, v: int) -> None:
        bisect.insort(self._sorted_revealed, v)

    def _add_frontier_node(self, node: int) -> None:
        bisect.insort(self._sorted_frontier_nodes, node)

    def _remove_frontier_node(self, node: int) -> None:
        idx = bisect.bisect_left(self._sorted_frontier_nodes, node)
        if idx < len(self._sorted_frontier_nodes) and self._sorted_frontier_nodes[idx] == node:
            self._sorted_frontier_nodes.pop(idx)

    def _add_revealed_edge_id(self, eid: int) -> None:
        self._revealed_edge_ids.append(eid)

    def _get_neighbors(self, node: int) -> List[int]:
        """邻居 ID 升序排列，保证 BFS 确定性。兼容有向/无向图。"""
        succ = set(self.g.successors(node).tolist())
        pred = set(self.g.predecessors(node).tolist())
        return sorted(succ | pred)

    def _get_edge_feat(self, u: int, v: int) -> torch.Tensor:
        """O(1) 查找边特征（已 clone）。"""
        if not self.has_edge_feat:
            raise RuntimeError("图无边特征")
        key = (min(u, v), max(u, v))
        eid = self._edge_key_to_id.get(key)
        if eid is None:
            raise ValueError(f"边 ({u}, {v}) 在图中不存在")
        return self.edge_feat[eid].clone()

    def _get_edge_feat_ref(self, u: int, v: int) -> torch.Tensor:
        """O(1) 查找边特征（只读引用，避免内存分配）。"""
        if not self.has_edge_feat:
            raise RuntimeError("图无边特征")
        key = (min(u, v), max(u, v))
        eid = self._edge_key_to_id.get(key)
        if eid is None:
            raise ValueError(f"边 ({u}, {v}) 在图中不存在")
        return self.edge_feat[eid]

    def _increment_budget(self) -> None:
        self._budget_used += 1

    def stats(self) -> dict:
        return {
            'seed': self.seed_node,
            'n_total': self.n_nodes_total,
            'n_edges_total': self.n_edges_total,
            'n_revealed': len(self.V_revealed),
            'n_frontier_remaining': len(self.frontier_edges),
            'n_revealed_edge_feats': len(self.revealed_edge_keys),
            'budget_used': self._budget_used,
            'reveal_ratio': round(self.get_revealed_ratio(), 4),
            'reveal_history': self.reveal_history.copy(),
            'has_edge_feat': self.has_edge_feat,
            'edge_feat_dim': self.edge_feat_dim,
        }
