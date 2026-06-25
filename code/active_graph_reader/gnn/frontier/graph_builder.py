"""FrontierGraphBuilder：从 GraphState 构建扩展子图。

扩展子图 = V_revealed ∪ frontier_nodes + revealed↔revealed 边（双向）+ revealed↔frontier 边（双向）
与 encoder/subgraph_builder.py 的 SubgraphBuilder 模式一致。
"""

from dataclasses import dataclass
from typing import List, Tuple

import dgl
import torch

from ...engine.graph_state import GraphState


@dataclass
class FrontierSubgraph:
    """扩展子图：编码器输入。

    Attributes
    ----------
    g : dgl.DGLGraph
        构建好的扩展子图，edata['feat'] 包含边特征。
        前 n_revealed 个节点为 revealed，其余为 frontier。
    n_revealed : int
        revealed 节点数（用于分配 node type embedding）。
    frontier_original_ids : List[int]
        frontier 节点的原图 ID（升序），与 g 中后部节点一一对应。
    """
    g: dgl.DGLGraph
    n_revealed: int
    frontier_original_ids: List[int]


class FrontierGraphBuilder:
    """将 GraphState 转换为扩展子图（FrontierNodesEncoder 的输入）。

    构建内容：
    - 节点：V_revealed ∪ frontier_nodes（升序排列）
    - 边：revealed↔revealed（双向）+ revealed↔frontier（双向）
    - 边特征：复制自原图 edge_feat，存入 g.edata['feat']，反向边复用特征
    - 无节点特征（由 encoder 分配可训练 type embedding）
    """

    def build(self, state: GraphState) -> FrontierSubgraph:
        """从 GraphState 构建单个扩展子图。"""
        revealed = state.get_revealed_sorted()
        frontier = state.get_frontier_nodes()

        n_revealed = len(revealed)
        n_total = n_revealed + len(frontier)

        if n_total == 0:
            g = dgl.graph(([], []), num_nodes=0)
            return FrontierSubgraph(g=g, n_revealed=0, frontier_original_ids=[])

        # 向量化 old→new ID 映射
        max_old = revealed[-1]
        if frontier:
            max_old = max(max_old, frontier[-1])
        old_to_new = torch.full((max_old + 1,), -1, dtype=torch.long)
        old_to_new[revealed] = torch.arange(n_revealed)
        if frontier:
            old_to_new[frontier] = torch.arange(n_revealed, n_total)

        src_parts = []
        dst_parts = []
        feat_parts: List[torch.Tensor] = []

        # revealed ↔ revealed 边（双向）
        if state.revealed_edge_keys:
            rk_list = list(state.revealed_edge_keys)
            rk_t = torch.tensor(rk_list, dtype=torch.long)
            edge_ids_r = [state._edge_key_to_id[k] for k in rk_list]
            feat_r = state.edge_feat[edge_ids_r]

            u_r = old_to_new[rk_t[:, 0]]
            v_r = old_to_new[rk_t[:, 1]]

            src_parts.append(u_r)
            src_parts.append(v_r)  # reverse
            dst_parts.append(v_r)
            dst_parts.append(u_r)  # reverse
            feat_parts.append(feat_r)
            feat_parts.append(feat_r)  # repeat for reverse

        # revealed ↔ frontier 边（双向）
        if state.frontier_edges:
            fr_keys = list(state.frontier_edges)
            fr_ids = [state._edge_key_to_id[(min(u, v), max(u, v))] for u, v in fr_keys]
            feat_f = state.edge_feat[fr_ids]

            fr_t = torch.tensor(fr_keys, dtype=torch.long)
            u_f = old_to_new[fr_t[:, 0]]
            v_f = old_to_new[fr_t[:, 1]]

            src_parts.append(u_f)
            src_parts.append(v_f)  # reverse: frontier → revealed
            dst_parts.append(v_f)
            dst_parts.append(u_f)  # reverse
            feat_parts.append(feat_f)
            feat_parts.append(feat_f)  # repeat for reverse

        # 构建 DGL 图
        if len(src_parts) == 0:
            g = dgl.graph(([], []), num_nodes=n_total)
            edge_feat = torch.empty(0, state.edge_feat_dim)
        else:
            src = torch.cat(src_parts, dim=0)
            dst = torch.cat(dst_parts, dim=0)
            edge_feat = torch.cat(feat_parts, dim=0)
            g = dgl.graph((src, dst), num_nodes=n_total)

        g.edata['feat'] = edge_feat
        return FrontierSubgraph(g=g, n_revealed=n_revealed, frontier_original_ids=frontier)

    def build_batch(self, states: List[GraphState]) -> List[FrontierSubgraph]:
        """批量构建扩展子图。"""
        return [self.build(state) for state in states]
