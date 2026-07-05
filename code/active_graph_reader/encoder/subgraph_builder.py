"""
子图构造器
==========

将揭示过程结束后的 GraphState 转换为标准的 DGLGraph，
供下游 GNN 编码器消费。

约定：
- 节点范围：仅 V_revealed 中的节点。
- 边范围：仅双方均已揭示的边（key 在 revealed_edge_feats 中）。
- 前沿边不加入子图——它们的未揭示端点不在 V_revealed 中。
- 原图节点 ID 到子图连续 ID 的映射通过 old_to_new 字典维护，
  供编码器查找 nn.Embedding 时使用。
"""

import logging
from typing import Dict, List, Set, Tuple
from ..engine.graph_state import GraphState
import dgl
import torch

logger = logging.getLogger(__name__)


class SubgraphBuilder:
    """将 GraphState 转换为 DGLGraph。

    使用方式：
        builder = SubgraphBuilder()
        sub_g, original_ids = builder.build(state)
        # sub_g.ndata 为空（节点特征由编码器的 node_emb 注入）
        # sub_g.edata['feat']  shape = (n_edges, edge_feat_dim)
        # original_ids 是 V_revealed 按升序排列的列表
    """

    def build(
        self,
        state: 'GraphState',
    ) -> Tuple[dgl.DGLGraph, List[int]]:
        """构造子图。

        Parameters
        ----------
        state : GraphState
            揭示循环结束后的图状态。

        Returns
        -------
        sub_g : dgl.DGLGraph
            仅含 V_revealed 节点和已揭示边的子图。
        original_ids : List[int]
            子图中节点对应的原始图节点 ID（升序排列），
            original_ids[new_id] = old_id。
        """
        # ---- 1. 节点映射 ---------------------------------------------------
        original_ids = sorted(state.V_revealed)
        old_to_new: Dict[int, int] = {
            old: i for i, old in enumerate(original_ids)
        }

        # ---- 2. 收集已揭示边 ------------------------------------------------
        # revealed_edge_feats 的 key 是 (min(u,v), max(u,v))，
        # 每条边构建两个方向（无向图 → GNN 消息双向传递）。
        src_list: List[int] = []
        dst_list: List[int] = []
        edge_feat_list: List[torch.Tensor] = []

        for (u, v), feat in state.revealed_edge_feats.items():
            if u in old_to_new and v in old_to_new:
                # u → v
                src_list.append(old_to_new[u])
                dst_list.append(old_to_new[v])
                edge_feat_list.append(feat)

                # v → u
                src_list.append(old_to_new[v])
                dst_list.append(old_to_new[u])
                edge_feat_list.append(feat)

        # ---- 3. 构造 DGLGraph -----------------------------------------------
        n_nodes = len(original_ids)
        if n_nodes == 0:
            raise ValueError("V_revealed 为空，无法构造子图")

        if len(src_list) == 0:
            # 无边情况：创建一个空边图
            sub_g = dgl.graph(([], []), num_nodes=n_nodes)
        else:
            src_t = torch.tensor(src_list, dtype=torch.long)
            dst_t = torch.tensor(dst_list, dtype=torch.long)
            sub_g = dgl.graph((src_t, dst_t), num_nodes=n_nodes)

        # ---- 4. 附加边特征 ---------------------------------------------------
        if edge_feat_list:
            sub_g.edata['feat'] = torch.stack(edge_feat_list, dim=0)
        else:
            # 无已揭示边→无 edata（下游编码器需处理此情况）
            logger.debug("子图无边特征 (%d 节点, 0 已揭示边)", n_nodes)

        logger.debug(
            "SubgraphBuilder: %d 节点, %d 边, edge_feat_dim=%d",
            n_nodes,
            sub_g.num_edges(),
            edge_feat_list[0].shape[0] if edge_feat_list else 0,
        )

        return sub_g, original_ids
