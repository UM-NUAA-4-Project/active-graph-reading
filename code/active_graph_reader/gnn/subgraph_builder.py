"""最终子图构造器：GraphState → DGLGraph（仅 V_revealed + revealed edges，供分类器推理）。

区别于 FrontierNodesEncoder 内部构建的"扩展子图"（V_revealed ∪ frontier_nodes
+ frontier edges），本模块构建的"最终子图"仅含已揭示节点和已揭示边，且边为双向
以支持 GNN 消息传递。
"""

import logging
from typing import Dict, List, Tuple
from ..engine.graph_state import GraphState
import dgl
import torch

logger = logging.getLogger(__name__)


class SubgraphBuilder:
    """将揭示完成后的 GraphState 转换为分类器可消费的 DGLGraph。

    - 节点：仅 V_revealed
    - 边：仅双方均已揭示的边（双向）
    - 边特征：复制自原图
    - 节点特征：若原图有节点特征，则复制 V_revealed 范围的节点特征
    """

    def build(
        self,
        state: GraphState,
    ) -> Tuple[dgl.DGLGraph, List[int]]:
        """构造最终子图（分类器推理用）。

        Returns
        -------
        sub_g : dgl.DGLGraph
        original_ids : List[int]  子图新 ID → 原图节点 ID 映射（升序）
        """
        original_ids = sorted(state.V_revealed)
        n_nodes = len(original_ids)
        if n_nodes == 0:
            raise ValueError("V_revealed 为空，无法构造子图")

        # 节点特征（适配有节点特征的数据集）
        if state.has_node_feat and state.node_feat is not None:
            node_feat_t = state.node_feat[original_ids]
        else:
            node_feat_t = None

        n_edges = len(state.revealed_edge_keys)
        if n_edges == 0:
            sub_g = dgl.graph(([], []), num_nodes=n_nodes)
            if node_feat_t is not None:
                sub_g.ndata['feat'] = node_feat_t
            return sub_g, original_ids

        # 向量化边收集
        edge_keys = list(state.revealed_edge_keys)
        edge_ids = [state._edge_key_to_id[k] for k in edge_keys]
        edge_feats_t = state.edge_feat[edge_ids]

        device = edge_feats_t.device
        edge_pairs = torch.tensor(edge_keys, dtype=torch.long, device=device)

        # old_to_new 映射
        max_old_id = max(original_ids)
        old_to_new = torch.full((max_old_id + 1,), -1, dtype=torch.long, device=device)
        old_to_new[original_ids] = torch.arange(n_nodes, device=device)

        u_new = old_to_new[edge_pairs[:, 0]]
        v_new = old_to_new[edge_pairs[:, 1]]

        K = u_new.shape[0]
        src = torch.empty(2 * K, dtype=torch.long, device=device)
        dst = torch.empty(2 * K, dtype=torch.long, device=device)
        src[0::2] = u_new
        src[1::2] = v_new
        dst[0::2] = v_new
        dst[1::2] = u_new
        edge_feats_bidir = edge_feats_t.repeat_interleave(2, dim=0)

        sub_g = dgl.graph((src, dst), num_nodes=n_nodes)
        sub_g.edata['feat'] = edge_feats_bidir
        if node_feat_t is not None:
            sub_g.ndata['feat'] = node_feat_t

        sub_g = dgl.add_self_loop(sub_g)
        return sub_g, original_ids

    def build_batch(
        self,
        states: List[GraphState],
    ) -> Tuple[dgl.DGLGraph, List[List[int]]]:
        """批量构造最终子图并 dgl.batch 合并。"""
        sub_graphs = []
        all_original_ids = []
        for state in states:
            sub_g, original_ids = self.build(state)
            sub_graphs.append(sub_g)
            all_original_ids.append(original_ids)
        batched_g = dgl.batch(sub_graphs)
        return batched_g, all_original_ids
