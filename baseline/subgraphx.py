"""SubgraphX 基线（AGR 协议适配版，planner 型）。

复现自 SubgraphX 官方实现（用户本地 SubgraphX-master，DIG 风格）：
MCTS 搜索连通子图（Q+UCB 选择、state_map 去重、逐 expansion 批量评估
子节点奖励）+ 冻结分类器打分。默认超参对齐 Configures.py
（n_rollout=20 / c_puct=5 / expand_atoms=12）。

与官方实现的三处协议适配（完整论证见 protocol-adaptation.md §5）：

1. **搜索方向反转 + 种子根约束**：官方从完整图自顶向下删点（explainer
   设定）；本实现以 seed 为根自底向上生长，子节点 = 当前 coalition 的
   frontier 节点（生长方向天然保持连通，无需 connected_components 剪枝），
   终止条件 = coalition 大小到达预算目标（budget+1 含 seed）。
2. **无标签奖励**：官方 value_func = 分类器对**已知目标类**的概率
   （目标类来自完整图预测，隐含全图信息）；改为类别无关
   reward_mode='prob'（max_c softmax(logit_c)，默认）/'logit'/'margin'。
3. **奖励弃用 Shapley**：官方 mc_l_shapley 的 zero_filling 构图保留完整图
   边结构作为上下文——评测态不可见（P4 违规）。改用官方 reward 家族中的
   gnn_score 成员：对候选 coalition 的**导出子图**（双向边 + 自环归一，
   与 SubgraphBuilder 终判构图同约定）整体打分。

定位声明：MCTS 规划期使用完整图结构（含未揭示节点的身份/特征）——这是
**planner 型基线**（与 teacher 同级的 oracle 口径），衡量"学习到的序贯
策略 vs 直接对同一分类器做测试时搜索"的增益；config 中以 planner 标记，
论文表格需注明。算力上限 max_evals（单次规划的唯一 coalition 评估数）
与 n_rollout/expand_atoms 一样作为可报告超参。
"""

import math
import sys
from collections import deque
from pathlib import Path

import torch
import dgl

_CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from active_graph_reader.core.base_reader import BaseReader  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402


class _TNode:
    """MCTS 树节点：coalition（排序 tuple）+ P/W/N 统计（与官方 MCTSNode 同构）。"""

    __slots__ = ('coalition', 'parent', 'children', 'W', 'N', 'P', 'c_puct')

    def __init__(self, coalition, parent=None, c_puct=5.0):
        self.coalition = coalition          # 排序 tuple
        self.parent = parent
        self.children = []
        self.W = 0.0                        # 累计奖励
        self.N = 0                          # 访问次数
        self.P = 0.0                        # 奖励（创建时评估一次）
        self.c_puct = c_puct

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, total_n: int) -> float:
        return self.c_puct * self.P * math.sqrt(total_n) / (1 + self.N)


class SubgraphXStrategy(BaseReader):
    """SubgraphX MCTS 规划 → 预算内最优连通揭示序（BFS 序揭示）。"""

    def __init__(self, n_rollout: int = 20, c_puct: float = 5.0,
                 expand_atoms: int = 12, reward_mode: str = 'prob',
                 max_evals: int = 2048):
        super().__init__()
        self.n_rollout = n_rollout
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.reward_mode = reward_mode
        self.max_evals = max_evals
        self._classifier = None
        self._device = None
        self._plan: list = []       # 当前 rollout 的揭示序（不含 seed）
        self._plan_idx = 0
        self._warned_fallback = False

    @property
    def name(self) -> str:
        return 'subgraphx'

    def configure(self, classifier, device) -> None:
        """注入冻结上界分类器（奖励 oracle）与设备；由 eval.py 调用。"""
        self._classifier = classifier
        self._device = torch.device(device)

    def reset(self) -> None:
        self._plan = []
        self._plan_idx = 0

    # ---------- 奖励评估（planner/oracle：真实特征 + 完整图结构） ----------

    def _edge_index(self, state: GraphState):
        """canonical 边键 + 原 edge id 列表（一次预计算，整个 rollout 复用）。"""
        if state.has_edge_feat:
            return list(state._edge_key_to_id.items())
        keys, eids = [], []
        src, dst = state.g.edges()
        seen = set()
        for u, v in zip(src.tolist(), dst.tolist()):
            key = (min(u, v), max(u, v))
            if key not in seen and u != v:
                seen.add(key)
                keys.append((key, len(keys)))
        return keys

    def _coalition_graphs(self, state: GraphState, coalitions: list) -> list:
        """coalition → 导出子图（双向边 + 自环归一 + 真实节点特征）。"""
        edge_index = self._edge_index(state)
        graphs = []
        for coalition in coalitions:
            cs = set(coalition)
            ids = sorted(cs)
            old_to_new = {old: i for i, old in enumerate(ids)}
            src, dst, eids = [], [], []
            for key, eid in edge_index:
                u, v = key
                if u in cs and v in cs:
                    nu, nv = old_to_new[u], old_to_new[v]
                    src += [nu, nv]
                    dst += [nv, nu]
                    eids += [eid, eid]
            g = dgl.graph((torch.tensor(src, dtype=torch.long),
                           torch.tensor(dst, dtype=torch.long)),
                          num_nodes=len(ids))
            if state.has_node_feat:
                g.ndata['feat'] = state.node_feat[ids]
            if state.has_edge_feat:
                g.edata['feat'] = state.edge_feat[eids]
            graphs.append(dgl.add_self_loop(dgl.remove_self_loop(g)))
        return graphs

    def _rewards(self, state: GraphState, coalitions: list) -> list:
        """批量奖励：classifier 对导出子图打分，类别无关聚合。"""
        graphs = self._coalition_graphs(state, coalitions)
        batch_g = dgl.batch(graphs).to(self._device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long,
                           device=self._device)
        with torch.no_grad():
            logits = self._classifier(batch_g, nids)
        if self.reward_mode == 'prob':
            scores = torch.softmax(logits, dim=-1).max(dim=-1).values
        elif self.reward_mode == 'logit':
            scores = logits.max(dim=-1).values
        elif self.reward_mode == 'margin':
            top2 = torch.softmax(logits, dim=-1).topk(2, dim=-1).values
            scores = top2[:, 0] - top2[:, 1]
        else:
            raise ValueError(f"未知 reward_mode '{self.reward_mode}'")
        return [float(s) for s in scores.cpu()]

    # ---------- MCTS（生长方向；与官方 mcts_rollout 同构） ----------

    def _adjacency(self, state: GraphState) -> dict:
        adj = {}
        src, dst = state.g.edges()
        for u, v in zip(src.tolist(), dst.tolist()):
            if u == v:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return {u: sorted(vs) for u, vs in adj.items()}

    def _expand(self, node: _TNode, adj: dict, state: GraphState,
                state_map: dict, evals: int) -> int:
        """扩展子节点 = coalition ∪ {frontier 节点}；返回更新后的评估计数。"""
        cs = set(node.coalition)
        frontier = sorted({v for u in node.coalition for v in adj.get(u, ())
                           if v not in cs})
        if not frontier:
            return evals
        # 官方按度数排序取 expand_atoms 个子节点；生长方向按 coalition 内
        # 连接度（即 frontier_degree 语义）降序，平分取小 id
        frontier.sort(key=lambda v: (-sum(1 for u in node.coalition
                                          if v in adj.get(u, ())), v))
        children = []
        for f in frontier[:self.expand_atoms]:
            coalition = tuple(sorted(cs | {f}))
            existing = state_map.get(coalition)
            if existing is not None:
                if all(existing is not c for c in node.children):
                    node.children.append(existing)
            else:
                children.append(coalition)
        if children:
            if evals + len(children) > self.max_evals:
                return evals          # 算力上限：该节点视作叶子
            rewards = self._rewards(state, children)
            evals += len(children)
            for coalition, p in zip(children, rewards):
                child = _TNode(coalition, parent=node, c_puct=self.c_puct)
                child.P = p
                state_map[coalition] = child
                node.children.append(child)
        else:
            # 全部子节点已在 state_map：仍需挂接为 children 以便 UCB 选择
            for f in frontier[:self.expand_atoms]:
                coalition = tuple(sorted(cs | {f}))
                existing = state_map.get(coalition)
                if existing is not None and all(existing is not c
                                                for c in node.children):
                    node.children.append(existing)
        return evals

    def _mcts_plan(self, state: GraphState) -> list:
        """搜索最优 budget+1 连通 coalition；返回 BFS 揭示序（不含 seed）。"""
        target = (state._budget_limit or 0) + 1
        adj = self._adjacency(state)
        root = _TNode((state.seed_node,), c_puct=self.c_puct)
        state_map = {root.coalition: root}
        evals = 0
        for _ in range(self.n_rollout):
            node, path = root, [root]
            while len(node.coalition) < target:
                if not node.children:
                    evals = self._expand(node, adj, state, state_map, evals)
                if not node.children:
                    break                       # frontier 耗尽 / 算力上限
                total_n = sum(c.N for c in node.children)
                node = max(node.children,
                           key=lambda c: (c.Q() + c.U(total_n),
                                          tuple(-x for x in c.coalition)))
                path.append(node)
            v = node.P
            for n in path:
                n.W += v
                n.N += 1

        leaves = [n for n in state_map.values() if len(n.coalition) == target]
        if leaves:
            pool = leaves
        else:
            # 目标大小不可达（分量小 / 算力上限）：在最深节点中选最优，
            # 计划尽量填满可达分量
            max_size = max(len(n.coalition) for n in state_map.values())
            pool = [n for n in state_map.values()
                    if len(n.coalition) == max_size and n is not root]
        if not pool:
            return []
        best = max(pool, key=lambda n: (n.P, len(n.coalition), n.coalition))
        return self._bfs_order(adj, best.coalition, state.seed_node)

    @staticmethod
    def _bfs_order(adj: dict, coalition, seed: int) -> list:
        """coalition 内自 seed 的 BFS 序（每步节点必然在前置揭示节点的 frontier 上）。"""
        cs = set(coalition)
        order, seen = [], {seed}
        q = deque([seed])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v in cs and v not in seen:
                    seen.add(v)
                    order.append(v)
                    q.append(v)
        if len(order) < len(cs) - 1:            # 理论不可达（coalition 连通由构造保证）
            order += sorted(cs - seen - {seed})
        return order

    # ---------- BaseReader 接口 ----------

    def select(self, graph_state: GraphState):
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._classifier is None:
            raise RuntimeError("subgraphx 策略需先 configure(classifier, device)"
                               "（eval.py --strategy subgraphx 自动完成）")
        if self._plan_idx >= len(self._plan):
            self._plan = self._mcts_plan(graph_state)
            self._plan_idx = 0

        while self._plan_idx < len(self._plan):
            v = self._plan[self._plan_idx]
            if v not in graph_state.V_revealed:
                if any(edge[1] == v for edge in frontier):
                    self._plan_idx += 1
                    return v
                self._plan_idx += 1      # 计划节点不在当前 frontier：跳过
                continue
            self._plan_idx += 1
        # 计划耗尽（预算未满、frontier 仍有）：退化为 max-frontier-degree
        if not self._warned_fallback:
            print("  [subgraphx] 计划耗尽，退化 max-frontier-degree 选点")
            self._warned_fallback = True
        best, best_deg = None, -1
        for edge in frontier:
            deg = graph_state.get_frontier_degree(edge[1])
            if deg > best_deg:
                best_deg, best = deg, edge[1]
        return best
