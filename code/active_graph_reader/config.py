"""
配置管理模块
============

提供 RevealConfig dataclass，统一管理揭示协议的所有可配置参数。
支持从命令行参数构造，也可直接实例化用于编程调用。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RevealConfig:
    """揭示协议配置。

    Attributes
    ----------
    strategy : str
        前沿边揭示策略名称。可选: 'random', 'bfs', 'max-frontier-degree'。
    cost_model : str
        成本模型名称。v0 固定为 'unit'。M4+ 可扩展为 'atomic' 等。
    budget : int or None
        允许揭示的最大节点数（不含种子）。None 表示揭示所有可达节点。
    max_ratio : float or None
        允许揭示的最大节点比例 (0~1)。与 budget 同时设置时取较小上限。
    seed_node : int or None
        固定的种子节点 ID。None 表示随机选取。
    graph_index : int or None
        OGBG-PPA 数据集中的图索引。None 表示使用训练集第一张图。
    verbose : bool
        是否输出详细日志。默认 True。
    k_seeds : int
        种子节点数量。v0 协议固定为 1。
    """

    strategy: str = 'random'
    cost_model: str = 'unit'
    budget: Optional[int] = None
    max_ratio: Optional[float] = None
    seed_node: Optional[int] = None
    graph_index: Optional[int] = None
    verbose: bool = True
    k_seeds: int = 1

    # ---- 预留的 M4+ 扩展字段 ------------------------------------------------
    # 原子成本参数（v0 不使用，M4+ 激活）
    c_n: float = 1.0       # reveal_node 成本（节点存在性）
    c_e: float = 1.0       # reveal_edge 成本（边存在性）
    c_nf: float = 2.0      # reveal_node_f 成本（节点+特征）
    c_ef: float = 2.0      # reveal_edge_f 成本（边+特征）

    def __post_init__(self):
        """验证配置参数合法性。"""
        # 策略名称
        valid_strategies = {'random', 'bfs', 'max-frontier-degree'}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"无效策略 '{self.strategy}'，可选: {sorted(valid_strategies)}")

        # 成本模型名称
        valid_cost_models = {'unit', 'atomic'}
        if self.cost_model not in valid_cost_models:
            raise ValueError(
                f"无效成本模型 '{self.cost_model}'，"
                f"可选: {sorted(valid_cost_models)}")

        # max_ratio 范围
        if self.max_ratio is not None and not (0 < self.max_ratio <= 1):
            raise ValueError(
                f"max_ratio 必须在 (0, 1] 范围内，实际: {self.max_ratio}")

        # budget 非负
        if self.budget is not None and self.budget < 0:
            raise ValueError(f"budget 不能为负数，实际: {self.budget}")

        # k_seeds
        if self.k_seeds < 1:
            raise ValueError(f"k_seeds 至少为 1，实际: {self.k_seeds}")

    @classmethod
    def from_args(cls, args) -> 'RevealConfig':
        """从 argparse.Namespace 构造配置。"""
        return cls(
            strategy=args.strategy,
            cost_model=getattr(args, 'cost_model', 'unit'),
            budget=args.budget,
            max_ratio=args.max_ratio,
            seed_node=args.seed,
            graph_index=args.graph_index,
            verbose=getattr(args, 'verbose', True),
        )
