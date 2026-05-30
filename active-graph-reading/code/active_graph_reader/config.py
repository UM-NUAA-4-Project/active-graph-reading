"""配置管理模块：统一管理揭示协议的所有可配置参数。"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RevealConfig:
    """揭示协议配置。

    Attributes
    ----------
    strategy : str
        前沿边揭示策略名称: 'random', 'bfs', 'max-frontier-degree'.
    budget : int or None
        允许揭示的最大节点数（不含种子）。None 表示无限制。
    max_ratio : float or None
        允许揭示的最大节点比例 (0~1)。
    seed_node : int or None
        固定种子节点 ID。None 表示随机选取。
    verbose : bool
        是否输出详细日志。
    """

    strategy: str = 'random'
    cost_model: str = 'unit'
    budget: Optional[int] = None
    max_ratio: Optional[float] = None
    seed_node: Optional[int] = None
    graph_index: Optional[int] = None
    verbose: bool = True
    k_seeds: int = 1

    def __post_init__(self):
        valid_strategies = {'random', 'bfs', 'max-frontier-degree', 'learnable'}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"无效策略 '{self.strategy}'，可选: {sorted(valid_strategies)}")

        if self.max_ratio is not None and not (0 < self.max_ratio <= 1):
            raise ValueError(
                f"max_ratio 必须在 (0, 1] 范围内，实际: {self.max_ratio}")

        if self.budget is not None and self.budget < 0:
            raise ValueError(f"budget 不能为负数，实际: {self.budget}")

        if self.k_seeds < 1:
            raise ValueError(f"k_seeds 至少为 1，实际: {self.k_seeds}")
