"""核心引擎模块。

包含三个子模块：
- graph_state   —— 图状态管理（已揭示节点、前沿、边特征）
- reveal_action —— reveal_node(v) 核心动作实现
- reveal_loop   —— 揭示循环（策略选择 + 动作执行 + 状态更新）
"""

from .graph_state import GraphState
from .reveal_action import reveal_node
from .reveal_loop import RevealLoop

__all__ = [
    'GraphState',
    'reveal_node',
    'RevealLoop',
]
