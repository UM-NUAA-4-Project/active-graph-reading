"""
成本模型注册表
==============

提供成本模型的注册、查找和列举功能。支持通过名称字符串动态获取模型实例，
实现成本模型的可插拔切换。

使用方式：
    from active_graph_reader.cost.registry import get_cost_model

    model = get_cost_model('unit')       # → UnitCostModel 实例
    model = get_cost_model('atomic')     # → AtomicCostModel 实例 (M4+)
"""

from typing import Dict, Type

from .base import CostModel
from .unit_cost import UnitCostModel

# ---- 内置成本模型注册 -------------------------------------------------------
# 新增成本模型只需在此处注册，或在外部调用 register_cost_model。
_BUILTIN_COST_MODELS: Dict[str, Type[CostModel]] = {
    'unit': UnitCostModel,
}

# 运行时注册表（叠加上用户外部注册）
_registry: Dict[str, Type[CostModel]] = dict(_BUILTIN_COST_MODELS)


def register_cost_model(name: str, model_cls: Type[CostModel]) -> None:
    """注册一个新的成本模型类。

    Parameters
    ----------
    name : str
        成本模型名称（用于配置和命令行指定）。
    model_cls : Type[CostModel]
        成本模型类（须为 CostModel 的子类）。

    Raises
    ------
    TypeError
        如果 model_cls 不是 CostModel 的子类。
    """
    if not issubclass(model_cls, CostModel):
        raise TypeError(
            f"'{model_cls.__name__}' 必须是 CostModel 的子类")
    _registry[name] = model_cls


def get_cost_model(name: str) -> CostModel:
    """根据名称获取成本模型实例。

    Parameters
    ----------
    name : str
        成本模型名称。

    Returns
    -------
    CostModel
        成本模型实例。

    Raises
    ------
    ValueError
        如果指定的成本模型名称未注册。
    """
    if name not in _registry:
        available = sorted(_registry.keys())
        raise ValueError(
            f"未知成本模型 '{name}'，可用: {available}")
    return _registry[name]()


def get_cost_model_class(name: str) -> Type[CostModel]:
    """根据名称获取成本模型类（不实例化）。"""
    if name not in _registry:
        available = sorted(_registry.keys())
        raise ValueError(
            f"未知成本模型 '{name}'，可用: {available}")
    return _registry[name]


def list_cost_models() -> Dict[str, Type[CostModel]]:
    """列举所有已注册的成本模型。

    Returns
    -------
    Dict[str, Type[CostModel]]
    """
    return dict(_registry)
