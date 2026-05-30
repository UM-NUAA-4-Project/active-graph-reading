"""Cost model: unit cost (v0). Every reveal_node costs 1.0."""

from typing import Dict, Type


class UnitCostModel:
    """Unit cost: every reveal_node(v) costs 1.0. M4+ can extend with composite models."""

    @property
    def name(self) -> str:
        return 'unit'

    def compute(self, node, graph_state) -> float:
        return 1.0

    def reset(self):
        pass


_registry: Dict[str, Type] = {'unit': UnitCostModel}


def register_cost_model(name: str, cls: Type) -> None:
    _registry[name] = cls


def get_cost_model(name: str) -> UnitCostModel:
    if name not in _registry:
        raise ValueError(f"Unknown cost model '{name}', available: {sorted(_registry)}")
    return _registry[name]()


def list_cost_models() -> Dict[str, Type]:
    return dict(_registry)
