"""baseline 策略注册：把协议适配基线接入 fixed strategy 注册表。

sagpool 需要加载 scorer checkpoint（带 device），由 eval.py 特殊处理构造；
random-walk 无参数，注册后即可被 create_fixed_reader('random-walk') 构造。
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _REPO_ROOT / 'code'
for _p in (_CODE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def register_baseline_strategies() -> None:
    """注册 random-walk 到 fixed strategy 注册表（幂等）。"""
    from active_graph_reader.reader.fixed import register_strategy
    from baseline.randwalk import RandomWalkStrategy
    register_strategy('random-walk', RandomWalkStrategy)
