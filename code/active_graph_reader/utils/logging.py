"""
日志工具模块
============

提供统一的日志配置和格式化工具，支持 verbose 开关。
"""

import logging
import sys
from typing import Optional

# 模块级 logger
_logger: Optional[logging.Logger] = None


def get_logger(name: str = 'active_graph_reader',
               verbose: bool = True) -> logging.Logger:
    """获取或创建模块级 logger。

    Parameters
    ----------
    name : str
        Logger 名称。
    verbose : bool
        是否启用 INFO 级别日志。若为 False，仅输出 WARNING 及以上。

    Returns
    -------
    logging.Logger
    """
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    # 避免重复添加 handler
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-7s %(message)s',
            datefmt='%H:%M:%S',
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    # 防止日志传播到根 logger
    _logger.propagate = False

    return _logger


def set_verbosity(verbose: bool) -> None:
    """动态调整日志级别。

    Parameters
    ----------
    verbose : bool
        True 启用详细日志，False 仅输出警告及以上。
    """
    logger = get_logger()
    level = logging.DEBUG if verbose else logging.WARNING
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
