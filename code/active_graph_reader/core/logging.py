"""日志工具模块。"""

import logging
import sys


def get_logger(name: str = 'active_graph_reader') -> logging.Logger:
    """获取或创建模块级 logger。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-7s %(message)s',
            datefmt='%H:%M:%S',
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


def set_verbosity(verbose: bool) -> None:
    """动态调整日志级别。"""
    logger = get_logger()
    level = logging.DEBUG if verbose else logging.WARNING
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
