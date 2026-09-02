"""Trajectory 数据结构与 Teacher 抽象（实现在 teachers/base.py，此处再导出）。

保持 m4 原布局：定义在 teachers/base.py；本模块为兼容别名。
（定义放 teachers 侧是为了避免 teachers.base → core 包初始化 →
core.generator → teachers.gradient → teachers.base 的循环导入。）
"""

from ..teachers.base import Teacher, Trajectory, LEGACY_PICKLE_MODULE  # noqa: F401
