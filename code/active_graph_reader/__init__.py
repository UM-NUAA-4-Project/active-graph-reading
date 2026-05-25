"""
Active Graph Reader — M3 v0 Base 协议实现
===========================================

基于 myworkplace/docs/M3/base-protocol-v0.md 定义的 v0 协议，
构建模块化、可扩展的主动图阅读框架。

模块组成：
- config      —— 配置管理（dataclass）
- dataset     —— 数据加载与预处理（OGBG-PPA）
- engine      —— 核心引擎（图状态、揭示动作、揭示循环）
- strategies  —— 可插拔前沿边揭示策略（random / bfs / max-frontier-degree）
- cost        —— 成本模型抽象（v0 单位成本 + M4+ 预留扩展点）
- utils       —— 工具函数（日志等）
"""
