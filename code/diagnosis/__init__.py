"""诊断工具包：checkpoint 诊断、轨迹多样性分析、难度过滤、可视化、
前沿规模分析、oracle 上界估计。

自 m4/m5/m6/m7 各 diagnosis 脚本泛化迁移：数据集经 datasets 注册表
（--dataset/--root 任意切换），reader 经 train_reader.core 从 checkpoint
自动重建（base/attention 皆可），种子公式统一走 tools.seeds。
"""
