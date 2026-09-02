"""轨迹管线包：教师生成 → 过滤 → 平衡/增强 → 统计分析。

目录结构：
- core/       库（Trajectory、TrajectoryGenerator、filtering、rollout 工具）
- teachers/   教师（gradient / info-gain / structural / hybrid）
- 顶层 CLI：generate_trajectories.py / filter_trajectories.py /
  balance_trajectories.py / augment_trajectories.py / analyze_trajectories.py
"""
