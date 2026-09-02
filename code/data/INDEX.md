# data/ 目录结构

轨迹数据与分析产物，按数据集一级目录：

```
data/{dataset}/
├── trajectories/                     # generate_trajectories.py 原始输出
├── trajectories_filtered/            # filter_trajectories.py 过滤后
├── trajectories_augmented/           # augment_trajectories.py 合并中间态
├── trajectories_balanced/            # balance_trajectories.py 平衡输出
├── trajectories_stage{5,6,7}/        # （PPA 历史）m4 时代九段流水线产物
└── frontier_analysis/                # 前沿规模分析（diagnosis/frontier_analysis.py）
```

## PPA 迁移映射（2026-09-02，自旧 m4/data、m6 目录 move 而来）

| 旧位置 | 新位置 |
|---|---|
| `m4/data/raw/` | `data/ppa/raw/` |
| `m4/data/reader_trajectories/` | `data/ppa/reader_trajectories/` |
| `m4/data/stage5_filtered/` | `data/ppa/stage5_filtered/` |
| `m4/data/stage6_merged/` | `data/ppa/stage6_merged/` |
| `m4/data/stage7_filtered/` | `data/ppa/stage7_filtered/` |
| `m6/frontier_analysis/` | `data/ppa/frontier_analysis/` |

## CIFAR10 迁移映射（自旧 m7/data move 而来）

| 旧位置 | 新位置 |
|---|---|
| `m7/data/trajectories/` | `data/cifar10/trajectories/` |
| `m7/data/trajectories_filtered/` | `data/cifar10/trajectories_filtered/` |
| `m7/data/rich_balanced_trajectories_filtered/` | `data/cifar10/rich_balanced_trajectories_filtered/` |
| `m7/data/frontier_analysis/` | `data/cifar10/frontier_analysis/` |

轨迹文件为 torch pickle（`{'trajectories': {teacher: [[Trajectory...]]}, 'config'}`）。
旧文件引用的 `m4.teachers.base.Trajectory` 类已迁至 `trajectories/core/trajectory.py`，
读取一律走 `tools/trajectory_io.load_trajectory_file()`（自动重映射 pickle 路径）。
