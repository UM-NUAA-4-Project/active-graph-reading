# Active Graph Reader — 预算限制下的图阅读策略研究

## 项目概述

研究在有限预算（可揭示节点数）下，如何通过自适应策略逐步阅读图结构以达到最优分类性能。

**核心问题**：给定预算 B，哪些节点最值得揭示？

**问题形式化**（v1 协议）：

```
State:  G_t = (V_revealed^(t), E_known^(t), X_E^(t), X_V^(t))
Action: a_t = reveal_node(v), v ∈ frontier_nodes
Transition: G_{t+1} = reveal_node(G_t, v)  — 确定性转移
Reward:  R = Accuracy(classifier(sub_G))  — episode-level
```

初始观测 G_0：随机 1 个 seed，知晓其邻居 ID + seed→frontier 边特征（不消耗 budget）。每步从前沿节点中选一个揭示，揭示后该节点的边特征变为已知、新邻居加入前沿。预算耗尽后基于子图分类。

**v1 核心变更**：frontier edge feature 在选择 frontier node 之前对 Reader 可见——这是 reader 的决策信号来源。

## 环境

- **WSL2**：Windows conda 通过 `/mnt/d/anoconda/` 访问
- **Conda 环境**：`gnn`（Python 3.10, PyTorch 2.3.0+cu121, DGL 2.2.1）
- **Shell 命令**：直接使用 `python`、`conda`
- **OGB 数据根目录**：`D:\ogb-master\dataset`（Python 是 Windows 版，需用 Windows 路径格式）

## 目录结构

```
active-graph-reading/
├── CLAUDE.md                           # 项目文档
├── code/
│   ├── active_graph_reader/            # 核心库
│   │   ├── engine/
│   │   │   ├── graph_state.py          #  G_t 部分可观测状态管理
│   │   │   ├── reveal_action.py        #  reveal_node(v) 状态转移
│   │   │   └── reveal_loop.py          #  主循环协调器
│   │   ├── strategies/
│   │   │   ├── random_strategy.py      #  P1: 均匀随机
│   │   │   ├── bfs_strategy.py         #  P2: BFS 层序
│   │   │   ├── max_frontier_degree_strategy.py  # P3: 最大前沿度
│   │   │   └── registry.py             #  策略注册（get/register/list）
│   │   ├── encoder/
│   │   │   ├── gin_encoder.py          #  GINEEncoder（对齐 OGB 官方）
│   │   │   ├── classifier.py           #  GraphClassifier = Encoder + Head
│   │   │   └── subgraph_builder.py     #  GraphState → DGLGraph（最终子图）
│   │   ├── reader/
│   │   │   ├── base_reader.py          #  BaseReader 抽象接口
│   │   │   ├── learnable_reader.py     #  LearnableReader（FrontierEncoder + MLP）
│   │   │   ├── frontier_encoder.py     #  FrontierNodesEncoder（扩展子图 GNN 编码）
│   │   │   └── facade.py               #  reveal_one / evaluate / create_dataloader
│   │   ├── cost/registry.py            #  成本模型（v0: unit cost = 1.0）
│   │   ├── dataloader/
│   │   │   ├── loader.py               #  PPADataset 封装
│   │   │   └── reader_dataset.py       #  ReaderDataset（IterableDataset）
│   │   └── config.py                   #  RevealConfig
│   ├── m3/                             # M3 基准评测脚本
│   │   ├── train_full_graph.py         #  全图 GIN 训练
│   │   ├── benchmark_m3.py             #  多 seed 评测 + 预算曲线
│   │   └── smoke_test_v2.py            #  冒烟测试
│   ├── m4/                             # M4 轨迹生成
│   │   ├── trajectory.py               #  Trajectory 数据结构
│   │   ├── teachers/                   #  Teacher 实现（T1-T4）
│   │   ├── trajectory_generator.py     #  批量生成 + 断点续传
│   │   ├── trajectory_scorer.py        #  4 维评分
│   │   ├── trajectory_filter.py        #  过滤 + Top-K
│   │   ├── oracle_estimator.py         #  Oracle 上界
│   │   ├── feasibility_analyzer.py     #  DFI + M5 路线推荐
│   │   └── generate_trajectories.py    #  主脚本
│   └── split-tools-analyze/            # 数据分割与训练工具
└── docs/                               # 设计文档
    ├── proposal.md                     #  项目 proposal
    ├── open-questions.md               #  待解决问题
    ├── progress/daily/                 #  每日进展
    └── progress/weekly/                #  周报
```

## 关键设计决策

### 1. v1 协议

- **k=1 随机 seed**，action = reveal_node(v)，unit cost，预算 = 揭示节点数（不含 seed）
- **Frontier edge feature 在 reveal 前可见**（v1 vs v0 的核心差异）：一端已揭示即可查询，不消耗 budget。这是 Reader 区分 frontier 候选节点的唯一特征通道。
- **可见 ≠ 揭示**：frontier 边特征可查询不消耗 budget，通过 `state.get_frontier_edge_feat(u, v)` 访问
- **确定性种子公式**：`seed_node = (graph_index * 31 + seed_i * 17 + 7) % n_nodes`

### 2. PPA 数据集特性

- 无节点特征：所有节点共享 `nn.Embedding(1, hidden_dim=300)`，差异来自拓扑位置
- 7 维边特征：frontier edge feature 是 reader 决策的关键输入
- 全图需加自环：`dgl.add_self_loop(g)` — 对齐 OGB 官方预处理
- 158,100 图，37 类，平均 |V|~300，|E|~2200（稠密图）

### 3. 子图构建的两种类型（重要区分）

| | 扩展子图（FrontierEncoder 用） | 最终子图（Classifier 用） |
|---|---|---|
| 节点 | V_revealed ∪ frontier_nodes | 仅 V_revealed |
| 边 | revealed↔revealed(双向) + revealed→frontier(单向) | revealed↔revealed(双向) |
| 建图时机 | 每 reveal step 一次（B 次） | reveal 结束后一次 |
| 节点特征 | type embedding (revealed=0, frontier=1) | 无（PPA）或原图 node feat |
| 边特征 | 原图边特征（区分 frontier 节点的关键） | 原图边特征 |
| 代码位置 | `reader/frontier_encoder.py` | `encoder/subgraph_builder.py` |

### 4. Learnable Reader 工作流

```
G_t → FrontierNodesEncoder（扩展子图 + GNN）→ frontier_embeds
    → ReaderMLP → scores → softmax → probs → argmax → next_node
```

训练时三种策略优化方法（优先级顺序）：
1. **Behavior Cloning**（首选）：M4 teacher 轨迹 → CrossEntropy(scores, a*_t) → 稳定预训练
2. **Gumbel-Softmax**（fine-tune）：STE 近似可微 → 端到端微调 Reader + Classifier
3. **REINFORCE**（backup）：无偏梯度估计 → BC+Gumbel 不收敛时的兜底方案

### 5. M4 Teacher 轨迹

四种 teacher 生成方式：
- **T1 (gradient)**：选使 classifier loss 下降最快的 frontier node
- **T2 (info-gain)**：选使 classifier entropy 下降最多的 frontier node
- **T3 (structural)**：选 degree/pagerank 最高的 frontier node
- **T4 (hybrid)**：α·T1 + (1-α)·T3

轨迹评价：CrossEntropy 越低越好；top-k 多样性维护通过 |V| 维 bool 数组异或。

### 6. 进入 M5 的前提条件

- M3 gate PASS：BFS 在 ≤10% budget 下未达到 full-graph accuracy 的 90% → 策略选择有空间
- M3 三种策略差异显著（需扩大样本确认）
- M4 teacher trajectory 质量足够（accuracy 上限显著高于 random）

### 7. 性能优化要点

- GraphState 维护 `_sorted_revealed` 和 `_sorted_frontier_nodes`（bisect 增量插入，避免每步重新排序）
- FrontierEncoder 边构建已向量化（tensor fancy indexing，消除 Python 循环）
- `_edge_key_to_id` 预计算（O(1) 边特征查找）
- SubgraphBuilder 仅调用一次，无需特殊优化

## 常用命令

```bash
# 全图 GIN 训练（M3 上界）
python m3/train_full_graph.py --root D:\ogb-master\dataset --device cuda:0

# M3 基准评测
python m3/benchmark_m3.py --pretrained best_model.pt \
  --root D:\ogb-master\dataset --device cuda:0

# M3 快速评测（100 图, 3 seed, 3 预算）
python m3/benchmark_m3.py --pretrained best_model.pt --quick

# M4 轨迹生成
python -m m4.generate_trajectories --pretrained best_model.pt --device cuda:0

# M4 快速模式（200 验证图, B=[5,10], 跳过 T2）
python -m m4.generate_trajectories --pretrained best_model.pt --quick

# 数据分割（首次需运行, ~32 GB 内存）
python split-tools-analyze/split_and_save_ppa.py

# 子集轮换训练
python split-tools-analyze/train_rotate_subsets.py --rotate_epochs 10 --switch_delay 10
```

## 代码约定

- 所有 Python 文件 UTF-8 编码，中文注释
- M3/M4 脚本通过 `sys.path.insert(0, str(_script_dir.parent))` 导入 `active_graph_reader`
- 训练脚本接受 `--root` 指定 OGB 数据集路径（Windows 格式如 `D:\ogb-master\dataset`）
- 模型权重文件：`best_model.pt`（state_dict + val_acc）
- Trajectory 使用 `@dataclass`，序列化用 `torch.save/load`
