# Active Graph Reader — Code

## 环境

- **Conda 环境**: `gnn` (Python 3.10, PyTorch 2.3.0+cu121, DGL 2.2.1)
- **Python 路径**: `/mnt/d/anoconda/envs/gnn/python.exe`
- **数据集**: OGBG-PPA（158,100 图，37 类，7 维边特征，无节点特征）

```bash
pip install requirements.txt
```

## 数据路径

数据集已下载到 `D:\ogb-master\dataset\`（约 30 GB）。所有脚本默认从此路径加载，无需额外配置：

```bash
# 默认加载
python m3/smoke_test.py

# 指定其他路径
python m3/smoke_test.py --root /path/to/dataset
```

**不要使用相对路径 `dataset`**——会触发 OGB 重新下载 30 GB 数据。

## Smoke Test

所有冒烟测试已统一到 `m3/smoke_test.py`。

### 合成图测试（默认，无需 OGBG-PPA，推荐先跑）

```bash
python m3/smoke_test.py
```

测试内容：
- G_0 初始化 + frontier edge feature 可见性（不消耗 budget）
- reveal_node 协议正确性
- 三种策略（random / bfs / max-frontier-degree）
- SubgraphBuilder → GINEEncoder → GraphClassifier 端到端
- 多图 batch 推理
- get_frontier_edge_feat 错误守卫

### OGBG-PPA 测试（需要数据集）

```bash
python m3/smoke_test.py --real-data
```

在合成图测试基础上追加真实 PPA 图测试（graph 0），包括 Reader → Encoder → Classifier 全流程。

## Quick Benchmark

### 协议验证（无需预训练模型）

```bash
# 30 validation graphs, 2 seeds, 3 budgets — 仅验证协议可运行
python m3/benchmark_m3.py --quick --device cpu
```

输出：协议运行统计 + JSON 报告。accuracy 为随机值（无预训练模型）。

### 正式 Quick Benchmark（需要预训练模型）

```bash
# 先训练全图模型
python m3/train_full_graph.py --device cuda:0

# 再跑 quick benchmark
python m3/benchmark_m3.py --pretrained best_model.pt --quick --device cpu

# 或指定参数
python m3/benchmark_m3.py --pretrained best_model.pt \
  --k-per-class 1 --seeds-per-graph 2 \
  --budgets-rel 0.05 0.10 0.20 \
  --strategies random bfs max-frontier-degree \
  --device cpu
```

输出：`m3/m3_benchmark_report.json` + `m3/m3_benchmark_curves.png`。

## 正式 Benchmark

```bash
# 全图 GIN 训练（上界）
python m3/train_full_graph.py --device cuda:0

# 多 seed 多策略多预算评测
python m3/benchmark_m3.py --pretrained best_model.pt \
  --k-per-class 10 --seeds-per-graph 5 \
  --device cuda:0

# M4 轨迹生成
python -m m4.generate_trajectories \
  --pretrained best_model.pt --device cuda:0

# 数据分割（首次使用需运行一次，~32 GB 内存）
python split-tools-analyze/split_and_save_ppa.py
```

## 目录结构

```
code/
├── active_graph_reader/     # 核心库
│   ├── engine/              #   揭示协议引擎 (GraphState, reveal_node, RevealLoop)
│   ├── strategies/          #   random / bfs / max-frontier-degree
│   ├── encoder/             #   GINEEncoder, GraphClassifier, SubgraphBuilder (向量化)
│   ├── cost/                #   成本模型
│   ├── dataloader/          #   PPADataset + ReaderDataset (IterableDataset)
│   ├── reader.py            #   ActiveGraphReader 高层封装
│   └── config.py
├── m3/                      # M3 脚本
│   ├── train_full_graph.py  #   全图 GIN 训练
│   ├── benchmark_m3.py      #   多策略多种子评测（使用 ActiveGraphReader）
│   ├── M3_finetune_reader.py
│   ├── smoke_test.py        #   冒烟测试（合成图 + 可选 OGBG-PPA）
│   └── smoke_test_v2.py     #   优化后 reader pipeline 冒烟测试
├── m4/                      # M4 轨迹模块
│   ├── generate_trajectories.py
│   ├── teachers/            #   gradient / info_gain / structural / hybrid
│   └── ...
└── split-tools-analyze/     # 数据分割工具
```

## 关键协议细节

- **Frontier edge feature 可见**: Reader 在选择下一个 frontier node 之前，可通过 `state.get_frontier_edge_feat(u, v)` 读取连接已揭示节点和前沿节点的边特征。G_0 初始化后，seed→frontier 的边特征立即可读，不消耗 budget。
- **Seed 不消耗 budget**: G_0 的 seed node 揭示不计入 budget。
- **协议 v1**: k=1 seed, reveal_node(v) per step, unit cost, budget = 揭示节点数。Frontier edge feature 在 reveal 前可见。
