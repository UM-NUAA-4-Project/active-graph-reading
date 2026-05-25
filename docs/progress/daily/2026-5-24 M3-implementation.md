# M3 阶段实现文档

> 日期：2026-05-24
> 状态：编码器-分类器对齐 OGB GIN 完成，全图上界训练/微调代码就绪
> 前提：Reader（reveal 协议 + 三种策略）已在 M3 早期实现

---

## 一、总体架构

```
mycode/
├── active_graph_reader/          # 核心库
│   ├── engine/                   # 揭示协议引擎（M3 早期完成）
│   │   ├── graph_state.py        #   G_t 部分可观测状态
│   │   ├── reveal_action.py      #   reveal_node(v) 动作执行
│   │   └── reveal_loop.py        #   主循环：策略→动作→更新状态
│   ├── strategies/               # 前沿边选择策略
│   │   ├── base.py               #   策略抽象接口
│   │   ├── registry.py           #   策略注册表
│   │   ├── random_strategy.py    #   随机策略（训练用，唯一无偏）
│   │   ├── bfs_strategy.py       #   BFS 策略
│   │   └── max_frontier_degree_strategy.py  # 最大前沿度策略
│   ├── cost/                     # 成本模型
│   │   ├── base.py               #   成本接口
│   │   ├── registry.py           #   成本注册表
│   │   └── unit_cost.py          #   单位成本（v0: 每节点=1）
│   ├── encoder/                  # 编码器-分类器（M3 新增）
│   │   ├── subgraph_builder.py   #   GraphState → DGLGraph
│   │   ├── gin_encoder.py        #   GINEEncoder（对齐 OGB GIN）
│   │   ├── adapter.py            #   BottleneckAdapter（微调用）
│   │   └── classifier.py         #   GraphClassifier = Encoder + Head
│   ├── dataloader/               # 数据加载
│   │   └── loader.py             #   PPADataset 封装
│   ├── utils/logging.py          # 日志工具
│   ├── config.py                 # RevealConfig 配置类
│   └── main.py                   # 命令行入口
├── train_full_graph.py           # 全图上界训练脚本
├── finetune_reader.py            # Reader 子图微调脚本
├── smoke_test_phase1.py          # 冒烟测试
└── docs/
    ├── roadmap.md                # 项目路线图
    └── 2026-5-20 reader-encoder-classifier.md  # M3 设计文档
```

### 数据流

```
全图训练：
  Full Graph → GINEEncoder → mean pool → Linear head → logits → CE loss
  （不经过 Reader，直接使用完整图）

微调训练：
  Full Graph → Reader(random, budget) → 子图 → GINEEncoder → mean pool → Linear head → logits → CE loss
  （Reader 先读出部分节点/边，再编码分类）
```

---

## 二、Reader 引擎（M3 早期已实现）

### 2.1 揭示协议

- **G_0**：k=1 随机种子节点 + 其邻居 ID 列表/边特征可见
- **动作**：`reveal_node(v)` 揭示节点 v，同时揭示 v 到已揭示节点的边特征
- **成本**：每节点 = 1（单位成本模型）
- **预算 B**：允许揭示的最大节点数（不含种子）

### 2.2 GraphState

管理揭示过程中的部分可观测状态：

```
V_revealed         : Set[int]           已揭示节点
frontier_edges     : Set[(u,v)]         前沿边
revealed_edge_feats: Dict[(u,v),Tensor] 已揭示边特征（7维，PPA原始）
```

### 2.3 RevealLoop

```
while budget > 0:
    a_t = strategy.select(frontier_edges)   # 策略选边
    if a_t is None: break
    reveal_node(state, a_t)                 # 揭示目标节点
    budget -= cost(a_t)                     # 扣除成本
```

### 2.4 三种策略


| 策略                | 选择逻辑                       | 训练使用       |
| ------------------- | ------------------------------ | -------------- |
| random              | 从前沿边中均匀随机选           | 是（唯一无偏） |
| bfs                 | 优先最早加入前沿的边           | 否（评估对比） |
| max-frontier-degree | 前沿节点在已揭示子图中的度最大 | 否（评估对比） |

---

## 三、编码器-分类器（M3 核心输出）

### 3.1 与 OGB 官方 GIN 的一致性

编码器严格对齐 OGB 官方 `examples/graphproppred/ppa/conv.py` 中的 GINConv，逐项对比如下：


| 组件      | OGB 官方 (PyG)                                                | 本实现 (DGL)                                | 一致 |
| --------- | ------------------------------------------------------------- | ------------------------------------------- | ---- |
| 节点嵌入  | `Embedding(1, 300)` 全图共享                                  | `Embedding(1, hidden_dim)`                  | ✓   |
| 边编码器  | 每层独立`Linear(7, 300)`                                      | 每层独立`Linear(edge_feat_dim, hidden_dim)` | ✓   |
| GIN MLP   | `Linear(d→2d) → BN → ReLU → Linear(2d→d)` （无最终激活） | 同                                          | ✓   |
| 消息函数  | `ReLU(src_h + edge_emb)`                                      | 同                                          | ✓   |
| 聚合      | sum                                                           | `fn.sum`                                    | ✓   |
| eps       | 可学习，初始化为 0                                            | 同                                          | ✓   |
| 残差连接  | 无                                                            | 无                                          | ✓   |
| 末层 ReLU | 跳过                                                          | 跳过                                        | ✓   |
| 图读出    | mean pooling                                                  | `dglnn.AvgPooling`                          | ✓   |
| 分类头    | `Linear(300, num_classes)`                                    | `nn.Linear(hidden_dim, n_classes)`          | ✓   |

**参考超参**（OGBG-PPA leaderboard）：hidden_dim=300, n_layers=5, dropout=0.5

### 3.2 SubgraphBuilder

将 `GraphState` → 标准 `DGLGraph`：

- 节点范围：仅 `V_revealed`
- 边范围：仅 `revealed_edge_feats` 中的边（双向各一条）
- 边特征：原始 7 维向量，存放在 `g.edata['feat']`
- 输出 `(sub_g, original_ids)`，供编码器使用

### 3.3 GINEEncoder

```
GINEEncoder:
  node_emb      : Embedding(1, hidden_dim)           # 全图共享
  convs[0..4]   : GINELayer × 5                      # 每层含 edge_encoder + MLP + eps
  bns[0..4]     : BatchNorm1d × 5
  adapters[0..4]: Identity / BottleneckAdapter       # 微调时注入
  pool          : AvgPooling (mean readout)

forward(g, node_ids):
  h = node_emb(node_ids)           # (N, d)
  for i in 0..4:
    h = convs[i](g, h, edge_feat)  # GIN 卷积
    h = bns[i](h)
    if i < 4: h = ReLU(h)         # 末层跳过 ReLU
    h = Dropout(h)
    if adapters[i] is not Identity:
      h = h + adapters[i](h)       # 残差适配器
  return pool(g, h)
```

### 3.4 GraphClassifier

```
GraphClassifier = GINEEncoder + Linear(head)
  encoder : GINEEncoder(edge_feat_dim, hidden_dim, n_layers, dropout)
  head    : Linear(hidden_dim, n_classes)
```

加载预训练权重与微调策略 API：

```python
model = GraphClassifier(edge_feat_dim=7, n_classes=37)
model.load_pretrained("best_model.pt")           # strict=True 加载
model.apply_finetune_strategy(                    # 配置冻结层 + 适配器
    freeze_layers={0, 1},
    adapter_layers=set(),
    freeze_node_emb=True,
)
model.log_trainable()                             # 打印可训练参数统计
```

### 3.5 BottleneckAdapter

Houlsby et al. (ICML 2019) 瓶颈适配器，用于参数高效微调：

```
Adapter(x) = Linear_up(ReLU(Linear_down(x)))
h_out      = GINLayer(h_in) + Adapter(GINLayer(h_in))  # 残差连接

结构：Linear(d → d//ratio) → ReLU → Linear(d//ratio → d)
参数量（ratio=16, d=300）： ~11K / 层
初始化：近零初始化（weight ∼ N(0, 0.02)，bias = 0），使适配器初始近似恒等变换
```

---

## 四、全图上界训练

### 4.1 train_full_graph.py

在完整 PPA 图上训练 GIN，建立性能天花板。

```bash
python train_full_graph.py --root /mnt/d/ogb-master/dataset
python train_full_graph.py --epochs 50 --device cuda:0 --root dataset
python train_full_graph.py --eval-only --load best_model.pt --root dataset
```

**训练参数**（对齐 OGB 参考配置）：


| 参数       | 值    |
| ---------- | ----- |
| hidden_dim | 300   |
| n_layers   | 5     |
| dropout    | 0.5   |
| batch_size | 32    |
| epochs     | 100   |
| lr         | 0.001 |
| optimizer  | Adam  |

**输出文件**（默认保存到脚本所在目录）：


| 文件            | 说明                                                  |
| --------------- | ----------------------------------------------------- |
| `checkpoint.pt` | 每 5 epoch 覆盖保存（含 optimizer state）             |
| `best_model.pt` | 最佳验证准确率时保存（仅 model_state_dict + val_acc） |
| `curves.pt`     | 训练/验证/测试准确率曲线                              |

**训练流程**：

1. 加载 OGBG-PPA 数据集（DGL 格式）
2. 所有图添加自环（`dgl.add_self_loop`）
3. DataLoader + `collate_dgl` 批处理
4. 每 epoch：训练 → 验证 → 测试评估
5. 用 OGB Evaluator 计算准确率

---

## 五、Reader 子图微调

### 5.1 finetune_reader.py

加载全图预训练权重，在 Reader 产生的部分子图上微调。

```bash
python finetune_reader.py --strategy A   --pretrained best_model.pt --root dataset
python finetune_reader.py --strategy B1  --pretrained best_model.pt --root dataset
python finetune_reader.py --strategy A+B1 --pretrained best_model.pt --root dataset
```

**三种微调策略**：

```
策略 A：冻结早期层
  Layer 0: [FROZEN]
  Layer 1: [FROZEN]
  Layer 2: [TRAINABLE]
  Layer 3: [TRAINABLE]
  Layer 4: [TRAINABLE]
  Head   : [TRAINABLE]
  node_emb: 冻结
  可训练参数：~1.1M (60%)

策略 B1：全冻结 + 每层瓶颈适配器
  Layer 0: [FROZEN] + BottleneckAdapter(300→18→300)
  Layer 1: [FROZEN] + BottleneckAdapter(300→18→300)
  Layer 2: [FROZEN] + BottleneckAdapter(300→18→300)
  Layer 3: [FROZEN] + BottleneckAdapter(300→18→300)
  Layer 4: [FROZEN] + BottleneckAdapter(300→18→300)
  Head   : [TRAINABLE]
  node_emb: 冻结
  可训练参数：~66K (3.6%)

策略 A+B1：冻结早期层 + 后期层加适配器
  Layer 0: [FROZEN]                          ← 无适配器
  Layer 1: [FROZEN]                          ← 无适配器
  Layer 2: [TRAINABLE] + BottleneckAdapter(300→18→300)
  Layer 3: [TRAINABLE] + BottleneckAdapter(300→18→300)
  Layer 4: [TRAINABLE] + BottleneckAdapter(300→18→300)
  Head   : [TRAINABLE]
  node_emb: 冻结
  可训练参数：~1.1M + ~33K ≈ 1.13M (62%)
```

**微调训练细节**：


| 参数             | 值                                    |
| ---------------- | ------------------------------------- |
| 训练 Reader 策略 | random（唯一无偏）                    |
| 预算分布         | Uniform{1%, 2%, 5%, 10%, 20%, 50%} × |
| lr               | 1e-4（低于全图训练的 1e-3）           |
| epochs           | 20（默认）                            |
| inner_batch      | 8（子图 mini-batch）                  |
| 验证预算         | {5%, 10%, 20%, 50%}                   |
| 适配器瓶颈比     | 16（默认，即 300→18）                |

**参数加载顺序是关键**：

```
1. 创建 GraphClassifier（此时 adapters 全为 Identity，无适配器参数）
2. load_pretrained(path)  ← strict=True，双方均无 adapter key，加载成功
3. apply_finetune_strategy(...)  ← 添加适配器、冻结指定层
4. 开始微调训练
```

---

## 六、冒烟测试

`smoke_test_phase1.py` 验证全链路前向通过、无 shape 错误：

额外测试：batch 多图（2 张不同预算子图）、无边子图（budget=0）边界情况。

```
Reader(BFS, B=10) → SubgraphBuilder → GINEEncoder → GraphClassifier
                                                      ↓
                                              logits shape: (1, 37) ✓
```

---

## 七、与 M3 Roadmap 的对照


| Roadmap Item          | 状态                                |
| --------------------- | ----------------------------------- |
| 确认数据集 OGBG-PPA   | ✓                                  |
| 全图分类器基线 GIN    | ✓ (train_full_graph.py)            |
| v0 协议实现           | ✓ (engine/)                        |
| 部分图表示            | ✓ (GraphState + SubgraphBuilder)   |
| 三种固定策略          | ✓ (random/bfs/max-frontier-degree) |
| 编码器-分类器（尺子） | ✓ (encoder/)                       |
| 全图上界训练          | ✓ (train_full_graph.py)            |
| 预训练 + 微调管线     | ✓ (finetune_reader.py)             |
| 多种子平均评估        | 待运行                              |
| 预算-性能曲线绘制     | 待运行                              |
| Kill criterion 检查   | 待运行                              |

---

## 八、下一步

1. **运行全图训练**获得 `best_model.pt` 和上界数值
2. **运行三种微调策略**比较 Reader 子图上的性能
3. **多预算 × 多种子评估**：绘制 budget-accuracy 曲线，回答 M3 核心问题
4. **Kill criterion 判断**：BFS <10% 预算是否已达 ~90% 上界
5. 若 headroom 存在，推进 M4（轨迹收集与评分）
