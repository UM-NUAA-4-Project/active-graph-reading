# Plan: M4 Trajectory V3 智能筛选

质量优先 + 多样性补充 + 类别平衡 + 难度感知，四合一。

---

## 1. 难度分类（用户定义）

基于跨 budget CE 行为模式（`analysis_cross_budget_ce.png`）：


| 层级    | 标准                                | 类别                                                              | 数量 |
| ------- | ----------------------------------- | ----------------------------------------------------------------- | ---- |
| **T0**  | 任何预算下均无成功案例              | 0, 32                                                             | 2    |
| **T1**  | 高预算 CE ≥ 0.4，低预算误差极高    | 13, 25, 30                                                        | 3    |
| **T2**  | 高预算 CE 0.2~0.4，低预算误差较高   | 5, 8, 11, 12, 14, 18, 19, 20, 29, 31, 34                          | 11   |
| **T3a** | 所有预算 CE < 0.2，且轨迹量 > 3000  | 2, 6, 17                                                          | 3    |
| **T3b** | 所有预算 CE < 0.2，但轨迹量 ≤ 3000 | 1, 3, 4, 7, 9, 10, 15, 16, 21, 22, 23, 24, 26, 27, 28, 33, 35, 36 | 18   |

---

## 2. 每类预算分配


| 层级 | 策略         | 每类预算          | 精英比例 | 多样性阈值 |
| ---- | ------------ | ----------------- | -------- | ---------- |
| T0   | 无数据，跳过 | —                | —       | —         |
| T1   | 全部保留     | `actual_count`    | 0        | 0.95       |
| T2   | 尽可能保留   | `min(count, 900)` | 0.3      | 0.75       |
| T3a  | 激进削减     | `min(count, 300)` | 0.4      | 0.60       |
| T3b  | 适度保留     | `min(count, 500)` | 0.4      | 0.65       |

**T0 说明**：在任何 budget 下 classifier 都无法正确分类的类，没有轨迹可供筛选——自然跳过。

**T1 说明**（13, 25, 30）：极少量轨迹（每类 < 50 条），每条都是"高 budget 下好不容易成功的路径"。全部保留，精英比例 0（无需区分），去重阈值 0.95（仅去真正的比特级重复）。

**T2 说明**（class 18 为代表）：困难类，CE 不随 budget 改善（class 18 gradient CE: 0.190→0.275），但轨迹量不算少。保留尽可能多（上限 900），精英 30% 保证质量底线，多样性阈值 0.75 保留不同阅读策略。

**T3a 说明**（17, 6, 2）：极易类 + 海量轨迹。class 17 从 17,751→300（削减 98.3%）。300 条中 120 条按最低 CE 直接保留（精英），180 条在剩余中按多样性（cos_sim < 0.60）补充。

**T3b 说明**：容易但不过量的类，适度保留。500 条中 200 精英 + 300 多样性补充。

---

## 3. 类内分层筛选算法

对每个类别 c，预算 B：

```
1. 收集该类所有轨迹（跨 teacher、graph、ratio）
2. 按 CE 升序排列
3. 精英池：取前 ⌊B × elite_ratio⌋ 条，不做多样性检查，直接保留
4. 多样性池：从剩余候选中贪心选入：
   - 对每条候选轨迹，计算与所有已保留轨迹的最大余弦相似度
   - 若 max_cos_sim < diversity_threshold → 保留
   - 否则 → 丢弃
   - 直到已保留总数 = B 或候选耗尽
5. 图级上限：每个 (graph_idx, ratio) 最多保留 max_per_graph 条（CE 最低优先）
```

**余弦相似度**（与 V2 一致）：

```
轨迹 v → binary vector (reveal_order 节点集合)
v_norm = v / ||v||₂
cos_sim = dot(v1_norm, v2_norm) = |交集| / sqrt(|v1| × |v2|)
```

**为什么精英池不做多样性检查？** 如果 top CE 轨迹都指向同一子结构 → 那是 class c 的黄金阅读路径 → reader 必须学会它。

---

## 4. 处理流程

```
输入: code/m4/output/trajectories_merged_r*.pt (4 个文件: 0_050, 0_100, 0_150, 0_200)

Step 1: 加载所有 ratio 文件，展平轨迹
Step 2: 按 final_pred（= 类别标签）分组
Step 3: 查找难度表，分配每类预算和参数
Step 4: 每类独立执行分层筛选
Step 5: 图级上限裁剪
Step 6: 按原嵌套结构重建 {teacher_name: [[Traj per graph], ...]}
Step 7: 按 ratio 分别保存
Step 8: 输出筛选报告 JSON

输出: code/m4/filtered_v3/trajectories_filtered_v3_r*.pt (4 个文件)
       code/m4/filtered_v3/filter_report.json
```

---

## 5. 实现

### 文件：`code/m4/filter_trajectories.py`（重写）

**难度表**（硬编码，基于用户分类）：

```python
DIFFICULTY_CONFIG = {
    # label: (tier, budget, elite_ratio, diversity_threshold)
}

T0_LABELS  = {0, 32}
T1_LABELS  = {13, 25, 30}
T2_LABELS  = {5, 8, 11, 12, 14, 18, 19, 20, 29, 31, 34}
T3a_LABELS = {2, 6, 17}
T3b_LABELS = {1, 3, 4, 7, 9, 10, 15, 16, 21, 22, 23, 24, 26, 27, 28, 33, 35, 36}
```

**CLI 参数**：


| 参数              | 默认值                      | 说明                    |
| ----------------- | --------------------------- | ----------------------- |
| `--input-dir`     | `m4/output`                 | 含 merged .pt 的目录    |
| `--output-dir`    | `m4/filtered_v3`            | 输出目录                |
| `--max-per-graph` | 5                           | 每图每 ratio 最大轨迹数 |
| `--input-glob`    | `trajectories_merged_r*.pt` | 输入匹配模式            |
| `--output-prefix` | `trajectories_filtered_v3`  | 输出前缀                |

（预算/精英/阈值从难度表读取，无需 CLI 参数）

**核心函数**：

```
build_difficulty_config() → Dict[int, TierConfig]
stratified_select(trajs, config) → List[Trajectory]
apply_graph_cap(selected, max_per_graph) → List[Trajectory]
reconstruct_nested(filtered_flat, original_structure) → dict
generate_report(before_stats, after_stats) → dict
```

### 新文件：`code/m4/filtered_v3/filter_report.json`

```json
{
  "config": { "max_per_graph": 5, ... },
  "per_class": {
    "17": {
      "tier": "T3a",
      "before": 17751,
      "after": 300,
      "elite_count": 120,
      "diversity_count": 180,
      "avg_ce_before": 0.048,
      "avg_ce_after": 0.042
    },
    ...
  },
  "totals": { "before": 40996, "after": 11400 }
}
```

---

## 6. 验证

1. `python code/m4/filter_trajectories.py --input-dir code/m4/output --output-dir code/m4/filtered_v3`
2. 检查 class 17: ~300 条（从 ~17,751）
3. 检查 T1 类全部保留
4. 检查 T0 类无变化
5. 输出文件可被 `analyze_trajectories.py` 加载
6. 运行 analyze 对比筛选前后分布：`python code/m4/analyze_trajectories.py --input-dir code/m4/filtered_v3 --output-dir code/m4/analysis_v3`
