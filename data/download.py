from dgl.data import TUDataset

dataset = TUDataset(name='COLLAB', raw_dir='D:\\data\\COLLAB')

print(f"📊 数据集名称: {dataset.name}")
print(f"📁 包含图的数量: {len(dataset)} 个")
print(f"🏷️ 分类类别数: {dataset.num_labels} 类")
num_feat = getattr(dataset, 'num_features', 0)
print(f"🔢 节点特征维度: {num_feat} (注意：COLLAB原始数据无节点特征)")
print("="*50)

# 统计信息
total_nodes = 0
total_edges = 0
graph_labels = []

# 遍历数据集（兼容元组返回）
for i, item in enumerate(dataset):
    # 解包：如果是元组则分离，否则直接使用
    if isinstance(item, tuple):
        g, label = item
    else:
        g = item
        label = g.ndata['label'][0].item() if 'label' in g.ndata else None

    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    total_nodes += num_nodes
    total_edges += num_edges
    if label is not None:
        graph_labels.append(label)

    # 打印前5个图
    if i < 5:
        print(f"  📌 图 {i+1}: 节点数={num_nodes}, 边数={num_edges}, 标签={label}")

# 计算平均值
if len(dataset) > 0:
    avg_nodes = total_nodes / len(dataset)
    avg_edges = total_edges / len(dataset)

print("... (中间省略了其余图) ...")
print("="*50)
print(f"📈 统计汇总:")
print(f"   - 总节点数: {total_nodes}")
print(f"   - 总边数: {total_edges}")
print(f"   - 平均节点数: {avg_nodes:.2f}")
print(f"   - 平均边数: {avg_edges:.2f}")

# 获取第一个图（同样要兼容）
first_item = dataset[0]
if isinstance(first_item, tuple):
    first_graph, first_label = first_item
else:
    first_graph = first_item
    first_label = first_graph.ndata['label'][0].item() if 'label' in first_graph.ndata else None

print("="*50)
print(f"🔍 第一个图 (索引0) 的详细信息:")
print(f"   - 节点数: {first_graph.num_nodes()}")
print(f"   - 边数: {first_graph.num_edges()}")
print(f"   - 图中边的类型: {first_graph.canonical_etypes}")
print(f"   - 节点特征形状: {first_graph.ndata['feat'].shape if 'feat' in first_graph.ndata else '无特征 (特征为空)'}")
print(f"   - 图标签: {first_label}")