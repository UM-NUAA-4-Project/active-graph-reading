# from dgl.data import TUDataset
# 
# dataset = TUDataset(name='COLLAB', raw_dir='D:\\data\\COLLAB')
# 
# print(f"📊 数据集名称: {dataset.name}")
# print(f"📁 包含图的数量: {len(dataset)} 个")
# print(f"🏷️ 分类类别数: {dataset.num_labels} 类")
# num_feat = getattr(dataset, 'num_features', 0)
# print(f"🔢 节点特征维度: {num_feat} (注意：COLLAB原始数据无节点特征)")
# print("="*50)
#
# # 统计信息
# total_nodes = 0
# total_edges = 0
# graph_labels = []
#
# # 遍历数据集（兼容元组返回）
# for i, item in enumerate(dataset):
#     # 解包：如果是元组则分离，否则直接使用
#     if isinstance(item, tuple):
#         g, label = item
#     else:
#         g = item
#         label = g.ndata['label'][0].item() if 'label' in g.ndata else None
# 
#     num_nodes = g.num_nodes()
#     num_edges = g.num_edges()
# 
#     total_nodes += num_nodes
#     total_edges += num_edges
#     if label is not None:
#         graph_labels.append(label)
# 
#     # 打印前5个图
#     if i < 5:
#         print(f"  📌 图 {i+1}: 节点数={num_nodes}, 边数={num_edges}, 标签={label}")
# 
# # 计算平均值
# if len(dataset) > 0:
#     avg_nodes = total_nodes / len(dataset)
#     avg_edges = total_edges / len(dataset)
# 
# print("... (中间省略了其余图) ...")
# print("="*50)
# print(f"📈 统计汇总:")
# print(f"   - 总节点数: {total_nodes}")
# print(f"   - 总边数: {total_edges}")
# print(f"   - 平均节点数: {avg_nodes:.2f}")
# print(f"   - 平均边数: {avg_edges:.2f}")
# 
# # 获取第一个图（同样要兼容）
# first_item = dataset[0]
# if isinstance(first_item, tuple):
#     first_graph, first_label = first_item
# else:
#     first_graph = first_item
#     first_label = first_graph.ndata['label'][0].item() if 'label' in first_graph.ndata else None
# 
# print("="*50)
# print(f"🔍 第一个图 (索引0) 的详细信息:")
# print(f"   - 节点数: {first_graph.num_nodes()}")
# print(f"   - 边数: {first_graph.num_edges()}")
# print(f"   - 图中边的类型: {first_graph.canonical_etypes}")
# print(f"   - 节点特征形状: {first_graph.ndata['feat'].shape if 'feat' in first_graph.ndata else '无特征 (特征为空)'}")
# print(f"   - 图标签: {first_label}")

import dgl
from dgl.data import PeptidesFunctionalDataset
import torch
import numpy as np

def print_dataset_stats(dataset):
    """打印 PeptidesFunctionalDataset 的详细统计信息"""
    print("="*50)
    print("PeptidesFunctionalDataset 统计信息")
    print("="*50)

    # 1. 基本配置信息
    print(f"\n[1] 基本配置")
    print(f"    - 数据集名称:        {dataset.name}")
    print(f"    - 任务类型:          图分类 (Graph Classification)")
    print(f"    - 类别数量:          {dataset.num_classes}")
    print(f"    - 标签类型:          10-way 多任务二分类")
    class_list = ['antifungal', 'cell_cell_communication', 'anticancer',
                  'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
                  'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']
    print(f"    - 功能类别(按顺序):  {class_list}")

    # 2. 数据划分信息
    print(f"\n[2] 数据划分")
    split_dict = dataset.get_idx_split()
    print(f"    - 训练集 (Train):    {len(split_dict['train'])} 个样本")
    print(f"    - 验证集 (Valid):    {len(split_dict['val'])} 个样本")
    print(f"    - 测试集 (Test):     {len(split_dict['test'])} 个样本")
    print(f"    - 总计:              {len(dataset)} 个样本")

    # 3. 图结构统计信息
    print(f"\n[3] 图结构统计")
    num_graphs = len(dataset)
    num_nodes_list = []
    num_edges_list = []
    node_feat_dim = None
    edge_feat_dim = None
    labels_list = []

    # 遍历数据集收集信息 (如果数据集较大，此步骤可能耗时)
    for i in range(num_graphs):
        graph, label = dataset[i]
        num_nodes_list.append(graph.num_nodes())
        num_edges_list.append(graph.num_edges())
        if node_feat_dim is None and 'feat' in graph.ndata:
            node_feat_dim = graph.ndata['feat'].shape[1]
        if edge_feat_dim is None and 'feat' in graph.edata:
            edge_feat_dim = graph.edata['feat'].shape[1]
        labels_list.append(label.numpy())

    # 计算平均值和标准差
    avg_nodes = np.mean(num_nodes_list)
    avg_edges = np.mean(num_edges_list)
    std_nodes = np.std(num_nodes_list)
    std_edges = np.std(num_edges_list)

    print(f"    - 总图数:            {num_graphs}")
    print(f"    - 总节点数:          {sum(num_nodes_list)}")
    print(f"    - 总边数:            {sum(num_edges_list)}")
    print(f"    - 平均节点数:        {avg_nodes:.2f} ± {std_nodes:.2f}")
    print(f"    - 平均边数:          {avg_edges:.2f} ± {std_edges:.2f}")
    print(f"    - 节点特征维度:      {node_feat_dim} (Atom Encoder)")
    print(f"    - 边特征维度:        {edge_feat_dim} (Bond Encoder)")

    # 4. 标签分布统计 (每个功能类别的正负样本数)
    print(f"\n[4] 标签分布 (多任务二分类)")
    labels_array = np.array(labels_list)  # shape: (num_graphs, 10)
    for i, class_name in enumerate(class_list):
        positive_count = np.sum(labels_array[:, i] == 1)
        negative_count = np.sum(labels_array[:, i] == 0)
        print(f"    - {class_name:25s}: 正类 = {positive_count:>6}, 负类 = {negative_count:>6}")

    print("\n" + "="*50)
    print("统计完成！")
    print("="*50)

if __name__ == "__main__":
    # 下载并加载数据集 (首次运行会自动下载)
    # raw_dir 参数可指定下载目录, 默认为 "~/.dgl/"
    print("正在下载并加载 PeptidesFunctionalDataset...")
    dataset = PeptidesFunctionalDataset(raw_dir = "D:/data")  # verbose=True 显示下载进度

    # 打印统计信息
    print_dataset_stats(dataset)