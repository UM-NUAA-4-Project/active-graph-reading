import dgl
import torch
from dgl.data import CIFAR10SuperPixelDataset
from torch.utils.data import DataLoader

# 1. 加载数据集
custm_dir = 'D:\CIFAR10-Superpixel'
print("正在加载 CIFAR10 超像素数据集（首次运行将自动下载 2.51GB 数据）...")
train_dataset = CIFAR10SuperPixelDataset(split="train", verbose=True, use_feature= True, raw_dir=custm_dir)
test_dataset = CIFAR10SuperPixelDataset(split="test", verbose=True, use_feature= True, raw_dir=custm_dir)
print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

# 2. 探索一个样本
graph, label = train_dataset[0]
print(f"\n样本图信息:\n{graph}")
print(f"样本标签: {label}")
print(f"节点特征维度: {graph.ndata['feat'].shape[1]}")  # 应为5[reference:10]
print(f"边特征维度: {graph.edata['feat'].shape[1]}")    # 应为1[reference:11]

# 3. 创建 DataLoader
def collate_fn(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 4. 检查一个批次
print("\n检查一个数据批次:")
for batched_graph, labels in train_loader:
    print(f"批次图数量: {batched_graph.batch_size}")
    print(f"批次总节点数: {batched_graph.num_nodes()}")
    print(f"批次总边数: {batched_graph.num_edges()}")
    print(f"标签形状: {labels.shape}")
    break