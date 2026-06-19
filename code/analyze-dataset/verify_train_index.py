"""验证 OGB PPA 训练集 index 是否从 0 开始连续。

结论：如果是 [0,1,2,...,N-1]，则 m4/generate_trajectories.py 的重新编号
      train_idx = list(range(len(train_dataset))) 碰巧正确；
      如果不是，则 train_bc.py 中的 load_graphs_for_trajectories 会发生索引错位。
"""

import sys
sys.path.insert(0, '.')
from ogb.graphproppred import DglGraphPropPredDataset

root = r'D:\ogb-master\dataset'
dataset = DglGraphPropPredDataset(name='ogbg-ppa', root=root)
splits = dataset.get_idx_split()

for name in ('train', 'valid', 'test'):
    idx = splits[name].tolist()
    n = len(idx)
    is_contiguous_from_zero = (idx[0] == 0 and idx[-1] == n - 1)
    is_monotonic = all(idx[i] < idx[i+1] for i in range(len(idx)-1))
    print(f"{name:6s}: len={n:>8,}  first={idx[0]:>8,}  last={idx[-1]:>8,}  "
          f"0-based contiguous: {is_contiguous_from_zero}  monotonic: {is_monotonic}")
    print(f"         first 10: {idx[:10]}")
    print(f"         last  10: {idx[-10:]}")
    if not is_contiguous_from_zero:
        # 检查是否有间隙
        expected = set(range(n))
        actual = set(idx)
        missing = sorted(expected - actual)[:20]
        extra = sorted(actual - expected)[:20]
        if missing:
            print(f"         missing (first 20): {missing}")
        if extra:
            print(f"         extra   (first 20): {extra}")

print()

# 关键测试: 验证 m4 重新编号后的 index 与原始 index 是否一致
train_idx = splits['train'].tolist()
reindexed = list(range(len(train_idx)))
match_count = sum(1 for a, b in zip(train_idx, reindexed) if a == b)
print(f"Train re-index match: {match_count}/{len(train_idx)} ({100*match_count/len(train_idx):.1f}%)")
if match_count == len(train_idx):
    print("CONCLUSION: re-indexing is safe for PPA — train split is [0, 1, ..., N-1]")
else:
    print("CONCLUSION: re-indexing BREAKS index mapping — trajectories store wrong graph_idx!")
