"""打印 analysis_stats.json 中所有 budget ratio 合并后的标签分布。"""
import json
import sys
from pathlib import Path

json_path = Path(__file__).resolve().parent.parent.parent / 'm4' / 'analysis' / 'analysis_stats.json'

with open(json_path) as f:
    data = json.load(f)

# 合并所有 budget ratio 的标签分布
merged = {}
total = 0
for ratio_key in ['0_050', '0_100', '0_150']:
    if ratio_key not in data:
        continue
    dist = data[ratio_key]['label_distribution']
    for label, count in dist.items():
        merged[label] = merged.get(label, 0) + count
        total += count

print(f"Total trajectories: {total}")
print(f"Classes: {len(merged)}")
print()
print(f"{'Label':>6} {'Count':>8} {'Ratio':>10}")
print("-" * 26)

for label in sorted(merged.keys(), key=int):
    count = merged[label]
    ratio = count / total if total > 0 else 0
    bar = '█' * int(ratio * 200)
    print(f"{label:>6} {count:>8} {ratio:>9.4f}  {bar}")

# 缺失的类
all_classes = set(str(i) for i in range(37))
missing = all_classes - set(merged.keys())
zero_classes = [k for k, v in merged.items() if v == 0]
if missing:
    print(f"\nMissing classes: {sorted(missing, key=int)}")
if zero_classes:
    print(f"Zero-count classes: {sorted(zero_classes, key=int)}")

# 按数量排序
print(f"\n--- Top 10 ---")
for label, count in sorted(merged.items(), key=lambda x: -x[1])[:10]:
    print(f"  class {label}: {count} ({count/total*100:.1f}%)")
print(f"\n--- Bottom 10 ---")
for label, count in sorted(merged.items(), key=lambda x: x[1])[:10]:
    print(f"  class {label}: {count} ({count/total*100:.1f}%)")
