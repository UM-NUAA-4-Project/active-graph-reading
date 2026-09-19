import sys, time, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code')); sys.path.insert(0, str(REPO))
import torch, dgl
from baseline.agentnet import AgentNetModel, symmetrize_with_feat
from datasets import get_dataset

device = 'cuda:0'
torch.manual_seed(0); random.seed(0)
ds = get_dataset('cifar10', root=r'D:\data\superpixels', splits=('train',), max_graphs=64)
graphs = [dgl.add_self_loop(symmetrize_with_feat(dgl.remove_self_loop(g))).to(device)
          for g in ds.train_graphs[:64]]
m = AgentNetModel(5, 1, 10, hidden=128, num_steps=16, mlp_width_mult=2,
                  reduce='log').to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=0.01)
m.train()
# 预热
for g in graphs[:8]:
    loss = torch.nn.functional.cross_entropy(m(g), torch.tensor([1], device=device))
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize(); t0 = time.perf_counter()
N = 48
for i in range(N):
    g = graphs[i % 64]
    loss = torch.nn.functional.cross_entropy(m(g), torch.tensor([1], device=device))
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize()
t_train = (time.perf_counter() - t0) / N * 1000
m.eval()
with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for g in graphs[:48]:
        m(g)
    torch.cuda.synchronize()
t_eval = (time.perf_counter() - t0) / 48 * 1000
print(f"train 样本(前向+反向+step): {t_train:.1f} ms | eval 样本: {t_eval:.1f} ms")

# 验收任务: 3 epochs × (5000 train + 5000 test)
acc = 3 * (5000 * t_train + 5000 * t_eval) / 60000
print(f"验收任务(5000图×3ep): 总计 ≈ {acc:.0f} min（不含数据加载~2min）")
# 全量重训: epochs E × (10000 train + 10000 test)
for E in (20, 50, 100):
    t = E * (10000 * t_train + 10000 * t_eval) / 3600
    print(f"全量重训 10000图 × {E} epochs ≈ {t:.1f} h")
