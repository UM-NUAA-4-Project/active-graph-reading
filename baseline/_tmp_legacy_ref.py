"""捕获 v2 改造前的 agentnet 行为作为回归基线。"""
import sys, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code')); sys.path.insert(0, str(REPO))
import torch, dgl
from baseline.agentnet import AgentnetStrategy
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder
from tools.seeds import deterministic_seed
from datasets import get_dataset

torch.manual_seed(0); random.seed(0)
ds = get_dataset('cifar10', root=r'D:\data\superpixels', splits=('test',), max_graphs=6)
graphs = [dgl.add_self_loop(dgl.remove_self_loop(g)) for g in ds.test_graphs]
agent = AgentnetStrategy()
agent.load(str(REPO / 'baseline/artifacts/cifar10/agentnet.pt'), 'cpu')
builder = SubgraphBuilder()

out = {'rollouts': [], 'own': []}
for gi, g in enumerate(graphs):
    n = g.num_nodes()
    for si in range(2):
        st = GraphState(g, seed_node=deterministic_seed(gi, si, n))
        st._budget_limit = max(1, int(0.10 * n))
        agent.reset(); agent.on_new_frontier(list(st._initial_frontier))
        seq = []
        while not st.is_frontier_empty and st.get_budget_used() < st._budget_limit:
            v = agent.select(st)
            if v is None: seq.append(None); break
            seq.append(v); reveal_node(st, v)
        out['rollouts'].append(seq)
        sub, ids = builder.build(st)
        out['own'].append(agent.own_predict(sub)[0].tolist())
        out.setdefault('seed_rows', []).append(ids.index(st.seed_node))
torch.save(out, REPO / 'baseline/_tmp_legacy_ref.pt')
print('captured', len(out['rollouts']), 'rollouts;', 'logits[0][:4] =', out['own'][0][:4])
