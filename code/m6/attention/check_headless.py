"""Verify headless AttentionReader: strip classifier keys + strict load ckpt + 3-tuple forward."""
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent   # code/m6/attention
_repo_root = _script_dir.parent.parent          # code/
_project_root = _repo_root.parent               # 仓库根目录
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_script_dir.parent))     # m6/ → 顶层 training_utils

import torch
import dgl

from active_graph_reader.reader.attention.attention_reader import AttentionReader
from training_utils import strip_classifier_keys
from m5.training_utils import infer_reader_config, patch_old_checkpoint
from active_graph_reader.engine.graph_state import GraphState

# ---- 1. strip + strict load BC checkpoint ----
ckpt_path = _repo_root / 'm6/attention/bc/reader_belief_epoch5.pt'
ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
sd = ckpt['reader_state_dict']
stripped = strip_classifier_keys(sd)
n_dropped = len(sd) - len(stripped)
print(f"keys: {len(sd)} -> {len(stripped)} (dropped {n_dropped})")
assert n_dropped > 0, "expected belief_state.classifier.* keys to be dropped"

hidden_dim, n_layers = infer_reader_config(stripped)
reader = AttentionReader(edge_feat_dim=7, hidden_dim=hidden_dim, n_layers=n_layers)
sd_patched, _ = patch_old_checkpoint(stripped)
missing, unexpected = reader.load_state_dict(sd_patched, strict=False)
print(f"missing: {missing}")
print(f"unexpected: {unexpected}")
assert not missing, f"strict-load failed: {missing}"
assert not unexpected, f"unexpected keys: {unexpected}"
print(f"params: {sum(p.numel() for p in reader.parameters()):,}")

# ---- 2. forward_states returns 3-tuple ----
g = dgl.graph(([0, 0, 1], [1, 2, 2]))
g.edata['feat'] = torch.randn(3, 7)
g = dgl.add_self_loop(g)
state = GraphState(g, seed_node=0)
state._budget_limit = 2
beliefs = reader.init_beliefs(1)
results = reader.forward_states([state], temperature=1.0, beliefs=beliefs)
assert len(results[0]) == 3, f"expected 3-tuple, got {len(results[0])}"
scores, probs, frontier_nodes = results[0]
print(f"frontier_nodes={frontier_nodes}  probs_shape={tuple(probs.shape)}")
print("OK: headless reader loads BC checkpoint (selection weights intact)")
