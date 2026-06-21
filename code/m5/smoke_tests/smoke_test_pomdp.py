"""Smoke test — POMDP Reader 核心推理链验证。

使用合成图测试（不加载 OGBG-PPA）：
1. GSAE 前向传播 + 重构 L_rec
2. EvidenceNetwork b + u = 1
3. LSTMBeliefUpdater 多步更新 + reset
4. FinalClassifier 双路径融合
5. POMDPReader 完整 B 步循环
6. POMDPReader vs LearnableReader 对同一 GraphState 做选择
"""

import random
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent.parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import torch
import dgl

from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.reader.belief_updater import LSTMBeliefUpdater
from active_graph_reader.reader.final_classifier import FinalClassifier
from active_graph_reader.reader.pomdp_reader import POMDPReader
from active_graph_reader.reader.learnable_reader import LearnableReader
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder

SEP = "=" * 60


def make_synthetic_graph(n_nodes=30, edge_feat_dim=7, seed=42):
    rng = random.Random(seed)
    src, dst = list(range(n_nodes)), list(range(1, n_nodes)) + [0]
    for _ in range(n_nodes * 2):
        u, v = rng.randint(0, n_nodes - 1), rng.randint(0, n_nodes - 1)
        if u != v:
            src.append(u); dst.append(v)
    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g); g = dgl.add_self_loop(g)
    g.edata['feat'] = torch.randn(g.num_edges(), edge_feat_dim)
    return g


def test_gsae_forward():
    print("\n[Test 1] GSAE forward + reconstruction...")
    g = make_synthetic_graph(30, seed=1)
    state = GraphState(g, seed_node=0)
    subgraph = FrontierGraphBuilder().build(state)

    gsae = GSAE(edge_feat_dim=7, hidden_dim=64, k_state=32, n_layers=3, dropout=0.0)
    gsae.eval()
    out = gsae(subgraph)

    assert out.z.shape == (1, 32), f"z shape {out.z.shape}"
    assert out.v.shape == (1, 64), f"v shape {out.v.shape}"
    assert out.v_hat.shape == (1, 64), f"v_hat shape {out.v_hat.shape}"
    assert out.frontier_embeds.shape[0] > 0, "frontier should not be empty"
    assert out.frontier_embeds.shape[1] == 64, f"frontier_embeds dim {out.frontier_embeds.shape[1]}"

    rec_loss = ((out.v_hat - out.v) ** 2).mean().item()
    print(f"  z: {out.z.shape}, v: {out.v.shape}, frontier: {out.frontier_embeds.shape[0]} nodes")
    print(f"  L_rec = {rec_loss:.6f} [OK]")


def test_evidence_network():
    print("\n[Test 2] EvidenceNetwork b + u = 1...")
    net = EvidenceNetwork(k_state=32, k_evidence=5)
    for _ in range(5):
        z = torch.randn(3, 32)
        out = net(z)
        assert out.e.shape == (3, 5)
        assert (out.e >= 0).all(), "evidence must be non-negative"
        assert torch.allclose(out.b.sum(dim=-1) + out.u, torch.ones(3), atol=1e-5), \
            f"b+u should be 1, got {out.b.sum(dim=-1) + out.u}"
    print(f"  e: {out.e[0].tolist()}, b: {out.b[0].tolist()}, u: {out.u[0]:.4f} [OK]")


def test_belief_updater():
    print("\n[Test 3] LSTMBeliefUpdater multi-step + reset...")
    updater = LSTMBeliefUpdater(k_evidence=5, action_emb_dim=64)
    updater.reset()

    b = torch.ones(1, 5) / 5
    for t in range(4):
        e = torch.abs(torch.randn(1, 5))
        u_t = 5.0 / (1.0 + e.sum())
        emb_a = torch.randn(1, 64)
        b = updater.forward(b, e, u_t, emb_a)
        assert torch.allclose(b.sum(), torch.tensor(1.0), atol=1e-5)
        print(f"  Step {t}: b sum=1, max={b.max().item():.4f}")

    alpha = updater.alpha_accum
    assert (alpha > 0).all(), f"alpha must be > 0 (Dirichlet constraint), min={alpha.min()}"

    updater.reset()
    assert torch.allclose(updater.alpha_accum, torch.ones(5))
    print(f"  After reset: OK [OK]")


def test_final_classifier():
    print("\n[Test 4] FinalClassifier dual-path fusion...")
    g = make_synthetic_graph(20, seed=2)
    state = GraphState(g, seed_node=0)
    subgraph = FrontierGraphBuilder().build(state)

    gsae = GSAE(edge_feat_dim=7, hidden_dim=64, k_state=32, n_layers=3, dropout=0.0)
    ev_net = EvidenceNetwork(k_state=32, k_evidence=5)
    fc = FinalClassifier()

    gsae.eval(); ev_net.eval()

    out_b = fc(subgraph, gsae, ev_net)
    assert torch.allclose(out_b.b_total.sum(dim=-1) + out_b.u_total, torch.ones(1), atol=1e-5)
    print(f"  Path B only: y_hat={out_b.y_hat.item()}, u={out_b.u_total.item():.4f}")

    alpha_lstm = torch.ones(5) + torch.rand(5) * 2
    out_dual = fc(subgraph, gsae, ev_net, alpha_lstm=alpha_lstm)
    assert torch.allclose(out_dual.b_total.sum(dim=-1) + out_dual.u_total, torch.ones(1), atol=1e-5)
    print(f"  Dual path: y_hat={out_dual.y_hat.item()}, u={out_dual.u_total.item():.4f}")

    diff = (out_dual.b_total - out_b.b_total).abs().sum().item()
    assert diff > 1e-6, "dual and single path should differ"
    print(f"  Diff: {diff:.4f} [OK]")


def test_pomdp_reader_loop():
    print("\n[Test 5] POMDPReader complete B-step loop...")
    g = make_synthetic_graph(25, seed=3)
    reader = POMDPReader(
        edge_feat_dim=7, hidden_dim=64, k_state=32, k_evidence=5, n_layers=3, dropout=0.0)
    reader.eval()
    reader.reset()

    state = GraphState(g, seed_node=0)
    B = 5
    chosen_nodes = []
    for step in range(B):
        chosen = reader.select(state)
        if chosen is None:
            print(f"  Step {step}: frontier empty, stopping")
            break
        chosen_nodes.append(chosen)
        reveal_node(state, chosen)
        print(f"  Step {step}: chose {chosen}, "
              f"|V_revealed|={len(state.V_revealed)}, "
              f"frontier={len(state.frontier_edges)}")

    assert len(chosen_nodes) == B, f"Expected {B} steps, got {len(chosen_nodes)}"

    final_out = reader.get_terminal_belief(state)
    assert torch.allclose(final_out.b_total.sum(dim=-1) + final_out.u_total, torch.ones(1), atol=1e-4)
    print(f"  Terminal: y_hat={final_out.y_hat.item()}, "
          f"u={final_out.u_total.item():.4f} [OK]")


def test_compare_with_learnable():
    print("\n[Test 6] POMDPReader vs LearnableReader on same state...")
    g = make_synthetic_graph(20, seed=4)
    state = GraphState(g, seed_node=0)

    pomdp = POMDPReader(
        edge_feat_dim=7, hidden_dim=64, k_state=32, k_evidence=5, n_layers=3, dropout=0.0)
    pomdp.eval(); pomdp.reset()
    c1 = pomdp.select(state)

    lr = LearnableReader(edge_feat_dim=7, hidden_dim=64, n_layers=2, dropout=0.0)
    lr.eval()
    c2 = lr.select(state)

    print(f"  POMDP chose: {c1}, Learnable chose: {c2}")
    assert c1 is not None and c2 is not None, "both readers should select a node"
    assert c1 in state.get_frontier_nodes(), f"{c1} not in frontier"
    assert c2 in state.get_frontier_nodes(), f"{c2} not in frontier"
    print(f"  Both selections valid [OK]")


def main():
    tests = [
        ("GSAE", test_gsae_forward),
        ("EvidenceNetwork", test_evidence_network),
        ("BeliefUpdater", test_belief_updater),
        ("FinalClassifier", test_final_classifier),
        ("POMDPReader loop", test_pomdp_reader_loop),
        ("vs LearnableReader", test_compare_with_learnable),
    ]
    passed = 0
    print(SEP)
    print("Smoke Test — POMDP Reader Core Inference Chain")
    print(SEP)

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{SEP}")
    print(f"{passed}/{len(tests)} tests passed.")
    print(SEP)
    return passed == len(tests)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
