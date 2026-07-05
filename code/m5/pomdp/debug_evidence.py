#!/usr/bin/env python3
"""GSAE + EvidenceNetwork terminal diagnostic REPL.

Features:
  1. Sample 1 graph per class, cache, release raw data
  2. Terminal commands: forward pass → text display of evidence/loss
  3. Backward pass → per-layer gradient norms table
  4. On-demand matplotlib chart windows (shown on screen, not saved)

Usage:
  /mnt/d/anoconda/envs/gnn/python.exe code/m5/pomdp/debug_evidence.py \
    --data-dir m4/data/stage7_filtered \
    --root D:/ogb-master/dataset \
    --checkpoint m5/pomdp/phase01/checkpoint_epoch20.pt \
    --device cuda:0
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_m5_dir = _script_dir.parent
_repo_root = _m5_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_m5_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from active_graph_reader.reader.gsae import GSAE
from active_graph_reader.reader.evidence_network import EvidenceNetwork
from active_graph_reader.engine.graph_state import GraphState
from active_graph_reader.engine.reveal_action import reveal_node
from active_graph_reader.gnn.frontier.graph_builder import FrontierGraphBuilder
from training_utils import merge_trajectories, edl_loss, load_graphs_for_trajectories

# ---------------------------------------------------------------------------
# matplotlib — non-blocking on-screen display (no file save)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()  # interactive mode — show() returns immediately

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_labels(root: str):
    csv_path = Path(root) / 'ogbg_ppa' / 'raw' / 'graph-label.csv.gz'
    return pd.read_csv(csv_path, compression='gzip', header=None).values.flatten()


def _load_checkpoint(path: str, gsae: GSAE, evidence_net: EvidenceNetwork,
                     device) -> dict:
    """Load GSAE + EvidenceNetwork from Phase 0+1 / Phase 2 / Phase 3 checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = {}

    if 'gsae_state_dict' in ckpt:
        gsae.load_state_dict(ckpt['gsae_state_dict'])
        evidence_net.load_state_dict(ckpt['evidence_state_dict'])
        config = ckpt.get('config', {})
        kind = 'Phase 0+1'
    elif 'reader_state_dict' in ckpt:
        rsd = ckpt['reader_state_dict']
        gsae_sd = {k[5:]: v for k, v in rsd.items() if k.startswith('gsae.')}
        ev_sd = {k[14:]: v for k, v in rsd.items() if k.startswith('evidence_net.')}
        gsae.load_state_dict(gsae_sd)
        evidence_net.load_state_dict(ev_sd)
        config = ckpt.get('config', {})
        kind = 'Phase 2/3'
    else:
        kind = 'UNKNOWN'

    print(f"  Loaded {kind} checkpoint: epoch {ckpt.get('epoch', 'N/A')}")
    return config


# ---------------------------------------------------------------------------
# terminal colors (ANSI escapes)
# ---------------------------------------------------------------------------

class TC:
    """Terminal color escapes."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RST = '\033[0m'


def _bar(frac: float, width: int = 40) -> str:
    """ASCII bar chart for a single value."""
    n = int(frac * width)
    if frac > 0.8:
        color = TC.RED
    elif frac > 0.3:
        color = TC.YELLOW
    else:
        color = TC.GREEN
    return f"{color}{'█' * n}{'░' * (width - n)}{TC.RST}"


# ---------------------------------------------------------------------------
# chart display (on-screen windows, non-blocking)
# ---------------------------------------------------------------------------

def show_evidence_chart(e_vec, b_vec, true_label, info):
    """Display evidence + belief chart in a non-blocking window."""
    K = len(e_vec)
    green, blue, red = '#2ecc71', '#3498db', '#e74c3c'

    fig, (ax_e, ax_b) = plt.subplots(2, 1, figsize=(14, 8), facecolor='white',
                                      num='Evidence & Belief')
    fig.clf()
    (ax_e, ax_b) = fig.subplots(2, 1)

    e_colors = [green if i == true_label else blue for i in range(K)]
    b_colors = [green if i == true_label else blue for i in range(K)]

    ax_e.bar(range(K), e_vec, color=e_colors, width=0.8, edgecolor='white', linewidth=0.3)
    ax_e.axvline(x=true_label - 0.4, color=green, linewidth=2, alpha=0.5, linestyle='--')
    ax_e.set_title(f'Evidence e  (max={e_vec.max():.4f} at class {np.argmax(e_vec)},  '
                   f'mean={e_vec.mean():.4f})', fontweight='bold', fontsize=13)
    ax_e.set_xlim(-1, K)
    ax_e.grid(axis='y', alpha=0.3)

    ax_b.bar(range(K), b_vec, color=b_colors, width=0.8, edgecolor='white', linewidth=0.3)
    ax_b.axvline(x=true_label - 0.4, color=green, linewidth=2, alpha=0.5, linestyle='--')
    pred = int(np.argmax(b_vec))
    correct = 'CORRECT' if pred == true_label else 'WRONG'
    c = green if pred == true_label else red
    ax_b.set_title(f'Belief b  (pred={pred}, true={true_label} → {correct})',
                   fontweight='bold', fontsize=13, color=c)
    ax_b.set_xlabel('class')
    ax_b.set_xlim(-1, K)
    ax_b.grid(axis='y', alpha=0.3)

    info_text = (
        f"graph_idx={info['graph_idx']}  |V|={info['n_total']}  |E|={info['n_edges_total']}  "
        f"label={info['label']}  step={info['step']}/{info['max_steps']}  "
        f"revealed={info['n_revealed']}  frontier={info['n_frontier']}"
    )
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=8, color='#7f8c8d',
             fontfamily='monospace')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.show()


def show_gradient_chart(grad_dict):
    """Display gradient norms chart in a non-blocking window."""
    if not grad_dict:
        return

    sorted_items = sorted(grad_dict.items(), key=lambda x: x[1]['grad_norm'], reverse=True)
    top_n = min(40, len(sorted_items))
    items = sorted_items[:top_n]

    names = [k for k, v in items]
    norms = [v['grad_norm'] for v in items]

    green, yellow, red = '#2ecc71', '#f39c12', '#e74c3c'
    colors = []
    for v in items:
        gn = v[1]['grad_norm']
        if gn > 1e-4:
            colors.append(green)
        elif gn > 1e-6:
            colors.append(yellow)
        else:
            colors.append(red)

    fig, ax = plt.subplots(figsize=(14, 12), facecolor='white', num='Gradient Norms')
    fig.clf()
    ax = fig.subplots(1, 1)
    y_pos = range(len(names))
    ax.barh(y_pos, norms, color=colors, height=0.7, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5, fontfamily='monospace')
    ax.invert_yaxis()
    ax.set_xlabel('gradient norm')
    ax.set_xscale('log')

    for i, (name, v) in enumerate(items):
        ax.annotate(f'{v["ratio"]:.1e}', (v['grad_norm'], i),
                    fontsize=6, va='center', ha='left', color='#7f8c8d',
                    xytext=(3, 0), textcoords='offset points')

    n_ok = sum(1 for c in colors if c == green)
    n_warn = sum(1 for c in colors if c == yellow)
    n_dead = sum(1 for c in colors if c == red)
    ax.set_title(f'Gradient Norms (top {top_n})  '
                 f'OK={n_ok}  WARN={n_warn}  DEAD={n_dead}',
                 fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.show()


# ---------------------------------------------------------------------------
# main debugger
# ---------------------------------------------------------------------------

class Debugger:
    """Terminal REPL for GSAE + EvidenceNetwork diagnosis."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.gsae = GSAE(
            edge_feat_dim=7, hidden_dim=args.hidden_dim, k_state=128,
            n_layers=args.n_layers, dropout=args.dropout,
            use_struct=not args.no_struct,
        ).to(self.device)

        self.evidence_net = EvidenceNetwork(
            hidden_dim=args.hidden_dim, k_evidence=37,
        ).to(self.device)

        self.builder = FrontierGraphBuilder()

        # checkpoint
        self.lambda_kl = args.lambda_kl
        if args.checkpoint:
            print(f"\n=== Loading checkpoint: {args.checkpoint} ===")
            cfg = _load_checkpoint(
                str(_repo_root / args.checkpoint),
                self.gsae, self.evidence_net, self.device,
            )
            self.lambda_kl = cfg.get('lambda_kl', args.lambda_kl)

        # data
        self.sampled = []          # [(label, traj), ...]
        self.graph_cache = {}      # graph_idx → (g, label)

        # state
        self._idx = 0              # current class index
        self._step = None          # reveal step (None = final)
        self._info = {}            # current subgraph info
        self._last_loss = None
        self._losses = {}
        self._ev_out = None
        self._grad_dict = {}

    # ---- data loading ----

    def load_data(self):
        """Sample 1 trajectory per class, cache graphs, release raw data."""
        data_dir = _repo_root / self.args.data_dir

        print("\n=== Loading labels ===")
        labels_arr = _read_labels(self.args.root)
        print(f"  Total graphs: {len(labels_arr)}, classes: {len(set(labels_arr))}")

        print("\n=== Merging trajectories ===")
        all_trajs = merge_trajectories(data_dir)
        print(f"  Total trajectories: {len(all_trajs)}")

        by_label = defaultdict(list)
        for t in all_trajs:
            by_label[int(labels_arr[t.graph_idx])].append(t)

        present = sorted(by_label)
        missing = sorted(set(range(37)) - set(present))
        print(f"  Classes with data: {len(present)}/37")
        if missing:
            print(f"  Missing: {missing}")

        rng = np.random.RandomState(42)
        for label in present:
            trajs = by_label[label]
            self.sampled.append((label, trajs[rng.randint(0, len(trajs))]))
        # rarest first
        self.sampled.sort(key=lambda x: len(by_label[x[0]]))

        # load only needed graphs
        needed = {t.graph_idx for _, t in self.sampled}
        print(f"\n=== Loading {len(needed)} graphs ===")
        self.graph_cache = load_graphs_for_trajectories(self.args.root, needed)

        # filter out any that failed to load
        self.sampled = [(l, t) for l, t in self.sampled
                         if t.graph_idx in self.graph_cache]

        del all_trajs, by_label, labels_arr
        print(f"  Cached {len(self.sampled)} classes, {len(self.graph_cache)} graphs")

    # ---- current trajectory access ----

    def _traj(self):
        return self.sampled[self._idx]

    def _graph(self):
        label, traj = self._traj()
        return self.graph_cache[traj.graph_idx]

    def _max_step(self):
        _, traj = self._traj()
        return len(traj.reveal_order)

    # ---- subgraph construction ----

    def build_subgraph(self, step=None):
        label, traj = self._traj()
        g, _ = self._graph()
        max_s = len(traj.reveal_order)

        if step is None or step < 0:
            step = max_s
        step = min(step, max_s)

        state = GraphState(g, seed_node=traj.seed_node)
        for i in range(step):
            reveal_node(state, traj.reveal_order[i])

        sub = self.builder.build(state)
        self._info = {
            'graph_idx': traj.graph_idx,
            'n_total': g.num_nodes(),
            'n_edges_total': g.num_edges(),
            'label': label,
            'step': step,
            'max_steps': max_s,
            'n_revealed': sub.n_revealed,
            'n_frontier': len(sub.frontier_original_ids),
            'n_sub_edges': sub.g.num_edges() if sub.n_revealed > 0 else 0,
            'seed': traj.seed_node,
        }
        self._step = step
        return sub

    # ---- forward pass ----

    def forward(self):
        self.zero_grad()
        sub = self.build_subgraph(self._step)

        self.gsae.train()
        self.evidence_net.train()

        gsae_out = self.gsae(sub)
        ev_out = self.evidence_net(gsae_out.v)

        label = self._info['label']
        dev = ev_out.e.device

        # EDL loss
        y_oh = torch.zeros(1, 37, device=dev)
        y_oh[0, label] = 1.0
        edl_tot, edl_mse, edl_var, edl_kl = edl_loss(
            ev_out.e,
            ev_out.alpha,
            ev_out.S,
            y_oh, 37,
            lambda_kl=self.lambda_kl,
            return_components=True,
        )

        # CE
        ce = F.cross_entropy(ev_out.b,
                             torch.tensor([label], device=dev))

        pred = ev_out.b.argmax().item()

        self._losses = {
            'edl_total': edl_tot.item(),
            'edl_mse': edl_mse.item(),
            'edl_var': edl_var.item(),
            'edl_kl': edl_kl.item(),
            'ce': ce.item(),
            'pred': pred,
            'correct': (pred == label),
        }
        self._last_loss = edl_tot
        self._ev_out = ev_out

        return ev_out

    # ---- backward pass ----

    def backward(self):
        if self._last_loss is None:
            print(f"{TC.RED}No loss computed. Run 'f' first.{TC.RST}")
            return {}

        self._last_loss.backward()

        grad_dict = {}
        for m_name, mod in [('gsae', self.gsae), ('evidence', self.evidence_net)]:
            for p_name, p in mod.named_parameters():
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    pn = p.data.norm().item()
                    grad_dict[f'{m_name}.{p_name}'] = {
                        'grad_norm': gn, 'param_norm': pn,
                        'ratio': gn / max(pn, 1e-10),
                    }
        self._grad_dict = grad_dict
        return grad_dict

    def zero_grad(self):
        self.gsae.zero_grad()
        self.evidence_net.zero_grad()
        self._grad_dict = {}
        self._last_loss = None

    # ---- display helpers ----

    def _header(self):
        info = self._info or {}
        label, traj = self._traj()
        g, _ = self._graph()
        idx = self._idx
        n = len(self.sampled)
        s = info.get('step', self._step or '?')
        ms = info.get('max_steps', self._max_step())
        br = (info.get('step', 0) / max(g.num_nodes(), 1)) if info else 0
        print(f"\n{TC.BOLD}{'='*70}{TC.RST}")
        print(f"{TC.BOLD}Class {label}  [{idx+1}/{n}]  "
              f"graph_idx={traj.graph_idx}  |V|={g.num_nodes()}  |E|={g.num_edges()}{TC.RST}")
        print(f"Step {s}/{ms}  (budget {br:.1%})  "
              f"revealed={info.get('n_revealed','?')}  frontier={info.get('n_frontier','?')}")

    def _print_forward_result(self):
        ev = self._ev_out
        info = self._info

        e_np = ev.e.squeeze(0).detach().cpu().numpy()
        b_np = ev.b.squeeze(0).detach().cpu().numpy()
        u_val = ev.u.item()
        S_val = ev.S.item()

        # top-5 evidence classes
        top5_e = np.argsort(e_np)[::-1][:5]
        top5_b = np.argsort(b_np)[::-1][:5]

        pred = self._losses['pred']
        true_label = info['label']
        correct = self._losses['correct']

        print(f"\n{TC.BOLD}── Evidence ──{TC.RST}")
        print(f"  e max={e_np.max():.4f}  mean={e_np.mean():.4f}  min={e_np.min():.4f}")
        print(f"  Top-5 classes (by evidence): ", end='')
        parts = []
        for c in top5_e:
            mark = f"{TC.GREEN}*{TC.RST}" if c == true_label else ""
            parts.append(f"{mark}{c}:{e_np[c]:.4f}")
        print('  '.join(parts))

        print(f"\n{TC.BOLD}── Belief ──{TC.RST}")
        print(f"  b max={b_np.max():.4f}  mean={b_np.mean():.4f}")
        print(f"  Top-5 classes (by belief):  ", end='')
        parts = []
        for c in top5_b:
            mark = f"{TC.GREEN}*{TC.RST}" if c == true_label else ""
            parts.append(f"{mark}{c}:{b_np[c]:.4f}")
        print('  '.join(parts))

        print(f"\n{TC.BOLD}── Dirichlet ──{TC.RST}")
        u_str = f"{TC.RED}{u_val:.4f}{TC.RST}" if u_val > 0.9 else f"{TC.YELLOW}{u_val:.4f}{TC.RST}" if u_val > 0.5 else f"{TC.GREEN}{u_val:.4f}{TC.RST}"
        S_str = f"{TC.GREEN}{S_val:.1f}{TC.RST}" if S_val > 50 else f"{TC.YELLOW}{S_val:.1f}{TC.RST}" if S_val > 40 else f"{TC.RED}{S_val:.1f}{TC.RST}"
        print(f"  S (strength) : {S_str}  (K=37)")
        print(f"  u (uncertain): {u_str}  {_bar(u_val)}")
        print(f"  Model is     : {'CERTAIN' if u_val < 0.5 else ('UNCERTAIN' if u_val < 0.9 else 'CLUELESS')}")

        print(f"\n{TC.BOLD}── Losses ──{TC.RST}")
        r = self._losses
        print(f"  EDL total : {r['edl_total']:.4f}")
        print(f"    MSE     : {r['edl_mse']:.6f}")
        print(f"    Var     : {r['edl_var']:.6f}")
        print(f"    KL      : {r['edl_kl']:.6f}  (lambda_kl={self.lambda_kl})")
        ce_str = f"{TC.RED}{r['ce']:.4f}{TC.RST}" if r['ce'] > 3.5 else f"{TC.YELLOW}{r['ce']:.4f}{TC.RST}" if r['ce'] > 2.0 else f"{TC.GREEN}{r['ce']:.4f}{TC.RST}"
        print(f"  CE        : {ce_str}  (ln(37) = {np.log(37):.4f})")

        print(f"\n{TC.BOLD}── Prediction ──{TC.RST}")
        if correct:
            print(f"  {TC.GREEN}CORRECT  pred={pred} == true={true_label}{TC.RST}")
        else:
            print(f"  {TC.RED}WRONG    pred={pred} != true={true_label}{TC.RST}")

    def _print_gradient_result(self):
        if not self._grad_dict:
            return

        sorted_items = sorted(self._grad_dict.items(),
                              key=lambda x: x[1]['grad_norm'], reverse=True)

        # summary
        norms = [v['grad_norm'] for v in self._grad_dict.values()]
        n_ok = sum(1 for x in norms if x > 1e-4)
        n_warn = sum(1 for x in norms if 1e-6 <= x <= 1e-4)
        n_dead = sum(1 for x in norms if x < 1e-6)
        total = len(norms)

        print(f"\n{TC.BOLD}── Gradients ──{TC.RST}")
        print(f"  Total params with grad: {total}")
        print(f"  {TC.GREEN}OK    (>1e-4): {n_ok:3d}{TC.RST}")
        print(f"  {TC.YELLOW}WARN  (1e-6~1e-4): {n_warn:3d}{TC.RST}")
        print(f"  {TC.RED}DEAD  (<1e-6): {n_dead:3d}{TC.RST}")

        # per-module summary
        for prefix, name in [('gsae.', 'GSAE'), ('evidence.', 'Evidence')]:
            mod_items = [(k, v) for k, v in sorted_items if k.startswith(prefix)]
            if not mod_items:
                continue
            mod_norms = [v['grad_norm'] for _, v in mod_items]
            print(f"\n  {TC.BOLD}{name}:{TC.RST}  "
                  f"min={min(mod_norms):.2e}  max={max(mod_norms):.2e}  "
                  f"mean={np.mean(mod_norms):.2e}")

        # top 20 table
        print(f"\n{TC.BOLD}  Top 20 by grad norm:{TC.RST}")
        print(f"  {'param':<50s} {'grad':>10s} {'weight':>10s} {'ratio':>10s}")
        print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*10}")
        for key, v in sorted_items[:20]:
            gn = v['grad_norm']
            if gn > 1e-4:
                color = TC.GREEN
            elif gn > 1e-6:
                color = TC.YELLOW
            else:
                color = TC.RED
            print(f"  {color}{key:<50s} {gn:10.2e} {v['param_norm']:10.4f} {v['ratio']:10.2e}{TC.RST}")

    def _print_help(self):
        print(f"""
{TC.BOLD}Commands:{TC.RST}
  {TC.CYAN}f{TC.RST} / {TC.CYAN}forward{TC.RST}    GSAE → Evidence → loss
  {TC.CYAN}b{TC.RST} / {TC.CYAN}backward{TC.RST}   loss.backward() → gradient table
  {TC.CYAN}fb{TC.RST}              forward + backward
  {TC.CYAN}n{TC.RST} / {TC.CYAN}next{TC.RST}       next class
  {TC.CYAN}p{TC.RST} / {TC.CYAN}prev{TC.RST}       previous class
  {TC.CYAN}g{TC.RST} / {TC.CYAN}goto N{TC.RST}     jump to class index N
  {TC.CYAN}s{TC.RST} / {TC.CYAN}step N{TC.RST}     set reveal step (0=seed only, -1=final)
  {TC.CYAN}plot{TC.RST}            show evidence+belief chart window
  {TC.CYAN}gradplot{TC.RST}        show gradient chart window
  {TC.CYAN}z{TC.RST} / {TC.CYAN}zero{TC.RST}       zero gradients
  {TC.CYAN}i{TC.RST} / {TC.CYAN}info{TC.RST}       list all classes + graph info
  {TC.CYAN}q{TC.RST} / {TC.CYAN}quit{TC.RST}       exit
  {TC.CYAN}h{TC.RST} / {TC.CYAN}help{TC.RST}       this help
""")

    # ---- command handlers ----

    def cmd_forward(self):
        try:
            self.forward()
        except Exception as e:
            print(f"{TC.RED}Forward error: {e}{TC.RST}")
            import traceback
            traceback.print_exc()
            return
        self._print_forward_result()

    def cmd_backward(self):
        if self._last_loss is None:
            print(f"{TC.RED}No loss. Run 'f' first.{TC.RST}")
            return
        try:
            self.backward()
        except Exception as e:
            print(f"{TC.RED}Backward error: {e}{TC.RST}")
            import traceback
            traceback.print_exc()
            return
        self._print_gradient_result()

    def cmd_next(self):
        if self._idx < len(self.sampled) - 1:
            self._idx += 1
            self._step = None
            self.zero_grad()
            self._info = {}
            self._header()
        else:
            print("Already at last class.")

    def cmd_prev(self):
        if self._idx > 0:
            self._idx -= 1
            self._step = None
            self.zero_grad()
            self._info = {}
            self._header()
        else:
            print("Already at first class.")

    def cmd_goto(self, idx: int):
        if 0 <= idx < len(self.sampled):
            self._idx = idx
            self._step = None
            self.zero_grad()
            self._info = {}
            self._header()
        else:
            print(f"Invalid index. Range: 0-{len(self.sampled)-1}")

    def cmd_step(self, val):
        if val is None:
            # toggle between 0, max//2, max
            if self._step is None or self._step == self._max_step():
                val = 0
            elif self._step == 0:
                val = self._max_step() // 2
            else:
                val = self._max_step()
        self._step = int(val)
        self.zero_grad()
        self._info = {}
        # build subgraph to update info
        self.build_subgraph(self._step)
        self._header()

    def cmd_plot(self):
        if self._ev_out is None:
            print(f"{TC.RED}Run 'f' first.{TC.RST}")
            return
        e_np = self._ev_out.e.squeeze(0).detach().cpu().numpy()
        b_np = self._ev_out.b.squeeze(0).detach().cpu().numpy()
        show_evidence_chart(e_np, b_np, self._info['label'], self._info)

    def cmd_gradplot(self):
        if not self._grad_dict:
            print(f"{TC.RED}Run 'b' first.{TC.RST}")
            return
        show_gradient_chart(self._grad_dict)

    def cmd_info(self):
        self._header()
        if self._info:
            info = self._info
            print(f"  seed_node  : {info.get('seed', '?')}")
            print(f"  subgraph   : |V|={info['n_revealed'] + info['n_frontier']}  "
                  f"|E|={info['n_sub_edges']}")
            print(f"  revealed   : {info['n_revealed']}")
            print(f"  frontier   : {info['n_frontier']}")
        # also show class distribution summary
        print(f"\n  Class samples (total {len(self.sampled)}):")
        for i, (label, traj) in enumerate(self.sampled):
            mark = f"{TC.BOLD}>{TC.RST}" if i == self._idx else " "
            g, _ = self.graph_cache[traj.graph_idx]
            print(f"  {mark} [{i:2d}] class {label:2d}  |V|={g.num_nodes():4d}  "
                  f"graph_idx={traj.graph_idx}")

    # ---- REPL ----

    def run(self):
        print(f"\n{TC.BOLD}{'='*70}{TC.RST}")
        print(f"{TC.BOLD}  GSAE + Evidence Debugger{TC.RST}")
        print(f"  {len(self.sampled)} classes loaded, {len(self.graph_cache)} graphs cached")
        print(f"  Type 'h' for help, 'q' to quit")
        print(f"{TC.BOLD}{'='*70}{TC.RST}")
        self._header()

        while True:
            try:
                raw = input(f"\n{TC.CYAN}> {TC.RST}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ('q', 'quit', 'exit'):
                break
            elif cmd in ('h', 'help', '?'):
                self._print_help()
            elif cmd in ('f', 'forward'):
                self.cmd_forward()
            elif cmd in ('b', 'backward'):
                self.cmd_backward()
            elif cmd == 'fb':
                self.cmd_forward()
                self.cmd_backward()
            elif cmd in ('n', 'next'):
                self.cmd_next()
            elif cmd in ('p', 'prev'):
                self.cmd_prev()
            elif cmd in ('g', 'goto'):
                try:
                    self.cmd_goto(int(arg))
                except (TypeError, ValueError):
                    print("Usage: goto <index>")
            elif cmd in ('s', 'step'):
                try:
                    self.cmd_step(int(arg) if arg is not None else None)
                except (TypeError, ValueError):
                    print("Usage: step <number>")
            elif cmd == 'plot':
                self.cmd_plot()
            elif cmd == 'gradplot':
                self.cmd_gradplot()
            elif cmd in ('z', 'zero'):
                self.zero_grad()
                print("Gradients zeroed.")
            elif cmd in ('i', 'info'):
                self.cmd_info()
            else:
                print(f"Unknown command: {cmd}. Type 'h' for help.")

        print(f"\n{TC.GRAY}Done.{TC.RST}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GSAE + EvidenceNetwork terminal debugger',
    )
    parser.add_argument('--data-dir', default='m4/data/stage7_filtered')
    parser.add_argument('--root', default=r'D:\ogb-master\dataset')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', default='',
                        help='Phase 0+1 / Phase 2 checkpoint (relative to repo root)')
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no-struct', action='store_true',
                        help='Disable structural features')
    parser.add_argument('--lambda-kl', type=float, default=0.01)
    args = parser.parse_args()

    dbg = Debugger(args)
    dbg.load_data()
    if len(dbg.sampled) == 0:
        print("ERROR: No trajectories sampled. Check --data-dir and --root.")
        sys.exit(1)
    dbg.run()


if __name__ == '__main__':
    main()
