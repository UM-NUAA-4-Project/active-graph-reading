# -*- coding: utf-8 -*-
"""Attention-distribution figures for the state-attention readout.

Reads eval_state_attention.json (fresh + frozen) for CIFAR10/COLLAB and renders:
  attn_dist_heatmap.(pdf|png)  fresh mass by budget x relative step position,
                               per-row entropy and terminal-mass annotations
  attn_dist_curves.(pdf|png)   per-budget fresh distributions (solid, H in
                               legend) vs frozen-decoder distributions (dashed)
Outputs land beside this script.
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path.home() / ".zcode" / "skills" / "nature-figure" / "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})
BLUES = ["#9ECAE1", "#4292C6", "#08519C"]
DATASETS = [("cifar10", "CIFAR10 superpixels"), ("collab", "COLLAB")]


def load(kind):
    out = {}
    for ds, _ in DATASETS:
        p = RES / ds / "experiments" / kind / "eval_state_attention.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = []
        for i, (b, bv) in enumerate(sorted(d["splits"]["test"]["by_budget"].items(),
                                           key=lambda kv: float(kv[0]))):
            b = float(b)
            rel = [c / b for c in bv["attn_dist"]["bin_centers"]]
            rows.append({"b": b, "rel": rel, "dist": bv["attn_dist"]["dist"],
                         "H": bv["alpha_norm_entropy"],
                         "last": bv["alpha_last_mass"], "rank": i})
        out[ds] = rows
    return out


def style_panel(ax, tag, name):
    ax.set_title(f"{tag}  {name}", loc="left", fontweight="bold", fontsize=7)
    ax.set_xlabel("relative step position (fraction of budget)", fontsize=6.5)
    ax.tick_params(labelsize=6.5)


def draw(fig, name, exclude=()):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(HERE / f"{name}.alignment.json"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        exclude_axes=exclude)


fresh = load("state-attn")
frozen = load("state-attn-freeze")

# ---------------- Fig 1: heatmap (fresh) ----------------
vmax = max(max(r["dist"]) for ds, _ in DATASETS for r in fresh[ds])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55),
                         gridspec_kw={"wspace": 0.55})
im = None
for ax, (ds, name) in zip(axes, DATASETS):
    rows = fresh[ds]
    mat = np.array([r["dist"] for r in rows])
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.tick_params(which="minor", length=0)
    for i, r in enumerate(rows):
        for j, v in enumerate(r["dist"]):
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        rotation=90, rotation_mode="anchor",
                        fontsize=5.5,
                        color="white" if v > 0.45 * vmax else "black")
        ax.text(0.012, i, f"H={r['H']:.2f}   last={r['last']:.0%}",
                ha="left", va="center", fontsize=6, color="0.15",
                fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['b']:.0%}" for r in rows], fontsize=6.5)
    ax.set_ylabel("budget ratio", fontsize=6.5)
    style_panel(ax, "a" if ds == "cifar10" else "b", name)
    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks([0.5, 5.5, 10.5, 15.5, 19.5])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=6.5)
cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
cbar.set_label("attention mass", fontsize=6.5)
cbar.ax.tick_params(labelsize=6)
cbar.outline.set_linewidth(0.6)
draw(fig, "attn_dist_heatmap", exclude=[cbar.ax])
fig.savefig(HERE / "attn_dist_heatmap.svg", bbox_inches="tight")
fig.savefig(HERE / "attn_dist_heatmap.pdf", bbox_inches="tight")
fig.savefig(HERE / "attn_dist_heatmap.png", dpi=600, bbox_inches="tight")
fig.savefig(HERE / "attn_dist_heatmap.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 2: curves (fresh solid + frozen dashed) ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
for ax, (ds, name) in zip(axes, DATASETS):
    rows = fresh[ds]
    half = sum(sum(r["dist"][:10]) for r in rows) / len(rows)
    ax.axvspan(0, 0.5, color="0.94", lw=0, zorder=0)
    ax.axvline(1.0, ls=":", lw=0.7, color="0.45", zorder=1)
    for k, r in enumerate(frozen[ds]):
        ax.plot(r["rel"], r["dist"], ls="--", lw=0.8, color="0.55", zorder=2,
                label="frozen decoder" if k == 0 else None)
    for r in rows:
        ax.plot(r["rel"], r["dist"], lw=1.2, marker="o", ms=2.4,
                color=BLUES[r["rank"]], zorder=3,
                label=f"B={r['b']:.0%}, H={r['H']:.2f}")
    ax.text(1.0, 1.015, "terminal", rotation=0, ha="right", va="bottom",
            fontsize=5.5, color="0.35", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_ylabel("attention mass", fontsize=6.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.94", edgecolor="0.85"))
    labels.append(f"first half = {half:.1%} of mass")
    ax.legend(handles, labels, loc="upper left", fontsize=5.8,
              handlelength=1.6, borderaxespad=0.2)
    style_panel(ax, "a" if ds == "cifar10" else "b", name)
draw(fig, "attn_dist_curves")
fig.savefig(HERE / "attn_dist_curves.svg", bbox_inches="tight")
fig.savefig(HERE / "attn_dist_curves.pdf", bbox_inches="tight")
fig.savefig(HERE / "attn_dist_curves.png", dpi=600, bbox_inches="tight")
fig.savefig(HERE / "attn_dist_curves.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)

for ds, _ in DATASETS:
    half = sum(sum(r["dist"][:10]) for r in fresh[ds]) / len(fresh[ds])
    print(ds, "fresh first-half mass =", round(half, 4))
print("figures written to", HERE)
