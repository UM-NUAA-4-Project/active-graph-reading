"""Compute BC loss baseline from frontier_analysis.json.

用法: python belief/figures/compute_bc_baseline.py
"""

import json
from pathlib import Path
import numpy as np


def main():
    json_path = Path(__file__).resolve().parent / "frontier_analysis.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run frontier_analysis.py first.")
        return

    with open(json_path) as f:
        data = json.load(f)

    all_sizes = [v['frontier_sizes'] for v in data.values()]
    max_len = max(len(s) for s in all_sizes)

    print(f"Trajectories: {len(all_sizes)}")
    print(f"Max steps: {max_len}\n")
    print(f"{'Step':>5s}  {'N_traj':>7s}  {'Mean':>7s}  {'ln(mean)':>9s}")
    print('─' * 34)

    bc_vals = []
    for t in range(max_len):
        vals = [s[t] for s in all_sizes if t < len(s)]
        n = len(vals)
        m = np.mean(vals)
        bc_t = np.log(m)
        bc_vals.append(bc_t)
        print(f'{t+1:5d}  {n:7d}  {m:7.1f}  {bc_t:9.4f}')

    l = sum(bc_vals) / len(bc_vals)
    print(f'\n{"=" * 34}')
    print(f'l = (bc_1 + ... + bc_{max_len}) / {max_len} = {l:.4f}')
    print(f'Random BC baseline ≈ {l:.2f}')
    print(f'Any BC loss above this = worse than random guessing')


if __name__ == '__main__':
    main()
