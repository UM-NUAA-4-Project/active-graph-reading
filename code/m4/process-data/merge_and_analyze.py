"""Stage6: Merge teacher (stage5) and reader trajectories, dedup, cap, analyze.

Usage:
  python code/m4/process-data/merge_and_analyze.py
    --teacher-dir code/m4/data/stage5_filtered
    --reader-dir code/m4/data/reader_trajectories
    --output-dir code/m4/data/stage6_merged
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent.parent  # process-data/ -> m4/ -> code/
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

from m4.teachers.base import Trajectory

# ---- Constants ----
REGULAR_CLASSES = {1, 2, 3, 4, 6, 12, 17, 18, 21, 23, 24, 33, 36}
SCARCE_CLASSES = {0, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 22,
                  25, 26, 27, 28, 29, 30, 31, 32, 34, 35}
N_CLASSES = 37
SIM_THRESHOLD = 0.85
MAX_PER_GRAPH_BUDGET = 10  # for regular classes only
TARGET_PER_CLASS_BUDGET = 200

# Budget canonical keys and filename mappings
BUDGET_KEYS = ["0_050", "0_100", "0_150", "0_200"]
BUDGET_VALUES = [0.05, 0.10, 0.15, 0.20]
BUDGET_TO_KEY = dict(zip(BUDGET_VALUES, BUDGET_KEYS))
KEY_TO_BUDGET = dict(zip(BUDGET_KEYS, BUDGET_VALUES))

# Round reveal_ratio to canonical budget (snap to nearest)
BUDGET_SNAP = {0.05: 0.05, 0.10: 0.10, 0.15: 0.15, 0.20: 0.20}


def snap_budget(ratio: float) -> float:
    """Snap a reveal_ratio to the nearest canonical budget value."""
    val = round(ratio, 2)
    if val in BUDGET_SNAP:
        return BUDGET_SNAP[val]
    # fallback: nearest
    budgets = sorted(BUDGET_SNAP.keys())
    return min(budgets, key=lambda b: abs(b - val))


def similarity(set1: Set[int], set2: Set[int]) -> float:
    """Cosine similarity: |A ∩ B| / sqrt(|A| * |B|)"""
    n_intersection = len(set1 & set2)
    if n_intersection == 0:
        return 0.0
    return n_intersection / math.sqrt(len(set1) * len(set2))


def parse_args():
    p = argparse.ArgumentParser(description='Stage6: Merge & Analyze Trajectories')
    p.add_argument('--teacher-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'stage5_filtered'))
    p.add_argument('--reader-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'reader_trajectories'))
    p.add_argument('--output-dir', type=str,
                   default=str(_script_dir.parent / 'data' / 'stage6_merged'))
    p.add_argument('--sim-threshold', type=float, default=0.85)
    p.add_argument('--max-per-graph', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    # placeholder — filled in later tasks
    args = parse_args()
    print("Scaffold OK — args:", args)
