"""Shared constants and path setup for m6/attention (AttentionReader) scripts."""

import sys
from pathlib import Path

K = 37
BUDGET_OPTIONS = [0.05, 0.10, 0.15]

_script_dir = Path(__file__).resolve().parent      # code/m6/attention
_m6_dir = _script_dir.parent                       # code/m6
_repo_root = _m6_dir.parent                        # code/
_project_root = _repo_root.parent                  # 仓库根目录


def setup_paths():
    """Add code/ and m6/ directories to sys.path.

    code/ 提供 active_graph_reader 包与 m5.training_utils；
    m6/  提供顶层 training_utils（脚本目录自动在 sys.path 首位）。
    """
    sys.path.insert(0, str(_repo_root))
    sys.path.insert(0, str(_m6_dir))
