"""Shared constants and path setup for belief scripts."""

import sys
from pathlib import Path

K = 37
BUDGET_OPTIONS = [0.05, 0.10, 0.15]

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_repo_root = _project_root / 'code'
_m5_dir = _repo_root / 'm5'


def setup_paths():
    """Add repo code/, m5/, and belief/ directories to sys.path."""
    sys.path.insert(0, str(_repo_root))
    sys.path.insert(0, str(_m5_dir))
    sys.path.insert(0, str(_script_dir))
