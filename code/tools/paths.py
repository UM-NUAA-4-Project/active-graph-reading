"""路径引导：所有新脚本的第一行 import 前 call setup_path()。

统一替代每个脚本自带的 sys.path 样板；把 <repo>/code 插到 sys.path[0]，
并移除脚本自身所在目录（避免与包名互相遮蔽，如旧 m7 的 datasets 问题）。
"""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]   # <repo>/code
REPO_ROOT = CODE_DIR.parent


def setup_path() -> Path:
    """确保 `code/` 在 sys.path 首位；返回仓库根目录。"""
    s = str(CODE_DIR)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    # 脚本自身目录（如 code/train_reader）从路径中移除，防止遮蔽顶层包
    script_dir = str(Path(__file__).resolve().parent)
    here = str(Path().resolve())
    for d in (script_dir, here):
        if d != s and d in sys.path and Path(d).parent == CODE_DIR:
            sys.path.remove(d)
    return REPO_ROOT
