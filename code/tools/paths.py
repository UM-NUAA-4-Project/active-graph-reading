"""路径引导：所有新脚本的第一行 import 前 call setup_path()。

统一替代每个脚本自带的 sys.path 样板；把 <repo>/code 插到 sys.path[0]，
并移除脚本自身所在目录（避免与包名互相遮蔽，如旧 m7 的 datasets 问题）。
"""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]   # <repo>/code
REPO_ROOT = CODE_DIR.parent


def resolve(p) -> Path:
    """输入路径解析（双基准）：相对路径优先按 code/ 解释（新 data/、result/
    布局所在）；若 code/ 下不存在而仓库根下存在，则按仓库根解释（兼容
    'code/result/...' 这类仓库根形式）；都不存在时按 code/ 解析（新产物的
    默认落点）。绝对路径原样返回。"""
    path = Path(p)
    if path.is_absolute():
        return path
    for base in (CODE_DIR, REPO_ROOT):
        cand = base / path
        if cand.exists():
            return cand
    return CODE_DIR / path


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
