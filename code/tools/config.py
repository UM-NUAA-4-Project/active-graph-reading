"""config 文件体系：--config JSON 加载、CLI 覆盖合并、实验 config 落盘。

约定：
- effective config = --config JSON 提供 baseline，CLI 显式传入（≠ argparse 默认）的值覆盖；
- 每次实验把 effective config 写入 <output>/config.json；
- 训练脚本同时把同一份 config 内嵌进每个 epoch checkpoint；
- resume/eval 一律以 checkpoint 内嵌 config 为准重建模型。

merge_args_with_config 是旧 m7 train_rl/eval 中"CLI==默认值才用 ckpt 值"
逻辑的唯一实现（此前复制了两份且键名映射各自维护）。
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional


def load_json(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def git_commit() -> Optional[str]:
    """当前 git commit（失败返回 None，不阻断训练）。"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def merge_args_with_config(args, cfg: Dict,
                           key_map: Optional[Dict[str, str]] = None) -> Dict:
    """合并 CLI args 与 config JSON → effective config dict。

    规则：对 cfg 中每个键 k（经 key_map 映射到 args 的 dest），
    - args 上该值 == argparse 默认值 → 采用 cfg 值；
    - args 上被显式改过 → CLI 优先。
    args 中存在而 cfg 中没有的键原样保留；cfg 中 args 没有的键也保留
    （供模型构建使用，如逐层列表）。

    参数 args 需要是 argparse.Namespace 且其 parser 的默认值可通过
    parser.get_default 获取——为免传 parser，约定调用方把
    `args._defaults = {dest: default}` 附上（见 tools/cli.py）。
    """
    key_map = key_map or {}
    defaults = getattr(args, '_defaults', {})
    effective: Dict[str, object] = {}
    # 1) cfg 为 baseline
    effective.update(cfg)
    # 2) CLI 显式覆盖
    for dest in vars(args):
        if dest.startswith('_'):
            continue
        val = getattr(args, dest)
        if dest in defaults and val == defaults[dest]:
            continue                                   # 未显式修改，保持 cfg 值
        cfg_key = key_map.get(dest, dest)
        effective[cfg_key] = val
    return effective


def build_experiment_config(args, extra: Optional[Dict] = None) -> Dict:
    """实验元 config：CLI 全量 + git commit + 附加字段（dataset spec 等）。"""
    cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    cfg['git_commit'] = git_commit()
    if extra:
        cfg.update(extra)
    return cfg
