"""统一评估入口（兼容分发 shim，自 2026-09-19 拆分为两个专用脚本）。

reader 评估与 baseline 评估的实现已分别迁移至：
  - eval_reader.py    --checkpoint 模式 → <checkpoint 同目录>/eval_learnable.json
  - eval_baselines.py --strategy 模式  → result/{ds}/fixed/eval_fixed.json

本文件只做参数分发，保持既有命令/文档/脚本不破坏。新代码请直接使用
两个专用入口。
"""

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]


def main():
    argv = sys.argv[1:]
    has_ckpt = any(a == '--checkpoint' or a.startswith('--checkpoint=')
                   for a in argv)
    has_strategy = any(a == '--strategy' or a.startswith('--strategy=')
                       for a in argv)
    if not has_ckpt and not has_strategy:
        if '-h' in argv or '--help' in argv or not argv:
            print(__doc__.strip())
            return
        raise SystemExit(
            "eval.py 已拆分：--checkpoint 模式 → eval_reader.py，"
            "--strategy 模式 → eval_baselines.py（二者恰好传其一）")
    if has_ckpt and has_strategy:
        raise SystemExit(
            "eval.py 已拆分：--checkpoint 与 --strategy 不可同时传入；"
            "请直接使用 eval_reader.py / eval_baselines.py")
    if has_ckpt:
        from train_reader.eval_reader import main as reader_main
        reader_main()
    else:
        from train_reader.eval_baselines import main as baselines_main
        baselines_main()


if __name__ == '__main__':
    main()
