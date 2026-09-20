"""learnable reader 评估（base/evidence 统一入口）。

自 train_reader/eval.py 拆分（2026-09-19）：reader 评估与 baseline 评估
各自独立（基线走 eval_baselines.py；旧 eval.py 保留为兼容分发 shim）。
reader 结构完全由 checkpoint 内嵌 config 重建（无需 CLI 传模型参数）。

一次运行覆盖全部请求 split（默认 valid+test；无验证集的数据集如 cifar10
自动检测并跳过），结果合并为单个统一 schema JSON（comparison 脚本直接
可用）→ <checkpoint 同目录>/eval_learnable.json（--output-dir 可覆盖）。

evidence reader：rollout 走 forward_states_evidence（选点仍与 base 一致），
终判 = 读满预算后补充一次证据前向的 ev_logits argmax（决策-分类一体，
含最后一次 reveal 的证据）；冻结上界分类器对同一批最终子图的判定写为
参照键 acc_clf_*（落盘 acc_clf_reference_{split}）——协议口径见
docs/design/evidence-reader.md。
--export-confidence 导出逐样本 [最大类置信度, 正误] → 同目录
calibration.json（校准/可靠性分析，预测 5.3c；ev=证据头终判 / clf=冻结分类器）。

协议要点（AGENTS.md Protocol）：预算 = 揭示节点数（不含 seed），比例档
取自 DatasetSpec（默认 0.05/0.10/0.15）；种子 = 确定性公式
(tools/seeds)；前沿边特征揭示前可见（one-ended，零预算）；frontier 空
而预算未耗尽时按 (seed_node, B) 播种的确定性 RNG 跳转（消耗预算）。

用法:
  python code/train_reader/eval_reader.py --dataset cifar10 \
      --root D:\\data\\superpixels \
      --checkpoint result/cifar10/experiments/exp1/evidence/rl/checkpoint_epoch5.pt \
      --classifier result/cifar10/classifier/best_cifar10_classifier.pt
"""

import argparse
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (add_common_args, eval_resolve,  # noqa: E402
                               load_classifier_frozen,
                               load_reader_from_checkpoint, run_split_eval)
from train_reader.core.eval_loop import evaluate_reader  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='learnable reader 评估')
    add_common_args(parser)
    parser.add_argument('--checkpoint', required=True,
                        help='reader checkpoint（结构以内嵌 config 重建）')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='跨图 lockstep 的图批量大小')
    parser.add_argument('--export-confidence', action='store_true',
                        help='导出逐样本 [最大类置信度, 正误] → checkpoint 同目录 '
                             'calibration.json（校准/可靠性分析，预测 5.3c；'
                             'ev=证据头终判 / clf=冻结分类器）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # spec 只读探针：splits=() 的缓存项永不访问数据字段，不触发加载
    spec = get_dataset(args.dataset, root=args.root, splits=(),
                       max_graphs=args.max_graphs,
                       stratified=args.stratified).spec
    budget_options = spec.budget_options

    classifier = load_classifier_frozen(eval_resolve(args.classifier), device)
    print(f"Classifier: {args.classifier}")
    builder = SubgraphBuilder()

    ckpt_path = eval_resolve(args.checkpoint)
    print(f"Checkpoint: {ckpt_path}")
    reader = load_reader_from_checkpoint(ckpt_path, device,
                                         stochastic=False, temperature=1.0)
    print(f"Reader: kind={reader.config.kind} "
          f"hidden={reader.config.encoder_hidden_dims} "
          f"(from checkpoint config)")

    out_dir = (eval_resolve(args.output_dir) if args.output_dir
               else ckpt_path.parent)
    out_path = out_dir / 'eval_learnable.json'
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = {} if args.export_confidence else None
    config = {
        'dataset': spec.name,
        'mode': 'reader',
        'strategies': ['learnable'],
        'strategy_or_checkpoint': str(ckpt_path),
        'checkpoint': str(ckpt_path),
        'classifier': str(eval_resolve(args.classifier)),
        'budgets': list(budget_options),
        'n_graphs': {},          # 逐 split 累积
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'seed_method': SEED_METHOD_STR,
        'full_graph_ref': {},    # 逐 split 累积
    }

    def eval_split(split, graphs, labels):
        results = evaluate_reader(reader, classifier, builder, graphs, labels,
                                  budget_options, device, args.argmax,
                                  chunk_size=args.chunk_size,
                                  n_seeds=args.n_seeds,
                                  confidence_out=(calibration.setdefault(split, {})
                                                  if calibration is not None
                                                  else None))
        yield 'learnable', results

    run_split_eval(args, classifier, budget_options, config, out_path,
                   eval_split, calibration=calibration)
    print(f"JSON saved to {out_path}")
    if calibration:
        calib_path = out_dir / 'calibration.json'
        print(f"Calibration saved to {calib_path}")
        print(f"绘图: python code/comparison/plot_calibration.py --json {calib_path}")
    print(f"绘图: python code/comparison/plot_eval_results.py --json {out_path}")


if __name__ == '__main__':
    main()
