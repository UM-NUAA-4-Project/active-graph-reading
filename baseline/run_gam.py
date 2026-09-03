"""训练/评估 GAM baseline（单智能体，逐图预算比例）。

gam_src/ 内为 GAM 原实现（逐字复制，KDD 2018, Lee/Rossi/Kong，
benedekrozemberczki/GAM-master，MIT License）。本脚本只做对齐性适配，
不改原模块：

1. 单智能体：--repetitions 1（原默认 10 = 多次随机起点读取 + 多数投票，
   即"多智能体综合"语义，我们对其为 1）。
2. 预算 = 逐图比例：原实现 --time 是全数据集统一的步数；我们按
   Active Graph Reader 协议改为 time = round(budget_ratio × 图节点数)
   （GAM 从随机起点出发走 time 步，访问 time+1 个节点——与协议
   "预算不含 seed" 的计数一致）。
3. 训练后分别评估 valid 与 test（原实现只评估 test）。

每个预算档一个独立训练（模型重新初始化），记录写入
<output-root>/budget_{key}/：gam_logs.json、predictions_{split}.csv、
record.json；所有档位汇总写 <output-root>/summary.json。

用法:
  python baseline/run_gam.py --dataset collab --data-dir baseline/data/collab \
      --budgets 0.10 0.15 0.20 --repetitions 1 --epochs 10
"""

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_BASELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASELINE_DIR / 'gam_src'))

import gam  # noqa: E402  GAM 原实现（from utils import ... 需要该目录在 path 首位）


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ratio_key(r: float) -> str:
    return f'{r:.2f}'.replace('.', '_')


class BudgetGAMTrainer(gam.GAMTrainer):
    """GAMTrainer + 逐图预算比例（time = round(ratio × n_nodes)）。

    process_graph 复制自原实现，仅循环上界改为按图计算；score 拆成
    score_split 以便分别评估 valid/test。
    """

    def __init__(self, args, budget_ratio: float):
        super().__init__(args)
        self.budget_ratio = budget_ratio

    def _graph_time(self, graph) -> int:
        return max(1, int(round(self.budget_ratio * graph.number_of_nodes())))

    def process_graph(self, graph_path, batch_loss):
        data = json.load(open(graph_path))
        graph, features = gam.create_features(data, self.model.identifiers)
        node = random.choice(list(graph.nodes()))
        attention_loss = 0
        t = self._graph_time(graph)
        for step in range(t):
            predictions, node, attention_score = self.model(
                data, graph, features, node)
            target, prediction_loss = gam.calculate_predictive_loss(
                data, predictions)
            batch_loss = batch_loss + prediction_loss
            if step < t - 2:
                attention_loss += (self.args.gamma ** (t - step)) * \
                    torch.log(attention_score)
        reward = gam.calculate_reward(target, predictions)
        batch_loss = batch_loss - reward * attention_loss
        self.model.reset_attention()
        return batch_loss

    def score_split(self, folder: str, split_name: str):
        """单智能体评估：1 次随机起点读取，取最后一步 argmax。"""
        self.model.eval()
        paths = sorted(glob.glob(folder + '*.json'))
        assert paths, f'{folder} 下没有 JSON'
        self.predictions = []
        records = []
        for path in tqdm(paths, desc=f'score {split_name}'):
            data = json.load(open(path))
            graph, features = gam.create_features(
                data, self.model.identifiers)
            node_predictions = []
            for _ in range(self.args.repetitions):
                node = random.choice(list(graph.nodes()))
                for _ in range(self._graph_time(graph)):
                    prediction, node, _ = self.model(data, graph, features, node)
                node_predictions.append(int(np.argmax(prediction.detach())))
                self.model.reset_attention()
            pred = max(set(node_predictions), key=node_predictions.count)
            self.predictions.append(int(pred == data['target']))
            records.append((Path(path).name, pred, data['target']))
        acc = float(np.mean(self.predictions))
        return acc, records


def save_predictions(csv_path: Path, records):
    import pandas as pd
    pd.DataFrame(records, columns=['graph_id', 'predicted_label', 'target']) \
        .to_csv(csv_path, index=None)


def main():
    parser = argparse.ArgumentParser(description='GAM baseline（单智能体）')
    parser.add_argument('--dataset', default='collab')
    parser.add_argument('--data-dir', required=True,
                        help='convert_dataset.py 输出目录（含 train/ valid/ test/）')
    parser.add_argument('--budgets', type=float, nargs='+',
                        default=[0.10, 0.15, 0.20],
                        help='逐图预算比例列表')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='评分重复次数（1=单智能体，对齐我们的设定）')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--time-dims', type=int, default=32,
                        dest='step_dimensions')
    parser.add_argument('--combined-dims', type=int, default=64,
                        dest='combined_dimensions')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=10 ** -5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', default='',
                        help='输出根目录（默认 baseline/output/{dataset}）')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    folders = {s: (data_dir / s).as_posix().rstrip('/') + '/'
               for s in ('train', 'valid', 'test')}
    # cifar10 等 registry 数据集 valid 恒空（评估统一走 test）→ valid 可选
    for name in ('train', 'test'):
        assert glob.glob(folders[name] + '*.json'), \
            f'{folders[name]} 下没有 JSON'
    have_valid = bool(glob.glob(folders['valid'] + '*.json'))
    out_root = (Path(args.output_root) if args.output_root
                else _BASELINE_DIR / 'output' / args.dataset)
    out_root.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(data_dir / 'meta.json')) if (data_dir / 'meta.json').exists() else {}

    summary = []
    for ratio in args.budgets:
        key = ratio_key(ratio)
        budget_dir = out_root / f'budget_{key}'
        budget_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n===== budget={ratio} （单智能体 × {args.repetitions} '
              f'次读取）=====')
        set_seed(args.seed)

        # GAMTrainer 需要的完整参数命名空间（read_node_labels 用
        # train+test 文件夹收集标签词表；无 valid 时直接挂 test）
        trainer_args = argparse.Namespace(
            train_graph_folder=folders['train'],
            test_graph_folder=folders['valid'] if have_valid
            else folders['test'],
            prediction_path=str(budget_dir / 'predictions_test.csv'),
            log_path=str(budget_dir / 'gam_logs.json'),
            time=0,  # 逐图按比例计算（BudgetGAMTrainer）
            epochs=args.epochs, batch_size=args.batch_size,
            step_dimensions=args.step_dimensions,
            combined_dimensions=args.combined_dimensions,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, gamma=args.gamma,
            repetitions=args.repetitions)

        t0 = time.time()
        trainer = BudgetGAMTrainer(trainer_args, budget_ratio=ratio)
        n_identifiers = len(trainer.model.identifiers)
        trainer.fit()
        train_seconds = time.time() - t0

        record = {
            'dataset': args.dataset,
            'baseline': 'GAM (KDD18, single agent)',
            'budget_ratio': ratio,
            'time_rule': 'round(budget_ratio * n_nodes) per graph '
                         '(steps; visited = steps + 1 incl. random start)',
            'repetitions': args.repetitions,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'gamma': args.gamma,
            'step_dimensions': args.step_dimensions,
            'combined_dimensions': args.combined_dimensions,
            'seed': args.seed,
            'n_node_labels': n_identifiers,
            'node_label_mode': meta.get('node_labels', 'unknown'),
            'n_train': len(trainer.training_graphs),
            'train_seconds': round(train_seconds, 1),
        }

        if have_valid:
            valid_acc, valid_records = trainer.score_split(
                folders['valid'], 'valid')
            save_predictions(budget_dir / 'predictions_valid.csv',
                             valid_records)
            record['valid_accuracy'] = valid_acc
        else:
            print('valid: 该数据集无 valid split，跳过（评估走 test）')
            record['valid_accuracy'] = None
        test_acc, test_records = trainer.score_split(folders['test'], 'test')
        save_predictions(budget_dir / 'predictions_test.csv', test_records)
        record['test_accuracy'] = test_acc

        trainer.logs['valid_accuracy'] = record['valid_accuracy']
        trainer.logs['test_accuracy'] = test_acc
        with open(budget_dir / 'gam_logs.json', 'w') as f:
            json.dump(trainer.logs, f, indent=2)
        with open(budget_dir / 'record.json', 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f'budget={ratio}: '
              + (f'valid_acc={valid_acc:.4f} ' if have_valid else '')
              + f'test_acc={test_acc:.4f}（train {train_seconds:.0f}s）')
        print(f'Records: {budget_dir}')
        summary.append(record)

    with open(out_root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('\n===== GAM baseline 汇总 =====')
    for r in summary:
        v = ('valid=%.4f' % r['valid_accuracy']) if \
            r['valid_accuracy'] is not None else 'valid=n/a'
        print(f"  budget={r['budget_ratio']}: {v} "
              f"test={r['test_accuracy']:.4f}")
    print(f'Summary: {out_root / "summary.json"}')


if __name__ == '__main__':
    main()
