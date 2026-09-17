"""全图信息天花板（推论 3.3 第三层，A2，校准假设版）。

原理（数学分析文档第 3 章推论 3.3，含 2026-09-07 修正）：任意预算 B 的
读取策略满足 Fano 下界 err ≥ f^{-1}(H(Y|S_B))，f(e) = h(e) + e·ln(K−1)
在 [0, 1−1/K] 单调增。链 H(Y|S_B) ≥ H(Y) − I(Y;G) 要变成可计算的下界，
需对 I(Y;G) 取**上界**；分类器 CE 只能给 MI 下界（BA），直接代入方向
相反、无凭。修正：在显式校准假设

    KL(q ‖ P(Y|G)) ≤ KL(U ‖ P_Y)     （q = 分类器，U = 均匀分布）

下（经验上即 q 不过度自信、ceiling ≥ 一切已测曲线可检验），

    T := H(Y) − Î = CE − (ln K − H(Y)) ≤ H(Y|G)，
    故 err ≥ f^{-1}(H(Y|S_B)) ≥ f^{-1}(H(Y|G)) ≥ f^{-1}(T)，ceiling = 1 − f^{-1}(T)。

T = H − Î = CE − (lnK − H(Y))（平衡数据集 H=lnK 时 T=CE）。若违反假设
（天花板低于某条已测曲线，尤其低于全图参照），说明分类器失准——先重校准
再报此界。严格无假设的 Fano 天花板需要对 I(Y;G) 的上界，属第二层
（生成模型内可算）的职责。

用法:
  python code/comparison/info_ceiling.py --root D:\\data\\superpixels \
      --checkpoint result/cifar10/classifier/GIN-plus/best_cifar10_classifier.pt
  # dataset 缺省从 checkpoint 内嵌 config 推断；PPA 大测试集建议 --max-graphs
"""

import argparse
import datetime
import json
import math
import sys
from collections import Counter
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
import dgl  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from classifier import load_classifier  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import resolve  # noqa: E402


def h_nat(e: float) -> float:
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return -(e * math.log(e) + (1 - e) * math.log(1 - e))


def fano_fn(e: float, k: int) -> float:
    """f(e) = h(e) + e·ln(K−1)，nats；f 在 [0, 1−1/K] 单调增。"""
    return h_nat(e) + e * math.log(k - 1) if k > 2 else h_nat(e)


def invert_fano(target: float, k: int) -> float:
    """解 f(e) = target，e ∈ [0, 1−1/K]，二分（f 单调增）。"""
    lo, hi = 0.0, 1.0 - 1.0 / k
    if target <= 0.0:
        return 0.0
    if target >= fano_fn(hi, k):  # 理论上 = ln K ≥ H(Y) ≥ target，防御
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fano_fn(mid, k) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@torch.no_grad()
def evaluate(clf, graphs, labels, device, batch_size=256):
    """→ (acc, ce_mean, label_counts)。CE 为自然对数 nats。"""
    clf.eval()
    ce_sum, correct, total = 0.0, 0, 0
    counts = Counter()
    for start in range(0, len(graphs), batch_size):
        idx = range(start, min(start + batch_size, len(graphs)))
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = labels[start:start + len(idx)].to(device)
        logits = clf(batch_g, nids)
        ce_sum += float(F.cross_entropy(logits, y, reduction='sum').item())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += len(idx)
        counts.update(y.tolist())
    return correct / max(total, 1), ce_sum / max(total, 1), counts


@torch.no_grad()
def collect_logits(clf, graphs, labels, device, batch_size=256):
    """→ (logits[N,K] cpu, labels[N])。温度校准用（需逐样本 logits）。"""
    clf.eval()
    chunks, ys = [], []
    for start in range(0, len(graphs), batch_size):
        idx = range(start, min(start + batch_size, len(graphs)))
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        chunks.append(clf(batch_g, nids).float().cpu())
        ys.append(labels[start:start + len(idx)])
    return torch.cat(chunks), torch.cat(ys)


def nll_at(logits: torch.Tensor, y: torch.Tensor, tau: float) -> float:
    return float(F.cross_entropy(logits / tau, y, reduction='mean').item())


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """在 calib split 上最小化 NLL 的 τ（一维凸，粗网格 + 黄金分割细化）。"""
    grid = [0.2 * t for t in range(1, 51)]          # 0.2 .. 10.0
    best = min(grid, key=lambda t: nll_at(logits, y, t))
    a, b = max(best - 0.2, 1e-3), best + 0.2
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if nll_at(logits, y, c) < nll_at(logits, y, d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    return 0.5 * (a + b)


def main():
    parser = argparse.ArgumentParser(description='全图信息天花板（紧 Fano 反演）')
    parser.add_argument('--dataset', default='',
                        help='数据集名（默认从 checkpoint 内嵌 config 推断）')
    parser.add_argument('--root', required=True, help='数据集根目录')
    parser.add_argument('--checkpoint', required=True, help='全图分类器 checkpoint')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'],
                        help='口径 split（默认 test，与 reader eval 对齐）')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='split 图数上限（0=全部；PPA 全量建议设 2 万级采样）')
    parser.add_argument('--fit-temperature', action='store_true',
                        help='在校准 split 上拟合温度 τ 后重报天花板'
                             '（校准假设被违反时的标准补救，见 A.6）')
    parser.add_argument('--calib-split', default='valid', choices=['train', 'valid', 'test'],
                        help='温度拟合 split（默认 valid，禁用 test 防泄漏）')
    parser.add_argument('--calib-max-graphs', type=int, default=0,
                        help='校准 split 图数上限（0=全部；单参数拟合无需全量）')
    parser.add_argument('--calib-stratified', action='store_true',
                        help='校准 split 按类别比例分层抽样（默认前 N 截断）')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='',
                        help='默认 result/{ds}/comparison/')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt_path = resolve(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    ds_name = args.dataset or (ckpt.get('config') or {}).get('dataset', '')
    if not ds_name:
        raise SystemExit('无法确定数据集：checkpoint 无内嵌 config.dataset，请传 --dataset')

    ds = get_dataset(ds_name, root=args.root, splits=[args.split],
                     max_graphs=args.max_graphs)
    graphs = getattr(ds, f'{args.split}_graphs')
    labels = getattr(ds, f'{args.split}_labels')
    if not len(graphs):
        raise SystemExit(f"split '{args.split}' 为空")

    clf = load_classifier(str(ckpt_path), device=device, freeze=True)
    acc, ce, counts = evaluate(clf, graphs, labels, device, args.batch_size)

    k = ds.spec.n_classes
    n = len(graphs)
    p = torch.tensor([counts.get(c, 0) for c in range(k)], dtype=torch.float64)
    p = p / p.sum()
    h_y = float(-(p * p.clamp_min(1e-12).log()).sum())
    mi_lower = math.log(k) - ce                 # Î(Y;G) = ln K − CE（下界）
    target = h_y - mi_lower                     # T = H(Y) − Î
    e_tight = invert_fano(target, k)
    e_coarse = (target - 1.0) / math.log(k)     # 粗形式，仅供对照（常为负）

    print(f"[{ds_name}/{args.split}] n={n}  K={k}")
    print(f"  test acc = {acc:.4f}（与训练记录交叉核对见 history.json）")
    print(f"  CE_test  = {ce:.4f} nats")
    print(f"  H(Y)     = {h_y:.4f} nats  (label dist: "
          + ', '.join(f'{c}:{counts.get(c, 0)}' for c in range(k)) + ')')
    print(f"  Î(Y;G)   = ln K − CE = {mi_lower:.4f} nats  "
          f"(MI 下界仅参考；界限经校准假设 KL(q‖P) ≤ KL(U‖P_Y) 成立)")
    print(f"  T = H − Î = {target:.4f} nats  (校准假设下 ≤ H(Y|G))")
    print(f"  ceiling(calibrated-Fano) = 1 − f^{{-1}}(T) = {1 - e_tight:.4f}")
    print(f"  ceiling(loose, 同假设)   = {1 - e_coarse:.4f}"
          + ('   [负值 → 粗形式失效，用 tight]' if e_coarse < 0 else ''))
    if 1 - e_tight < acc:
        print(f"  ⚠ ceiling({1 - e_tight:.4f}) < 分类器自身 acc({acc:.4f})："
              f"校准假设被违反（KL(q‖P) 过大），此界无效——先重校准/换分类器")
    if 1 - e_tight > 1.0 or e_tight < 0:
        print("  ⚠ 数值越界，检查 CE/H 计算")

    temp_out = {}
    if args.fit_temperature:
        calib = get_dataset(ds_name, root=args.root, splits=[args.calib_split],
                            max_graphs=args.calib_max_graphs,
                            stratified=args.calib_stratified)
        calib_graphs = getattr(calib, f'{args.calib_split}_graphs')
        calib_labels = getattr(calib, f'{args.calib_split}_labels')
        test_logits, test_y = collect_logits(clf, graphs, labels, device,
                                             args.batch_size)
        cal_logits, cal_y = collect_logits(clf, calib_graphs, calib_labels,
                                           device, args.batch_size)
        tau = fit_temperature(cal_logits, cal_y)
        ce_tau = nll_at(test_logits, test_y, tau)
        target_tau = h_y - (math.log(k) - ce_tau)
        e_tau = invert_fano(target_tau, k)
        ceil_tau = 1 - e_tau
        print(f"  [温度校准] τ = {tau:.4f}（{args.calib_split} 拟合）  "
              f"CE_test(τ) = {ce_tau:.4f} nats  T(τ) = {target_tau:.4f}")
        print(f"  ceiling(calibrated-Fano, τ) = {ceil_tau:.4f}"
              + ('' if ceil_tau >= acc else
                 f"   ⚠ 仍 < acc({acc:.4f})，校准假设依旧被违反"))
        temp_out = {
            'temperature': round(tau, 6),
            'calib_split': args.calib_split,
            'ce_test_nats_calibrated': round(ce_tau, 6),
            'target_nats_calibrated': round(target_tau, 6),
            'ceiling_calibrated_fano_temperature': round(ceil_tau, 6),
        }

    out = {
        'dataset': ds_name, 'split': args.split, 'n': n, 'K': k,
        'test_acc': round(acc, 6),
        'ce_test_nats': round(ce, 6),
        'H_Y_nats': round(h_y, 6),
        'label_counts': {str(c): counts.get(c, 0) for c in range(k)},
        'mi_lower_nats': round(mi_lower, 6),
        'target_nats': round(target, 6),
        'ceiling_calibrated_fano': round(1 - e_tight, 6),
        'ceiling_loose_same_assumption': round(1 - e_coarse, 6),
        **temp_out,
        'config': {
            'checkpoint': str(ckpt_path),
            'max_graphs': args.max_graphs,
            'batch_size': args.batch_size,
            'note': '校准假设版 Fano 天花板：KL(q‖P(Y|G)) ≤ KL(U‖P_Y) 时对一切'
                    '读取策略有效；违反的信号是天花板低于某条已测曲线。换更强'
                    '分类器/重校准后须重算。推导见数学分析文档第 3 章推论 3.3'
                    '（2026-09-07 修正版）。',
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        },
    }
    out_dir = (resolve(args.output_dir) if args.output_dir
               else Path('result') / ds_name / 'comparison')
    out_dir = resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'info_ceiling.json'
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
