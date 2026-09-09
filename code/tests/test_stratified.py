"""stratified_take 单测：分层子采样（registry）的确定性 / 比例性 / 边界。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.registry import stratified_take


def test_proportional_and_exact_n():
    """3 类不均衡数据按比例抽取，总数恰为 n。"""
    # id 0..99，前 60 个类 0 / 中 30 个类 1 / 后 10 个类 2 → n=20 应抽 12/6/2
    labels = [0] * 60 + [1] * 30 + [2] * 10
    sel = stratified_take(range(100), labels, 20)
    assert len(sel) == 20 and sel == sorted(sel)
    from collections import Counter
    cnt = Counter(labels[i] for i in sel)
    assert cnt == {0: 12, 1: 6, 2: 2}, cnt


def test_deterministic():
    """同种子两次调用结果一致；不同 seed 结果（几乎必然）不同。"""
    labels = [i % 5 for i in range(200)]
    a = stratified_take(range(200), labels, 50)
    b = stratified_take(range(200), labels, 50)
    assert a == b
    c = stratified_take(range(200), labels, 50, seed=1)
    assert a != c


def test_callable_label_of():
    """label_of 支持函数形式（PPA：全局索引 → 全量标签张量）。"""
    ids = list(range(30))
    sel = stratified_take(ids, lambda i: i % 3, 9)
    assert len(sel) == 9
    from collections import Counter
    cnt = Counter(i % 3 for i in sel)   # 30 个 id 均分 3 类 → 每类恰 3 个
    assert cnt == {0: 3, 1: 3, 2: 3}, cnt


def test_edge_cases():
    """n>=len / n<=0 原样返回全部（升序）。"""
    ids = [5, 3, 1, 4, 2]
    assert stratified_take(ids, [1, 0, 1, 0, 1], 5) == [1, 2, 3, 4, 5]
    assert stratified_take(ids, [1, 0, 1, 0, 1], 99) == [1, 2, 3, 4, 5]
    assert stratified_take(ids, [1, 0, 1, 0, 1], 0) == [1, 2, 3, 4, 5]
    # n 小于类别数：最大余数法保证每类至少有机会、总数恰为 n
    sel = stratified_take(range(10), [i % 4 for i in range(10)], 2)
    assert len(sel) == 2


def test_tiny_class_not_dropped():
    """极小类在比例配额为 0 时仍按余数法参与补足（不保证入选但总数为 n）。"""
    labels = [0] * 99 + [1]
    sel = stratified_take(range(100), labels, 10)
    assert len(sel) == 10 and sel == sorted(sel)
    from collections import Counter
    cnt = Counter(labels[i] for i in sel)
    assert cnt[0] == 10 or cnt == {0: 9, 1: 1}


if __name__ == '__main__':
    for fn in [test_proportional_and_exact_n, test_deterministic,
               test_callable_label_of, test_edge_cases,
               test_tiny_class_not_dropped]:
        fn()
        print(f'  {fn.__name__}: ok')
    print('test_stratified: all passed')
