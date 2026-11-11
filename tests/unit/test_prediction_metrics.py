"""最小 Hamming、新位置命中率、頻繁位點召回率。"""
import numpy as np
from Research.utils.prediction_metrics import frequent_recall, min_hamming, novel_hit_rate


def test_min_hamming():
    A = np.array([[0, 1, 2, 3], [0, 0, 0, 0]]); B = np.array([[0, 1, 2, 0], [5, 5, 5, 5]])
    assert min_hamming(A, B).tolist() == [1, 2]   # 第二列到 B 第一列差 2 個位置


def test_novel_hit_rate():
    train_freq = np.array([0.5, 0, 0, 0]); test_mut = np.array([True, True, False, False])
    assert novel_hit_rate([[2, 3]], train_freq, test_mut) == 0.5      # 位置 2、3（1 起算）皆新；僅位置 2 在測試期出現
    assert novel_hit_rate([[1]], train_freq, test_mut) is None        # 沒有新位置 → 不適用


def test_frequent_recall():
    test_freq = np.array([0.5, 0.02, 0.0, 0.5])
    assert frequent_recall([1], test_freq, candidates=[1, 2, 3]) == 0.5   # 頻繁位點 {1,2}（4 不在候選）→ 命中 1/2
    assert frequent_recall([1], np.zeros(4), [1]) is None
