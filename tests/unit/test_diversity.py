"""多樣性指標（002 T009）：相異突變集、位點覆蓋、頻率相關、塌縮警示。"""
import numpy as np
from Research.StyleGAN.utils.diversity import (collapse_warning, distinct_sets, freq_correlation, mutation_count_hist,
                                      position_coverage, vectors_to_idx)


def test_distinct_sets_ignores_order_and_padding():
    idx = np.array([[3, 5], [5, 3], [3, -1], [-1, -1], [3, 5]])
    assert distinct_sets(idx) == 3            # {3,5}、{3}、{} 三種；順序不同視為同一集合
    assert position_coverage(idx) == 2        # 只算 3、5，-1 是填充
    assert mutation_count_hist(idx) == {0: 1, 1: 1, 2: 3}


def test_freq_correlation_identical_and_constant():
    freq = np.array([0.5, 0.3, 0.1, 0.0])
    rng = np.random.default_rng(0)
    # 以 freq 為機率抽很多樣本 → 相關應接近 1
    idx = vectors_to_idx((rng.random((5000, 4)) < freq).astype(int))
    assert freq_correlation(idx, freq) > 0.95
    # 所有樣本都只有位點 0 → 頻率向量為 [1,0,0,0]，不是常數，但與 freq 相關仍為正且 < 1
    const = np.zeros((10, 1), dtype=int)
    assert 0 < freq_correlation(const, freq) < 1
    # 全部是空集合 → 生成頻率為常數 0，相關無定義，約定回傳 nan
    assert np.isnan(freq_correlation(np.full((5, 2), -1), freq))


def test_vectors_to_idx_roundtrip():
    x = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    idx = vectors_to_idx(x)
    assert idx.shape == (3, 2) and distinct_sets(idx) == 3
    assert sorted(idx[0][idx[0] >= 0]) == [1, 2] and (idx[1] == -1).all()


def test_collapse_warning_threshold_boundaries():
    warn, thr = collapse_warning(distinct=3, n=1000)
    assert warn and thr == 10                 # 預設門檻 max(10, 1% n)
    assert not collapse_warning(10, 1000)[0] and collapse_warning(9, 1000)[0]
    assert collapse_warning(50, 10000)[1] == 100 and collapse_warning(50, 10000)[0]
    assert collapse_warning(5, 1000, threshold=5)[0] is False
