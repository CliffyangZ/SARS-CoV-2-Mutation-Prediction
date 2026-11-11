"""四種基準（FR-019，research R10）：參考序列、隨機位置、依訓練期頻率、依訓練期趨勢外推。

皆回傳 (n,k) 的位置索引（L 座標），只使用訓練期資料；reference 回傳 (n,0)（零突變）。
"""
import numpy as np


def reference(n, k, L, rng):
    """基準 1：參考序列本身，不含任何突變。"""
    return np.zeros((n, 0), dtype=int)


def random_positions(n, k, L, rng):
    """基準 2：在 L 個選定位點上隨機取 k 個不重複位置。"""
    return np.stack([rng.choice(L, k, replace=False) for _ in range(n)])


def weighted_positions(n, k, L, rng, weights):
    """依權重（機率正比）抽 k 個不重複位置。"""
    p = np.asarray(weights, float) + 1e-12
    p = p / p.sum()
    return np.stack([rng.choice(L, k, replace=False, p=p) for _ in range(n)])


def by_frequency(n, k, L, rng, train_freq):
    """基準 3：依訓練期加權突變頻率抽樣。"""
    return weighted_positions(n, k, L, rng, train_freq)


def by_trend(n, k, L, rng, pred_freq):
    """基準 4：依訓練期頻率隨時間的線性趨勢，外推到測試期起點的預測頻率抽樣。"""
    return weighted_positions(n, k, L, rng, np.clip(pred_freq, 0, None))


def all_baselines(n, k, L, seed, train_freq, pred_freq):
    """回傳 {名稱: (n,k) 索引}；各基準用獨立固定種子。"""
    mk = lambda off: np.random.default_rng(seed * 100 + off)
    return {"reference": reference(n, k, L, mk(1)), "random": random_positions(n, k, L, mk(2)),
            "freq": by_frequency(n, k, L, mk(3), train_freq), "trend": by_trend(n, k, L, mk(4), pred_freq)}
