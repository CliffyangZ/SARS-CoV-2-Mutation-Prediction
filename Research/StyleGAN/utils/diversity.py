"""生成多樣性指標與模式塌縮警示（002 R6，FR-001、FR-001a、FR-002）。

突變集一律用 (n, k_max) 的索引陣列表示，突變數不足處以 -1 填充（-1 不是位點，一律忽略）。
純函式、不依賴模型，訓練中與評估時共用。
"""
import numpy as np


def _sets(idx):
    """每列有效索引 → frozenset 清單（順序不影響集合相等）。"""
    idx = np.asarray(idx)
    return [frozenset(int(i) for i in row if i >= 0) for row in idx]


def distinct_sets(idx):
    """相異突變集數（空集合也算一種）。"""
    return len(set(_sets(idx)))


def position_coverage(idx):
    """被使用到的相異位點數。"""
    idx = np.asarray(idx)
    return int(len(np.unique(idx[idx >= 0])))


def mutation_count_hist(idx):
    """突變數分布 {突變數: 樣本數}。"""
    ks, cnt = np.unique(np.asarray(idx).__ge__(0).sum(1), return_counts=True)
    return {int(k): int(c) for k, c in zip(ks, cnt)}


def position_freq(idx, L):
    """各位點的突變頻率（出現次數 / 樣本數），長度 L。"""
    idx = np.asarray(idx)
    f = np.bincount(idx[idx >= 0].ravel(), minlength=L).astype(float)
    return f[:L] / max(len(idx), 1)


def freq_correlation(idx, ref_freq):
    """生成的位點頻率與參考頻率（訓練期）的 Pearson 相關。任一邊為常數時無定義，回傳 nan。"""
    ref = np.asarray(ref_freq, float)
    g = position_freq(idx, len(ref))
    if g.std() == 0 or ref.std() == 0:
        return float("nan")
    return float(np.corrcoef(g, ref)[0, 1])


def vectors_to_idx(X):
    """0/1 向量 (n,L) → 以 -1 填充的索引陣列 (n,k_max)（與生成結果同格式，供對照集使用）。"""
    X = np.asarray(X)
    kmax = int(X.sum(1).max()) if len(X) else 0
    out = np.full((len(X), kmax), -1, dtype=int)
    for r in range(len(X)):
        pos = np.where(X[r] > 0)[0]
        out[r, :len(pos)] = pos
    return out


def collapse_warning(distinct, n, threshold=None):
    """相異突變集數低於門檻視為疑似模式塌縮。門檻預設 max(10, 1% n)。回傳 (是否警示, 門檻)。"""
    thr = max(10, int(round(0.01 * n))) if threshold is None else threshold
    return bool(distinct < thr), thr


def report(idx, L, train_freq, threshold=None):
    """一組突變集的完整多樣性摘要（寫入 diversity.json 的單一區塊）。"""
    n = len(idx)
    d = distinct_sets(idx)
    warn, thr = collapse_warning(d, n, threshold)
    cov = position_coverage(idx)
    ks = np.asarray(idx).__ge__(0).sum(1) if np.asarray(idx).size else np.zeros(n, int)
    corr = freq_correlation(idx, train_freq)
    return dict(n=int(n), distinct_mutation_sets=d, position_coverage=cov, position_coverage_frac=cov / L,
                mutation_count_hist=mutation_count_hist(idx), mean_mutation_count=float(ks.mean()) if n else 0.0,
                empty_sets=int((ks == 0).sum()), freq_correlation=None if np.isnan(corr) else corr,
                collapse_warning=warn, warning_threshold=thr)
