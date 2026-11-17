"""正樣本配對：同譜系、親緣鄰近（突變距離＋日期窗，不用標籤）、單序列破壞。"""
import numpy as np

from .dataset import N_SYM

VARIANTS = {"VA": "VA", "same_lineage": "VA", "VB": "VB", "neighbor": "VB", "VC": "VC", "self_distort": "VC",
            "VD": "VD", "seq_nuisance": "VD"}


def canonical_variant(name):
    """接受 VA/VB/VC/VD 或 same_lineage/neighbor/self_distort/seq_nuisance。"""
    if name not in VARIANTS:
        raise ValueError(f"未知的正樣本變體：{name}")
    return VARIANTS[name]


def lineage_groups(y):
    """每個譜系的序列索引。"""
    return {int(k): np.where(y == k)[0] for k in np.unique(y)}


def neighbor_lists(ts, max_dist=4, max_months=1, cap=50, seed=0, chunk=1000):
    """每條序列的親緣鄰居（突變集合對稱差 ≤ max_dist 且採集月份差 ≤ max_months；無日期者無鄰居）。

    回傳 list[np.ndarray]；每條最多保留 cap 個（隨機抽）。
    """
    import scipy.sparse as sp

    n = len(ts)
    rows, cols = np.nonzero(ts.valid)
    tok = ts.pos[rows, cols].astype(np.int64) * N_SYM + ts.sym[rows, cols].astype(np.int64)
    S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, tok)), shape=(n, int(tok.max()) + 1 if len(tok) else 1))
    size = np.asarray(S.sum(1)).ravel()
    rng = np.random.default_rng(seed)
    out = []
    ST = S.T.tocsr()
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        inter = (S[s:e] @ ST).toarray()
        dist = size[s:e, None] + size[None, :] - 2 * inter
        ok = (dist <= max_dist) & (np.abs(ts.month[s:e, None] - ts.month[None, :]) <= max_months)
        ok &= (ts.month[s:e, None] >= 0) & (ts.month[None, :] >= 0)
        ok[np.arange(e - s), np.arange(s, e)] = False
        for r in range(e - s):
            c = np.where(ok[r])[0]
            if len(c) > cap:
                c = rng.choice(c, cap, replace=False)
            out.append(c)
    return out
