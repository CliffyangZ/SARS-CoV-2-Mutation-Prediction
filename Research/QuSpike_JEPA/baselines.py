"""簡單基準：突變集合最近鄰（餘弦），隨機初始化編碼器由 run.py 以 epochs=0 取得。"""
import numpy as np
import scipy.sparse as sp

from .dataset import N_SYM


def _sparse(ts):
    rows, cols = np.nonzero(ts.valid)
    tok = ts.pos[rows, cols].astype(np.int64) * N_SYM + ts.sym[rows, cols].astype(np.int64)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, tok)), shape=(len(ts), 29_500))


def nearest_mutset(train, eval_ts, n_classes, chunk=1000):
    """對每個評估序列取訓練集中餘弦最相近者的標籤；回傳 (預測, one-hot 機率)。"""
    A, B = _sparse(eval_ts), _sparse(train)
    na = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) + 1e-9
    nb = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-9
    BT = B.T.tocsr()
    pred = np.zeros(len(eval_ts), int)
    for s in range(0, len(eval_ts), chunk):
        sim = (A[s:s + chunk] @ BT).toarray() / (na[s:s + chunk, None] * nb[None, :])
        pred[s:s + chunk] = train.y[sim.argmax(1)]
    proba = np.eye(n_classes)[pred]
    return pred, proba


def mutset_linear(train, eval_ts, n_classes, seed=0, device="cpu", epochs=60, wd=1e-4):
    """監督式參考：對齊突變 token（不含 X）的 0/1 向量＋softmax 線性分類（無預訓練、無量子）。回傳 (預測, 機率)。"""
    import torch
    from torch import nn

    from .mutations import UNKNOWN

    keep = train.valid & (train.sym != UNKNOWN)
    vocab = np.unique(train.pos[keep].astype(np.int64) * N_SYM + train.sym[keep].astype(np.int64))

    def feats(ts):
        """把 TokenSet 轉成對齊後的 0/1 突變指標矩陣。"""
        m = ts.valid & (ts.sym != UNKNOWN)
        rows, cols = np.nonzero(m)
        tok = ts.pos[rows, cols].astype(np.int64) * N_SYM + ts.sym[rows, cols].astype(np.int64)
        j = np.searchsorted(vocab, tok)
        ok = (j < len(vocab)) & (vocab[np.minimum(j, len(vocab) - 1)] == tok)
        F = np.zeros((len(ts), len(vocab)), np.float32)
        F[rows[ok], j[ok]] = 1.0
        return torch.as_tensor(F, device=device)

    torch.manual_seed(int(seed))
    Xtr, Xte = feats(train), feats(eval_ts)
    y = torch.as_tensor(np.asarray(train.y, np.int64), device=device)
    m = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=g).to(device)
        for s in range(0, len(y), 512):
            b = perm[s:s + 512]
            opt.zero_grad()
            nn.functional.cross_entropy(m(Xtr[b]), y[b]).backward()
            opt.step()
    with torch.no_grad():
        proba = torch.softmax(m(Xte), 1).cpu().numpy()
    return proba.argmax(1), proba
