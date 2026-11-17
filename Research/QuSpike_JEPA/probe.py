"""凍結嵌入後的線性 probe（logistic regression，不做類別重新加權）與 kNN。"""
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from Research.lineage_baselines.metrics import compute_metrics, per_class_report


@torch.no_grad()
def embed(model, ts, device="cpu", batch=512, **kw):
    """對完整（不遮蔽）突變集合取嵌入 q 與潛向量 z̃。回傳 (q, z̃) numpy。"""
    model.eval()
    qs, zs = [], []
    for s in range(0, len(ts), batch):
        sl = np.arange(s, min(len(ts), s + batch))
        pos = torch.as_tensor(ts.pos[sl].astype(np.int64), device=device)
        sym = torch.as_tensor(ts.sym[sl].astype(np.int64), device=device)
        val = torch.as_tensor(ts.valid[sl], device=device)
        zt, _ = model.enc(pos, sym, val)
        qs.append(model.pred(zt, **kw).cpu().numpy())
        zs.append(zt.cpu().numpy())
    return np.concatenate(qs), np.concatenate(zs)


def _full_proba(clf, x, n_classes):
    p = clf.predict_proba(x)
    out = np.zeros((len(x), n_classes))
    out[:, clf.classes_] = p
    return out


def probe(f_train, y_train, f_eval_list, y_eval, n_classes, labels, knn_k=5, seed=0):
    """訓練線性 probe 與 kNN，對 f_eval_list 的每組特徵評估。回傳 dict（metrics 為第一組、extra 為其餘）。"""
    sc = StandardScaler().fit(f_train)
    t0 = time.time()
    lr = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(f_train), y_train)
    knn = KNeighborsClassifier(n_neighbors=knn_k, metric="cosine").fit(sc.transform(f_train), y_train)
    res = []
    for f in f_eval_list:
        x = sc.transform(f)
        pr = _full_proba(lr, x, n_classes)
        pk = _full_proba(knn, x, n_classes)
        res.append(dict(
            linear=compute_metrics(y_eval, pr.argmax(1), pr, n_classes),
            knn=compute_metrics(y_eval, pk.argmax(1), pk, n_classes),
            per_class=per_class_report(y_eval, pr.argmax(1), n_classes, labels),
        ))
    return res, time.time() - t0
