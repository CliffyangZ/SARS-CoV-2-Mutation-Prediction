"""分類指標：accuracy、precision／recall／F1（weighted 與 macro）、ROC AUC（one-vs-rest）。

定義依研究文件 D7：precision／recall 以 weighted 為主（weighted recall 等於 accuracy），
ROC AUC 為各類別 one-vs-rest 的 macro 平均；test 中缺少的類別略過並標示。
"""
import numpy as np
from sklearn import metrics as skm


def compute_metrics(y_true, y_pred, proba, n_classes):
    """計算全部指標並回傳字典；proba 為 (n, n_classes) 的機率。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"accuracy": float(skm.accuracy_score(y_true, y_pred))}
    for avg in ("weighted", "macro"):
        p, r, f, _ = skm.precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
    aucs, skipped = [], []
    for k in range(n_classes):
        pos = y_true == k
        if pos.sum() == 0 or (~pos).sum() == 0:
            skipped.append(k)
            continue
        aucs.append(skm.roc_auc_score(pos, proba[:, k]))
    out["roc_auc_ovr"] = float(np.mean(aucs)) if aucs else float("nan")
    out["roc_auc_skipped_classes"] = skipped
    return out


def per_class_report(y_true, y_pred, n_classes, labels):
    """每個譜系的 precision、recall、F1 與支援數。"""
    p, r, f, s = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    return [
        {"lineage": labels[k], "precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k])}
        for k in range(n_classes)
    ]
