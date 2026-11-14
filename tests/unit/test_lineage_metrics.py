"""指標計算與 sklearn 對照的單元測試（FR-005、D7）。"""
import numpy as np
from sklearn import metrics as skm

from Research.lineage_baselines import metrics as M


def _toy():
    """三類、含不平衡的小型預測。"""
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=200, p=[0.6, 0.3, 0.1])
    logits = rng.normal(size=(200, 3)) + 2.0 * np.eye(3)[y]
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, p.argmax(1), p


def test_matches_sklearn():
    """accuracy、weighted 與 macro 的 P／R／F1、AUC 與 sklearn 一致。"""
    y, pred, proba = _toy()
    m = M.compute_metrics(y, pred, proba, n_classes=3)
    assert np.isclose(m["accuracy"], skm.accuracy_score(y, pred))
    assert np.isclose(m["precision_weighted"], skm.precision_score(y, pred, average="weighted"))
    assert np.isclose(m["f1_weighted"], skm.f1_score(y, pred, average="weighted"))
    assert np.isclose(m["f1_macro"], skm.f1_score(y, pred, average="macro"))
    assert np.isclose(m["roc_auc_ovr"], skm.roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def test_weighted_recall_equals_accuracy():
    """weighted recall 依定義等於 accuracy。"""
    y, pred, proba = _toy()
    m = M.compute_metrics(y, pred, proba, n_classes=3)
    assert np.isclose(m["recall_weighted"], m["accuracy"])


def test_missing_class_skipped_and_flagged():
    """test 缺少某類別時，略過該類別 AUC 並標示，不報錯。"""
    y, pred, proba = _toy()
    keep = y != 2
    m = M.compute_metrics(y[keep], pred[keep], proba[keep], n_classes=3)
    assert 2 in m["roc_auc_skipped_classes"]
    assert np.isfinite(m["roc_auc_ovr"])


def test_per_class_structure():
    """每類別 precision／recall／F1／支援數。"""
    y, pred, proba = _toy()
    pc = M.per_class_report(y, pred, n_classes=3, labels=["a", "b", "c"])
    assert [r["lineage"] for r in pc] == ["a", "b", "c"]
    assert sum(r["support"] for r in pc) == len(y)
