"""七種傳統分類器 baseline：SVM、Naive Bayes、MLP、KNN、隨機森林、邏輯迴歸、決策樹。

輸入表示有兩種：自編碼器嵌入（論文分組）與 one-hot（對照）。論文未說明這些分類器的
超參數，全部採常見預設並標為 chosen（FR-010）。
"""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import autoencoder as AE
from . import data as D
from . import metrics as M


def build(model, rep, seed):
    """回傳（sklearn 估計器，超參數字典）。嵌入輸入會加標準化。"""
    hp = {}
    if model == "svm":
        hp = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "scores": "decision_function(ovr)"}
        est = SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr", random_state=seed)
    elif model == "nb":
        est = GaussianNB() if rep == "ae_embedding" else BernoulliNB()
        hp = {"variant": type(est).__name__}
    elif model == "mlp":
        hp = {"hidden_layer_sizes": [256, 128], "max_iter": 200, "solver": "adam"}
        est = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=seed)
    elif model == "knn":
        hp = {"n_neighbors": 5, "metric": "euclidean"}
        est = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif model == "rf":
        hp = {"n_estimators": 100}
        est = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    elif model == "lr":
        hp = {"max_iter": 1000, "C": 1.0}
        est = LogisticRegression(max_iter=1000, random_state=seed)
    elif model == "dt":
        est = DecisionTreeClassifier(random_state=seed)
    else:
        raise ValueError(f"未知傳統模型：{model}")
    if rep == "ae_embedding" and model != "nb":
        est = make_pipeline(StandardScaler(), est)
    return est, hp


def _scores(est, x, n_classes):
    """取得 (n, n_classes) 的分數：優先機率，否則決策函數（AUC 只需排序）。"""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(x)
    return est.decision_function(x)


def _features(rep, prep, seed, args, ctx):
    """取得 (train, test) 特徵；同一次執行內以 ctx 快取。"""
    if rep == "onehot":
        if "onehot" not in ctx:
            ctx["onehot"] = (D.one_hot_sparse(prep.x_train), D.one_hot_sparse(prep.x_test))
        return ctx["onehot"], {}
    key = ("ae", seed)
    if key not in ctx:
        t0 = time.time()
        tr, te, loss = AE.fit_embed(prep.x_train, prep.x_test, seed, epochs=getattr(args, "ae_epochs", None))
        ctx[key] = ((tr, te), {"ae_seconds": time.time() - t0, "ae_final_loss": loss})
    return ctx[key]


def run_one(model, rep, seed, prep, args, ctx):
    """訓練並評估一個（模型，表示，種子），回傳結果字典的主體。"""
    (xtr, xte), extra = _features(rep, prep, seed, args, ctx)
    est, hp = build(model, rep, seed)
    t0 = time.time()
    est.fit(xtr, prep.y_train)
    fit_s = time.time() - t0
    n_cls = len(prep.labels)
    pred = est.predict(xte)
    scores = _scores(est, xte, n_cls)
    if scores.shape[1] != n_cls:  # 訓練集缺類別時補零欄
        full = np.zeros((scores.shape[0], n_cls), dtype=scores.dtype)
        full[:, est.classes_ if hasattr(est, "classes_") else est[-1].classes_] = scores
        scores = full
    prov = {k: "chosen" for k in hp}
    prov["algorithm"] = "paper"
    out = {
        "hyperparameters": {**hp, **({"autoencoder": AE.AE_HPARAMS} if rep == "ae_embedding" else {})},
        "provenance": {**prov, **({"autoencoder": AE.AE_PROVENANCE} if rep == "ae_embedding" else {})},
        "metrics": M.compute_metrics(prep.y_test, pred, scores, n_cls),
        "per_class": M.per_class_report(prep.y_test, pred, n_cls, prep.labels),
        "train_seconds": fit_s + extra.get("ae_seconds", 0.0),
        "fit_seconds": fit_s,
        "epochs_done": None,
    }
    out.update(extra)
    return out
