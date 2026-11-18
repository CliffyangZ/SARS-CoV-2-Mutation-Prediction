"""在與 PQK 完全相同的 22 類切分上重跑古典基線（Q9，FR-005）。

論文表中的古典基線用 22,154/9,503 筆，PQK 用 22,175/9,514，兩者不可逐列比較。
這裡直接使用 `features.load_features` 回傳的同一組矩陣，結果才能同表並列。
CNN 基線（11 類、單種子）不重跑，報告中標為「不可比」。

只**呼叫**既有的 `Research.lineage_baselines`（該目錄有未提交修改，不改動它）。
"""
import time

import numpy as np

from Research.lineage_baselines import classical
from Research.lineage_baselines.metrics import compute_metrics, per_class_report
from Research.lineage_baselines.results_io import environment_info, result_path, write_result

MODELS = {"lr": "logistic_regression", "rf": "random_forest", "mlp": "mlp"}


def build_model(name, seed):
    """回傳 (估計器, 超參數)。LR 提高 max_iter 並保證可檢查收斂。"""
    if name == "lr":
        from sklearn.linear_model import LogisticRegression

        hp = {"max_iter": 3000, "C": 1.0, "solver": "lbfgs"}
        return LogisticRegression(random_state=seed, **hp), hp
    return classical.build(name, "onehot", seed)


def converged(est):
    """LR：n_iter_ 未達 max_iter 才算收斂；其他模型回傳 None（無此概念）。"""
    if hasattr(est, "n_iter_") and hasattr(est, "max_iter") and est.__class__.__name__ == "LogisticRegression":
        return bool(int(np.max(est.n_iter_)) < int(est.max_iter))
    return None


def fit_eval(name, fs, seed):
    """訓練並在測試集評估，回傳結果字典。"""
    est, hp = build_model(name, seed)
    t0 = time.time()
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        est.fit(fs.x_train, fs.y_train)
    conv = converged(est)
    n_cls = len(fs.labels)
    proba = np.zeros((len(fs.x_test), n_cls))
    p = est.predict_proba(fs.x_test)
    proba[:, np.asarray(est.classes_, np.int64)] = p
    pred = proba.argmax(1)
    res = dict(
        config=dict(name=f"baseline22_{name}", model=name, representation="onehot", n_classes=n_cls,
                    hyperparameters=hp, n_train=int(len(fs.y_train)), n_test=int(len(fs.y_test)),
                    seeds=None, converged=conv),
        seed=seed, split_version=fs.split_version, status="complete",
        metrics_test=compute_metrics(fs.y_test, pred, proba, n_cls),
        per_class_test=per_class_report(fs.y_test, pred, n_cls, fs.labels),
        converged=conv, convergence_warnings=[str(x.message) for x in w][:3],
        seconds={"total": time.time() - t0}, environment=environment_info())
    return res


def run_all(fs, models, seeds, out):
    """對每個模型與種子跑一次並寫結果檔。LR 未收斂時印出警告（FR-005）。"""
    for name in models:
        for seed in seeds:
            res = fit_eval(name, fs, seed)
            res["config"]["seeds"] = list(seeds)
            write_result(result_path(out, f"baseline22_{name}", "mut", seed), res)
            tag = "" if res["converged"] is not False else "  [警告] 未收斂"
            print(f"[baseline22 {name:3s} seed{seed}] macro F1={res['metrics_test']['f1_macro']:.4f} "
                  f"acc={res['metrics_test']['accuracy']:.4f}{tag}", flush=True)
