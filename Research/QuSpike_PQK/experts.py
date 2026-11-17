"""群內專家與 logits 合併（T022–T023）。

結構下限（FR-009）：最終 logits = 參考線 logits + Σ_g 1[樣本屬群 g] · 專家修正。
專家貢獻為零時整個模型還原成參考線，所以「追不上參考線」不能歸因於優化難度。

易分群不配專家，其預測因此與 C0 **逐筆完全相同**（SC-004）。
"""
import numpy as np

from . import kernels as K

MEM_BUDGET_BYTES = 2.0e9  # 超過則改用 Nyström（delta 群精確核為 364 MB，遠低於此）


def _fit_kernel_svm(k_train, y, seed=0, C=10.0):
    from sklearn.svm import SVC

    m = SVC(C=C, kernel="precomputed", decision_function_shape="ovr", random_state=int(seed))
    m.fit(np.asarray(k_train, np.float64), np.asarray(y, np.int64))
    return m


def _fit_mlp(x, y, hidden, seed=0, max_iter=1200):
    from sklearn.neural_network import MLPClassifier

    m = MLPClassifier(hidden_layer_sizes=(int(hidden),), activation="tanh",
                      random_state=int(seed), max_iter=int(max_iter))
    m.fit(np.asarray(x, np.float64), np.asarray(y, np.int64))
    return m


def _decision_scores(model, x):
    """統一取分數：SVM 用 decision_function，MLP 沒有這個方法，改用 log 機率。"""
    if hasattr(model, "decision_function"):
        return model.decision_function(np.asarray(x, np.float64))
    lp = model.predict_log_proba(np.asarray(x, np.float64))
    return lp - lp.mean(1, keepdims=True)  # 中心化，使其可直接加到參考線 logits 上


def _scores_to_logit_block(model, scores, class_ids, n_classes, n_rows, scale):
    """把專家的類別分數攤回 22 類的 logits 區塊（只動該群的類別）。"""
    out = np.zeros((n_rows, n_classes), np.float64)
    cols = np.asarray(model.classes_, np.int64)
    s = np.asarray(scores, np.float64)
    if s.ndim == 1:  # 二類時 decision_function 回傳 1 維
        s = np.stack([-s, s], 1)
    out[:, cols] = scale * s
    return out


def fit_group_expert(kind, phi_tr, phi_ev, y_tr, class_ids, n_classes, n_ev,
                     gamma=None, seed=0, scale=1.0, hidden=None):
    """訓練單一群的專家，回傳要加到參考線上的 logits 區塊與診斷。

    kind='kernel'：precomputed-kernel SVM（PQK／古典 RBF／RFF 都走這條）
    kind='mlp'   ：寬度匹配 MLP（對照組 C4）
    """
    diag = {}
    if kind == "mlp":
        model = _fit_mlp(phi_tr, y_tr, hidden or phi_tr.shape[1], seed=seed)
        s = _decision_scores(model, phi_ev)
        return _scores_to_logit_block(model, s, class_ids, n_classes, n_ev, scale), diag

    g = float(gamma) if gamma else K.auto_gamma(phi_tr)
    diag["gamma"] = g
    need = len(phi_tr) ** 2 * 8
    if need > MEM_BUDGET_BYTES:
        diag["approximation"] = "nystrom"
        k_tr = K.nystrom_kernel(phi_tr, phi_tr, g, seed=seed)
        k_ev = K.nystrom_kernel(phi_tr, phi_ev, g, seed=seed)
    else:
        diag["approximation"] = "exact"
        k_tr = K.rbf(phi_tr, None, g)
        k_ev = K.rbf(phi_ev, phi_tr, g)
    k_tr, psd = K.ensure_psd(k_tr)
    diag.update(psd)
    diag.update(K.diagnostics(k_tr))
    model = _fit_kernel_svm(k_tr, y_tr, seed=seed)
    s = _decision_scores(model, k_ev)
    return _scores_to_logit_block(model, s, class_ids, n_classes, n_ev, scale), diag


def combine(ref_logits, blocks):
    """參考線 logits + 各群專家區塊。blocks: [(列索引, logits 區塊)]。"""
    out = np.array(ref_logits, np.float64, copy=True)
    for rows, block in blocks:
        if len(rows):
            out[rows] += block
    return out
