"""群內專家與 logits 合併（T022–T023）。

結構下限（FR-009）：最終 logits = 參考線 logits + Σ_g 權重_g · 專家修正。
專家貢獻為零時整個模型還原成參考線，所以「追不上參考線」不能歸因於優化難度。

易分群不配專家，其預測因此與 C0 **逐筆完全相同**（SC-004）。

008 變更：擬合與評分拆開（`fit_expert` → `Expert.scores`），使推論時能對**任意列**評分，
而不是在擬合當下就綁死「只對真實群別的列」（FR-001，Q1）。
"""
import numpy as np

from . import kernels as K

MEM_BUDGET_BYTES = 2.0e9  # 精確核的訓練 Gram 超過此值則改用 Nyström（見 008 US4）
SCORE_CHUNK = 2048


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


def _scores_to_logit_block(model, scores, n_classes, n_rows, scale=1.0):
    """把專家的類別分數攤回 22 類的 logits 區塊（只動該群的類別）。"""
    out = np.zeros((n_rows, n_classes), np.float64)
    cols = np.asarray(model.classes_, np.int64)
    s = np.asarray(scores, np.float64)
    if s.ndim == 1:  # 二類時 decision_function 回傳 1 維
        s = np.stack([-s, s], 1)
    out[:, cols] = scale * s
    return out


class Expert:
    """已擬合的群內專家。`scores(phi)` 對任意列回傳 (n, n_classes) 的 logits 區塊（尺度 1）。"""

    def __init__(self, kind, model, phi_tr, gamma, n_classes, diag, psd_events=None, nys=None):
        self.kind, self.model, self.phi_tr = kind, model, phi_tr
        self.gamma, self.n_classes, self.diag = gamma, int(n_classes), diag
        self.psd_events = psd_events or []
        self.nys = nys

    def scores(self, phi_ev):
        """對 phi_ev 的每一列評分。"""
        phi_ev = np.asarray(phi_ev)
        n = len(phi_ev)
        if n == 0:
            return np.zeros((0, self.n_classes), np.float64)
        if self.kind == "mlp":
            return _scores_to_logit_block(self.model, _decision_scores(self.model, phi_ev),
                                          self.n_classes, n)
        out = np.zeros((n, self.n_classes), np.float64)
        if self.kind == "nystrom":
            for s in range(0, n, SCORE_CHUNK):
                f = self.nys.transform(phi_ev[s:s + SCORE_CHUNK])
                out[s:s + SCORE_CHUNK] = _scores_to_logit_block(
                    self.model, self.model.decision_function(f), self.n_classes, len(f))
            return out
        for s in range(0, n, SCORE_CHUNK):
            sl = slice(s, s + SCORE_CHUNK)
            k_ev = K.rbf(phi_ev[sl], self.phi_tr, self.gamma)
            out[sl] = _scores_to_logit_block(self.model, _decision_scores(self.model, k_ev),
                                             self.n_classes, len(k_ev))
        return out


AUTO_EXACT_BELOW = 2000  # 群訓練列數低於此值時，近似沒有速度優勢（邊界情況：小群自動選精確）


def resolve_kernel_mode(mode, n_train):
    """回傳 (實際模式, 理由)。auto：小群精確、大群近似。"""
    if mode == "auto":
        if n_train < AUTO_EXACT_BELOW:
            return "exact", f"auto：訓練列 {n_train} < {AUTO_EXACT_BELOW}，近似無速度優勢，選精確"
        return "nystrom", f"auto：訓練列 {n_train} ≥ {AUTO_EXACT_BELOW}，選 Nyström"
    return mode, f"指定 {mode}"


def fit_expert(kind, phi_tr, y_tr, n_classes, gamma=None, seed=0, hidden=None,
               kernel_mode="exact", nystrom_m=512, cache_dir=None):
    """訓練單一群的專家。

    kind='kernel'：RBF 核 SVM（PQK／古典 RBF／RFF 都走這條）；kernel_mode 決定精確或 Nyström
    kind='mlp'   ：寬度匹配 MLP（對照組 C4）
    """
    diag = {}
    if kind == "mlp":
        model = _fit_mlp(phi_tr, y_tr, hidden or phi_tr.shape[1], seed=seed)
        return Expert("mlp", model, None, None, n_classes, diag)
    g = float(gamma) if gamma else K.auto_gamma(phi_tr)
    diag["gamma"] = g
    mode, why = resolve_kernel_mode(kernel_mode, len(phi_tr))
    diag["kernel_mode"], diag["kernel_mode_reason"] = mode, why
    if mode == "exact" and len(phi_tr) ** 2 * 8 > MEM_BUDGET_BYTES:
        # 例如 C5（不分群，22k 列）：精確 Gram 需 >3.9 GB 且特徵分解為 O(N³)，改用 Nyström（v1 也如此）
        mode = diag["kernel_mode"] = "nystrom"
        diag["kernel_mode_reason"] = why + f"；N²·8 B 超過記憶體預算 {MEM_BUDGET_BYTES:.0e}，改用 Nyström"
        nystrom_m = max(int(nystrom_m or 512), 2048)
    if mode == "nystrom":
        from sklearn.svm import LinearSVC

        nys = K.NystromMap(phi_tr, g, m=nystrom_m or 512, seed=seed)
        feats = nys.transform(phi_tr)
        diag.update(approximation="nystrom", nystrom_m=int(nys.m), nystrom_rank=int(feats.shape[1]))
        model = LinearSVC(C=10.0, dual=False, random_state=int(seed), max_iter=2000)
        model.fit(feats, np.asarray(y_tr, np.int64))
        return Expert("nystrom", model, None, g, n_classes, diag, nys=nys)
    diag["approximation"] = "exact"
    k_tr, psd, hit = K.gram_cached(phi_tr, g, cache_dir)
    diag["gram_cache_hit"] = bool(hit)
    diag.update(psd)
    diag.update(K.diagnostics(k_tr))
    events = [dict(psd, group_size=int(len(phi_tr)))] if psd.get("psd_violation") else []
    model = _fit_kernel_svm(k_tr, y_tr, seed=seed)
    return Expert("kernel", model, np.asarray(phi_tr, np.float64), g, n_classes, diag, events)


def fit_group_expert(kind, phi_tr, phi_ev, y_tr, class_ids, n_classes, n_ev,
                     gamma=None, seed=0, scale=1.0, hidden=None):
    """相容舊介面：擬合後立刻對 phi_ev 評分，回傳 (logits 區塊, 診斷)。"""
    ex = fit_expert(kind, phi_tr, y_tr, n_classes, gamma=gamma, seed=seed, hidden=hidden)
    return scale * ex.scores(phi_ev), ex.diag


def combine(ref_logits, blocks):
    """參考線 logits + 各群專家區塊。blocks: [(列索引, logits 區塊)]。"""
    out = np.array(ref_logits, np.float64, copy=True)
    for rows, block in blocks:
        if len(rows):
            out[rows] += block
    return out
