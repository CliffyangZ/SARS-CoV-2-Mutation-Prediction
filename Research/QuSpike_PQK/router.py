"""古典路由與參考線（T018）。

路由器就是監督線性參考線本身（突變指標 → 22 類），這有兩個作用：
1. 它是對照組 C0，也是架構上的下限——量子路徑貢獻為零時整個模型還原成它（FR-009）。
2. 它決定每個樣本進哪一群；易分群直接用它的預測輸出（FR-010），
   所以 `others` 群的預測與 C0 必須逐筆完全相同（SC-004）。
"""
import numpy as np
import torch
from torch import nn


REFERENCE_KINDS = ("linear", "gbm", "blend")


class SklearnReference:
    """非線性參考線的轉接器：輸出 log 機率當 logits。

    softmax(log p) = p，且每列加常數不影響 softmax／argmax／群機率，
    所以 log 機率可直接取代線性 logits 進入殘差融合與軟路由。
    `blend` 為線性 logits 與 GBM 的 log-softmax 平均（幾何平均）。
    """

    def __init__(self, lin=None, gbm=None, n_classes=0):
        self.lin, self.gbm, self.n_classes = lin, gbm, int(n_classes)

    def _gbm_logp(self, x):
        p = np.full((len(x), self.n_classes), 1e-9)
        p[:, self.gbm.classes_] = self.gbm.predict_proba(np.asarray(x, np.float32))
        return np.log(np.clip(p, 1e-9, None))

    def logits_np(self, x, device="cpu"):
        parts = []
        if self.gbm is not None:
            parts.append(self._gbm_logp(x))
        if self.lin is not None:
            lg = logits(self.lin[0], x, self.lin[1])
            m = lg.max(1, keepdims=True)
            parts.append(lg - m - np.log(np.exp(lg - m).sum(1, keepdims=True)))
        return np.mean(parts, axis=0)


def fit_reference(x_train, y_train, n_classes, seed=0, epochs=60, lr=3e-3, wd=1e-4,
                  batch=512, device=None, kind="linear"):
    """訓練監督參考線。kind='linear'（預設）與 QuSpike_JEPA.baselines.mutset_linear 同一設定，
    行為與 v1／v2 完全相同；'gbm'／'blend' 僅供探索性實驗（見 exp_ref_proj.py）。"""
    if kind not in REFERENCE_KINDS:
        raise ValueError(f"未知的參考線：{kind}")
    if kind != "linear":
        from sklearn.ensemble import HistGradientBoostingClassifier

        gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                                             early_stopping=False, random_state=int(seed)
                                             ).fit(np.asarray(x_train, np.float32), np.asarray(y_train))
        lin = fit_reference(x_train, y_train, n_classes, seed, epochs, lr, wd, batch, device) \
            if kind == "blend" else None
        return SklearnReference(lin=lin, gbm=gbm, n_classes=n_classes), "cpu"
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    x = torch.as_tensor(np.asarray(x_train, np.float32), device=dev)
    y = torch.as_tensor(np.asarray(y_train, np.int64), device=dev)
    model = nn.Linear(x.shape[1], int(n_classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(int(epochs)):
        perm = torch.randperm(len(y), generator=g).to(dev)
        for s in range(0, len(y), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return model, dev


def logits(model, x, device, batch=4096):
    """參考線的 logits（float64 numpy）。"""
    if hasattr(model, "logits_np"):
        return np.asarray(model.logits_np(x), np.float64)
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.as_tensor(np.asarray(x[s:s + batch], np.float32), device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def group_of_class(groups):
    """類別索引 → 群名。"""
    return {c: g.name for g in groups for c in g.class_ids}


def route(ref_logits, groups):
    """以參考線的預測決定每個樣本的群。回傳 (群名陣列, 參考線預測)。"""
    pred = ref_logits.argmax(1)
    lut = group_of_class(groups)
    return np.array([lut[int(p)] for p in pred]), pred


def router_accuracy(routed, y_true, groups):
    """路由準確率：樣本被分到其**真實類別**所屬的群的比例（FR-010）。

    這是量子專家能改善的上限——被路錯群的樣本，專家根本看不到。
    """
    lut = group_of_class(groups)
    truth = np.array([lut[int(c)] for c in y_true])
    return float((routed == truth).mean())


# --- 008：軟路由（Q3，FR-007／FR-008）---

SOFT_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)  # 0 即硬路由；只在 validation 上選


def _softmax(x):
    z = np.asarray(x, np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def group_proba(ref_logits, groups):
    """群機率 p_g(x) = Σ_{c∈g} softmax(logits)_c。回傳 (n, G)，欄順序同 groups。"""
    p = _softmax(ref_logits)
    return np.stack([p[:, list(g.class_ids)].sum(1) for g in groups], 1)


def route_weights(ref_logits, groups, threshold=0.0, mass=0.95, max_groups=3):
    """每列對每群的專家權重 W (n, G)，**不使用任何標籤**。

    - 未觸發（threshold=0，或前兩名群機率差 ≥ threshold）：硬路由，W 為參考線預測所屬群的 one-hot，
      與 `route` 完全一致。
    - 觸發：取機率由高到低累積達 `mass` 的前若干群（最多 `max_groups` 群），
      W_g = p_g / Σ_{S} p_g。`others` 群也計入分母但不配專家，故專家權重和 ≤ 1（結構下限不變）。
    回傳 (W, 群名清單)。
    """
    names = [g.name for g in groups]
    p = group_proba(ref_logits, groups)
    lut = {n: i for i, n in enumerate(names)}
    glut = group_of_class(groups)
    hard = np.array([lut[glut[int(c)]] for c in np.asarray(ref_logits).argmax(1)])
    w = np.zeros_like(p)
    w[np.arange(len(p)), hard] = 1.0
    if threshold and threshold > 0 and len(names) > 1:
        order = np.argsort(-p, axis=1)
        sp = np.take_along_axis(p, order, 1)
        trig = np.where((sp[:, 0] - sp[:, 1]) < threshold)[0]
        for i in trig:
            cum, keep = 0.0, []
            for j in range(min(max_groups, len(names))):
                keep.append(order[i, j])
                cum += sp[i, j]
                if cum >= mass:
                    break
            sel = np.zeros(len(names))
            sel[keep] = p[i, keep]
            w[i] = sel / max(sel.sum(), 1e-12)
    return w, names


def coverage(w, names, y_true, groups):
    """真實群別的專家「有機會修正」的列比例（FR-008）。**只用於報告，不進推論。**

    硬路由時等於 router_accuracy（含 others）；軟路由時為真實群別是否落在候選集合內。
    """
    lut = group_of_class(groups)
    idx = {n: i for i, n in enumerate(names)}
    truth = np.array([idx[lut[int(c)]] for c in y_true])
    return float((w[np.arange(len(truth)), truth] > 0).mean())
