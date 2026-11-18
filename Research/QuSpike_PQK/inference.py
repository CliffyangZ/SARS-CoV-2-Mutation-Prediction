"""無標籤推論（Q1，FR-001／FR-002）。

論文 Algorithm 1 的推論只依路由結果決定專家套用範圍。v1 實作在
`pipeline._expert_blocks` 以「真實標籤屬於該群」再過濾一次——等於推論時偷看標籤。

本模組的保證：
- `predict` 與其呼叫的所有函式**簽名中沒有任何標籤參數**（測試以 inspect 斷言）；
- 專家特徵 φ 對**全部**評估列計算（φ 與種子、路由結果皆無關，可快取共用）；
- 只有 `predict_legacy_leaky` 讀標籤，僅供重現舊數字，結果檔必須標 `legacy_leaky`。
"""
from dataclasses import dataclass, field

import numpy as np

from . import experts, router


@dataclass
class InferenceState:
    """訓練完成後凍結的推論狀態。不含任何評估集標籤。"""

    ref_model: object
    device: str
    groups: list                    # 全部群（含 others），推論只用 name 與 class_ids
    targets: list                   # 配有專家的群
    n_classes: int
    projections: dict = field(default_factory=dict)
    experts: dict = field(default_factory=dict)
    featurize: object = None        # (群名, z, tag) -> φ；含快取
    grouped: bool = True
    scale: float = 1.0
    routing: str = "hard"
    soft_threshold: float = 0.0


def softmax(x):
    """數值穩定的 softmax。"""
    return router._softmax(x)


def _weights(state, ref, routing, threshold):
    """專家權重：{群名: (列索引, 權重)}。不使用標籤。"""
    out = {}
    if not state.grouped:  # C5：單一專家吃全部列
        for g in state.targets:
            out[g.name] = (np.arange(len(ref)), np.ones(len(ref)))
        return out, None
    thr = float(threshold) if routing == "soft" else 0.0
    w, names = router.route_weights(ref, state.groups, threshold=thr)
    for g in state.targets:
        j = names.index(g.name)
        rows = np.where(w[:, j] > 0)[0]
        out[g.name] = (rows, w[rows, j])
    return out, (w, names)


def expert_blocks(state, x_eval, tag="ev", rows_by_group=None):
    """各群專家對評估列的**未縮放** logits 區塊 {群名: (列索引, 區塊)}。

    rows_by_group=None 時對全部列評分（validation 掃描用）；否則只對指定列評分。
    """
    out = {}
    for g in state.targets:
        pr = state.projections[g.name]
        phi = state.featurize(g.name, pr.transform(x_eval), tag)
        rows = np.arange(len(x_eval)) if rows_by_group is None else rows_by_group[g.name]
        out[g.name] = (rows, state.experts[g.name].scores(phi[rows]) if len(rows) else None)
    return out


def assemble(state, ref, weights, blocks, scale):
    """ref + Σ scale · 權重 · 專家區塊。blocks 可含全部列（以列索引對齊）。"""
    out = np.array(ref, np.float64, copy=True)
    for g in state.targets:
        rows, w = weights[g.name]
        if not len(rows):
            continue
        brows, block = blocks[g.name]
        if len(brows) == len(ref):          # 全列區塊：直接取列
            sel = block[rows]
        else:                               # 只對 rows 評分：順序與 rows 相同
            sel = block
        out[rows] += scale * w[:, None] * sel
    return out


def predict(state, x_eval, tag="ev", scale=None, routing=None, soft_threshold=None):
    """無標籤推論。回傳 (預測, 機率, 資訊)。簽名刻意不含任何標籤參數（FR-001）。"""
    scale = state.scale if scale is None else scale
    routing = state.routing if routing is None else routing
    thr = state.soft_threshold if soft_threshold is None else soft_threshold
    ref = router.logits(state.ref_model, x_eval, state.device)
    weights, wn = _weights(state, ref, routing, thr)
    info = {"route_weights": wn}
    if scale == 0.0 or not state.targets:
        p = softmax(ref)
        return p.argmax(1), p, info
    blocks = expert_blocks(state, x_eval, tag, {k: v[0] for k, v in weights.items()})
    p = softmax(assemble(state, ref, weights, blocks, scale))
    return p.argmax(1), p, info


def predict_legacy_leaky(state, x_eval, y_eval, tag="ev", scale=None):
    """**v1 的洩漏行為**：專家只套用在「被路由到該群且真實標籤屬於該群」的列。

    僅供重現舊單種子數字與量化洩漏的影響；結果檔必須標 legacy_leaky=True，
    且 opt_report 不得將其當作最終數字。
    """
    scale = state.scale if scale is None else scale
    ref = router.logits(state.ref_model, x_eval, state.device)
    weights, _ = _weights(state, ref, "hard", 0.0)
    for g in state.targets:
        if g.name == "all":
            continue
        rows, w = weights[g.name]
        keep = np.isin(np.asarray(y_eval)[rows], list(g.class_ids))
        weights[g.name] = (rows[keep], w[keep])
    blocks = expert_blocks(state, x_eval, tag, {k: v[0] for k, v in weights.items()})
    p = softmax(assemble(state, ref, weights, blocks, scale))
    return p.argmax(1), p, {}
