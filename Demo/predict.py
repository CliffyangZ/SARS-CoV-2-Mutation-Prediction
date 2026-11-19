"""預測服務：Query → Prediction（Top-k、路由、參考 vs 融合的分數差）。

分類邏輯全部來自 Research/QuSpike_PQK：參考 logits 用 `router.logits`，
專家區塊用 `inference.expert_blocks`，融合用 `inference.assemble`（與 `inference.predict` 同一路徑），
所以 s=0 或 O 組時殘差必為 0，且 Top-1 與離線推論一致。本模組沒有任何標籤參數。
"""
from dataclasses import dataclass, field, asdict

import numpy as np

from Research.QuSpike_PQK import inference, router

from . import parse
from .bundle import ModelBundle

DISCLAIMER = "專家（量子路徑）的增益落在雜訊下限（δ=0.02）內，結果不代表量子優勢；機率未經校準。"
GROUP_DISPLAY = {"delta": "D", "b1": "B", "others": "O"}
LOW_CONF = 0.5
MAX_K = 64


@dataclass
class LineageScore:
    lineage: str
    prob: float
    ref_prob: float
    ref_logit: float
    fused_logit: float
    residual: float
    rank: int
    ref_rank: int


@dataclass
class Routing:
    group: str
    probs: dict
    expert_applied: bool


@dataclass
class Prediction:
    top_k: list
    routing: Routing
    s_used: float
    top1_changed_by_expert: bool
    confidence_note: str
    parsed: dict
    disclaimer: str = DISCLAIMER

    def to_dict(self):
        return asdict(self)


class Predictor:
    def __init__(self, bundle: ModelBundle):
        self.b = bundle
        self.st = bundle.state
        self._blocks_key, self._blocks = None, None

    def _logits(self, x, s):
        """回傳 (參考 logits, 融合 logits, 路由權重資訊)。"""
        st = self.st
        ref = router.logits(st.ref_model, x, st.device)
        weights, _ = inference._weights(st, ref, "hard", 0.0)
        if s == 0.0 or not st.targets or not any(len(r) for r, _ in weights.values()):
            return ref, ref.copy()
        rows = {k: v[0] for k, v in weights.items()}
        blocks = inference.expert_blocks(st, x, "ev", rows)
        return ref, inference.assemble(st, ref, weights, blocks, s)

    def predict(self, text, mode="auto", k=5, s=None):
        if not isinstance(k, int) or k < 1:
            raise parse.InputError("k 必須是 ≥ 1 的整數")
        if s is not None and (not np.isfinite(s) or s < 0):
            raise parse.InputError("s 必須是 ≥ 0 的數")
        parsed = parse.parse_input(text, mode)
        x, n_oov = parse.to_vector(parsed, self.b.vocab)
        s_used = float(self.b.s_default if s is None else s)
        ref, fused = self._logits(x, s_used)
        groups = self.st.groups
        gp = router.group_proba(ref, groups)[0]
        top_g = groups[int(gp.argmax())]
        routed, _ = router.route(ref, groups)
        grp = str(routed[0])
        gobj = next(g for g in groups if g.name == grp)
        applied = bool(gobj.is_quantum and s_used > 0)
        p, p_ref = inference.softmax(fused)[0], inference.softmax(ref)[0]
        order = np.argsort(-p, kind="stable")
        ref_rank = np.empty(len(p_ref), int)
        ref_rank[np.argsort(-p_ref, kind="stable")] = np.arange(1, len(p_ref) + 1)
        k = min(k, len(p))
        labels = self.b.labels
        top = [LineageScore(labels[i], float(p[i]), float(p_ref[i]), float(ref[0, i]), float(fused[0, i]),
                            float(fused[0, i] - ref[0, i]), r + 1, int(ref_rank[i]))
               for r, i in enumerate(order[:k])]
        names = {g.name: GROUP_DISPLAY.get(g.name, g.name) for g in groups}
        routing = Routing(GROUP_DISPLAY.get(grp, grp), {names[g.name]: float(v) for g, v in zip(groups, gp)}, applied)
        changed = bool(order[0] != np.argmax(p_ref))
        note = self._note(p[order[0]], parsed, n_oov)
        info = dict(kind=parsed.kind, n_mutations=len(parsed.mutations), n_unknown=parsed.n_unknown,
                    n_out_of_vocab=n_oov, length=parsed.length, warnings=list(parsed.warnings))
        return Prediction(top, routing, s_used, changed, note, info)

    @staticmethod
    def _note(p1, parsed, n_oov):
        notes = []
        if p1 < LOW_CONF:
            notes.append(f"最高機率只有 {p1:.2f}，信心低；可能是訓練未涵蓋的譜系")
        if n_oov:
            notes.append(f"{n_oov} 個突變不在訓練詞彙中，已被忽略")
        return "；".join(notes) or None
