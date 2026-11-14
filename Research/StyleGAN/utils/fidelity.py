"""保真度（論文式 1、式 2）與 10×10 熱圖。

純態下式 (1) 為重疊的平方：F1 = Tr(ρa ρb) = |<a|b>|²；式 (2) 為振幅重疊 F2 = |<a|b>| = sqrt(F1)。
即式 (1) 為式 (2) 的平方（FR-018）。向量先 L2 正規化（ρ 的跡為 1）。
"""
import numpy as np

PAPER_DELTA = 0.96


def _unit(v):
    """L2 正規化；零向量保持為零（對應「參考序列」基準，保真度為 0）。"""
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def fidelity_matrix(A, B, formula="eq1"):
    """A: (n,L)、B: (m,L) → (n,m) 保真度矩陣。"""
    ov = np.abs(_unit(A) @ _unit(B).T)          # |<a|b>|
    return ov ** 2 if formula == "eq1" else ov


def summarize(M):
    """回傳矩陣的最小／平均／最大值。"""
    """min / mean / max。"""
    return dict(min=float(M.min()), mean=float(M.mean()), max=float(M.max()))


def paper_check(ref_set, formula, source, interpretation, stat_mean):
    """論文門檻判定：只對「訓練期序列＋式 1＋字面解讀＋模型」且只對 Delta 0.96；其餘（含 Omicron）為 None。
    判定採平均值 > 0.96（論文為 Delta > 96%，未說明用最小或平均，此為本專案假設）。"""
    if (ref_set, formula, source, interpretation) == ("train", "eq1", "model", "literal"):
        return PAPER_DELTA, bool(stat_mean > PAPER_DELTA)
    return None, None
