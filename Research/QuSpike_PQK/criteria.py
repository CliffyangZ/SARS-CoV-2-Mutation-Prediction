"""四項事前判準的自動判定（T025）。

判準與雜訊下限在跑實驗**之前**就寫死在這裡（見 Doc/QuSpike-PQK.md §4），
結論因此不是事後挑選的。三態判定由程式產出，不留需人工解讀的欄位（SC-002）。
"""
import numpy as np

NOISE_FLOOR = 0.02  # 事前訂定；約為同種子 GPU 重跑差異的下限（研究報告 §7-R7）

# (宣稱, 設定 A, 設定 B 或 B 的候選集合)
CLAIMS = (
    ("框架有用", "MAIN", ("C0",)),
    ("量子特徵有用", "MAIN", ("C1", "C2", "C4")),   # 取最強的古典對照
    ("量子性有用（糾纏有作用）", "MAIN", ("C3",)),
    ("分群設計有用", "MAIN", ("C5",)),
)


def mean_std(vals):
    """(平均, 樣本標準差)；單一樣本時標準差為 0。"""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def verdict(a, b, floor=NOISE_FLOOR):
    """a 相對 b 的判定。回傳 (增量, 'holds'|'fails'|'within_noise'|'insufficient_data')。"""
    if not (np.isfinite(a[0]) and np.isfinite(b[0])):
        return float("nan"), "insufficient_data"
    d = a[0] - b[0]
    band = max(a[1], b[1], floor)
    if abs(d) <= band:
        return d, "within_noise"
    return d, "holds" if d > 0 else "fails"


def evaluate(scores, floor=NOISE_FLOOR):
    """scores: {設定代號: [各種子分數]}。回傳四項判準的紀錄清單。"""
    out = []
    for claim, a_name, b_names in CLAIMS:
        a = mean_std(scores.get(a_name, []))
        # 取 B 候選中平均最高者（最強的對照組），才是對 A 最嚴格的檢定
        cands = [(n, mean_std(scores[n])) for n in b_names if n in scores and len(scores[n])]
        if not np.isfinite(a[0]) or not cands:
            out.append(dict(claim=claim, a=a_name, b="|".join(b_names), delta=float("nan"),
                            std_a=a[1], std_b=float("nan"), noise_floor=floor,
                            verdict="insufficient_data"))
            continue
        b_name, b = max(cands, key=lambda kv: kv[1][0])  # kv[1][0] 為該對照組的平均
        d, v = verdict(a, b, floor)
        out.append(dict(claim=claim, a=a_name, b=b_name, delta=d,
                        std_a=a[1], std_b=b[1], noise_floor=floor, verdict=v))
    return out


def stop_condition_met(records):
    """提案 §4 的停止條件：「量子特徵有用」與「量子性有用」皆不成立。

    成立時不得再加大 n、改 τ／T／gamma 來求轉正——此約束存在是為了避免重演
    前一輪（JEPA）的調參循環。
    """
    keys = ("量子特徵有用", "量子性有用（糾纏有作用）")
    rel = [r for r in records if r["claim"] in keys]
    return bool(rel) and all(r["verdict"] != "holds" for r in rel)
