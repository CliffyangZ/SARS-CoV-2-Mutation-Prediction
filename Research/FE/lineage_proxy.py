"""以關鍵突變組合貼 lineage 代理標籤（近似標籤，非 Pango；優先序同 notebook 第 21 格）。"""
import numpy as np

from Research.FE.load_spike import AA

KEY = ["D614G", "N501Y", "A570D", "E484K", "L452R", "T478K"]
ORDER = ["Wuhan-like (D614)", "D614G only/other", "E484K other", "Epsilon-like (L452R)",
         "Beta/Gamma-like (N501Y+E484K)", "Alpha-like (N501Y+A570D)", "Delta-like (L452R+T478K)"]


def has_mutation(X_int, name):
    """X_int: (n, 1273) 整數編碼；name 如 'N501Y'（位置 1 起算）→ 布林陣列。"""
    return X_int[:, int(name[1:-1]) - 1] == AA.index(name[-1])


def lineage_labels(X_int):
    """後面的規則覆蓋前面的（優先序：Delta-like 最高）。"""
    h = {n: has_mutation(X_int, n) for n in KEY}
    lab = np.full(len(X_int), ORDER[0], dtype=object)
    lab[h["D614G"]] = ORDER[1]
    lab[h["E484K"]] = ORDER[2]
    lab[h["L452R"]] = ORDER[3]
    lab[h["N501Y"] & h["E484K"]] = ORDER[4]
    lab[h["N501Y"] & h["A570D"]] = ORDER[5]
    lab[h["L452R"] & h["T478K"]] = ORDER[6]
    return lab
