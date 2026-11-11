"""位點選取（只用訓練期資料，research R4）：1273 → L 個位點。

規則優先序：訓練期有突變的位點（依加權頻率）→ 其左右鄰近位點 → 隨機補滿；
突變位點多於 L 時只保留頻率最高的 L 個。
"""
import numpy as np
import pandas as pd

from Research.FE.load_spike import L_FULL


def weighted_mut_freq(X_train, ref, w):
    """各位點的加權突變頻率：殘基 ≠ 參考且 ≠ X(=20)。"""
    is_mut = (X_train != ref[None, :]) & (X_train != 20)
    return (is_mut * w[:, None]).sum(0) / w.sum()


def build_position_map(X_train, ref, w, L, seed=0):
    """回傳 DataFrame(index, spike_pos, origin, train_freq)；spike_pos 為 1 起算。"""
    freq = weighted_mut_freq(X_train, ref, w)
    mut_pos = np.where(freq > 0)[0]
    mut_pos = mut_pos[np.argsort(-freq[mut_pos], kind="stable")]
    chosen, origin = list(mut_pos[:L]), {int(p): "train_mutation" for p in mut_pos[:L]}
    if len(chosen) < L:                                   # 鄰近位點
        for p in mut_pos:
            for q in (p - 1, p + 1):
                if 0 <= q < L_FULL and q not in origin and len(chosen) < L:
                    chosen.append(q); origin[int(q)] = "neighbor"
    if len(chosen) < L:                                   # 隨機補滿
        rest = np.array([p for p in range(L_FULL) if p not in origin])
        for q in np.random.default_rng(seed).choice(rest, L - len(chosen), replace=False):
            chosen.append(int(q)); origin[int(q)] = "random"
    chosen = sorted(int(c) for c in chosen)               # 依 spike 位置排序，index 才有空間意義
    return pd.DataFrame({"index": range(L), "spike_pos": [c + 1 for c in chosen],
                         "origin": [origin[c] for c in chosen],
                         "train_freq": [freq[c] for c in chosen]})
