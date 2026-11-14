"""位點選取（只用訓練期資料，research R4）：1273 → L 個位點。

規則優先序：訓練期有突變的位點（依加權頻率）→ 其左右鄰近位點 → 隨機補滿；
突變位點多於 L 時只保留頻率最高的 L 個。
"""
import numpy as np
import pandas as pd

from Research.StyleGAN.FE.load_spike import L_FULL


def weighted_mut_freq(X_train, ref, w):
    """各位點的加權突變頻率：殘基 ≠ 參考且 ≠ X(=20)。"""
    is_mut = (X_train != ref[None, :]) & (X_train != 20)
    return (is_mut * w[:, None]).sum(0) / w.sum()


def excluded_positions(X_train, ref, w, threshold):
    """近常數位點：訓練期加權突變頻率 ≥ threshold（例如 D614G）。回傳 DataFrame(spike_pos, train_freq, reason)。"""
    freq = weighted_mut_freq(X_train, ref, w)
    pos = np.where(freq >= threshold)[0]
    return pd.DataFrame({"spike_pos": pos + 1, "train_freq": freq[pos], "reason": "excluded_const"})


def build_position_map(X_train, ref, w, L, seed=0, drop_const_threshold=None, order="spike"):
    """回傳 DataFrame(index, spike_pos, origin, train_freq)；spike_pos 為 1 起算。

    drop_const_threshold：訓練期頻率 ≥ 此值的位點完全排除（含補位）；None 表示不排除（legacy）。
    order："spike" 依 spike 位置排列（現行）；"freq" 依訓練期頻率由高到低（同頻率依位置）。
    """
    freq = weighted_mut_freq(X_train, ref, w)
    banned = set(np.where(freq >= drop_const_threshold)[0].tolist()) if drop_const_threshold is not None else set()
    mut_pos = np.array([p for p in np.where(freq > 0)[0] if p not in banned], dtype=int)
    mut_pos = mut_pos[np.argsort(-freq[mut_pos], kind="stable")]
    chosen, origin = list(mut_pos[:L]), {int(p): "train_mutation" for p in mut_pos[:L]}
    if len(chosen) < L:                                   # 鄰近位點
        for p in mut_pos:
            for q in (p - 1, p + 1):
                if 0 <= q < L_FULL and q not in origin and q not in banned and len(chosen) < L:
                    chosen.append(q); origin[int(q)] = "neighbor"
    if len(chosen) < L:                                   # 隨機補滿
        rest = np.array([p for p in range(L_FULL) if p not in origin and p not in banned])
        for q in np.random.default_rng(seed).choice(rest, L - len(chosen), replace=False):
            chosen.append(int(q)); origin[int(q)] = "random"
    if order == "freq":                                   # 依訓練期頻率排列：頻繁位點排在前面
        chosen = sorted((int(c) for c in chosen), key=lambda c: (-freq[c], c))
    else:                                                 # 依 spike 位置排序，index 才有空間意義
        chosen = sorted(int(c) for c in chosen)
    return pd.DataFrame({"index": range(L), "spike_pos": [c + 1 for c in chosen],
                         "origin": [origin[c] for c in chosen],
                         "train_freq": [freq[c] for c in chosen]})
