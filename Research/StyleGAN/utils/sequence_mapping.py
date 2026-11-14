"""突變位置 → 完整 1273 aa 序列（FR-017，research R11）。

背景：隨機抽取的訓練期真實序列；替代殘基：訓練期該位點最常見的非參考殘基（只用訓練期資料）。
"""
import numpy as np


def train_alt_residues(X_train, ref, w):
    """各位點在訓練期最常見的非參考殘基（加權）。訓練期沒有突變的位點無證據可用，
    退而求其次取 index 最小且 ≠ 參考的殘基（任意但固定的規則，列入報告假設）。"""
    alt = np.zeros(X_train.shape[1], dtype=np.uint8)
    for i in range(X_train.shape[1]):
        cnt = np.zeros(21)
        m = (X_train[:, i] != ref[i]) & (X_train[:, i] != 20)
        np.add.at(cnt, X_train[m, i], w[m])
        alt[i] = cnt.argmax() if cnt.sum() > 0 else (0 if ref[i] != 0 else 1)
    return alt


def map_to_sequences(mut_spike_pos, X_train, ref, alt, rng):
    """mut_spike_pos: (n,k) spike 位置（1 起算）。回傳 (mapped (n,1273) uint8, n_existing (n,))。
    背景在該位置已是突變（≠參考且≠X）則不改，並計入 n_existing（「已存在」）。"""
    n = len(mut_spike_pos)
    bg = X_train[rng.integers(len(X_train), size=n)].copy()
    existing = np.zeros(n, dtype=int)
    for r in range(n):
        for p in np.asarray(mut_spike_pos[r]).ravel() - 1:
            if p < 0:                      # 填充位置（突變數不足），不是位點
                continue
            if bg[r, p] != ref[p] and bg[r, p] != 20:
                existing[r] += 1
            else:
                bg[r, p] = alt[p]
    return bg, existing
