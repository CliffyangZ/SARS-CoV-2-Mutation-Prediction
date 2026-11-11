"""映射：訓練期背景、最常見替代殘基、已存在計數。"""
import numpy as np
from Research.utils.sequence_mapping import map_to_sequences, train_alt_residues


def test_alt_and_mapping():
    ref = np.zeros(10, np.uint8)
    X = np.tile(ref, (4, 1)); X[0, 2] = 3; X[1, 2] = 3; X[2, 2] = 5; X[3, 7] = 4
    w = np.array([1., 1., 1., 1.])
    alt = train_alt_residues(X, ref, w)
    assert alt[2] == 3 and alt[7] == 4                         # 最常見的非參考殘基
    mapped, ex = map_to_sequences(np.array([[3, 8]]), X[3:4], ref, alt, np.random.default_rng(0))   # 背景 = X[3]
    assert mapped[0, 2] == alt[2] and mapped[0, 7] == 4 and ex[0] == 1   # 位置 8 背景已突變 → 不改、計為已存在
    # 背景在位置 8（index 7）已突變 → 不改、計為已存在
    _, ex2 = map_to_sequences(np.array([[8]]), X[3:4], ref, alt, np.random.default_rng(0))
    assert ex2[0] == 1
