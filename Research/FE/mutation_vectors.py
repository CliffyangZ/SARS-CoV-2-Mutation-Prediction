"""0/1 突變向量與訓練期 k（research R5）。"""
import numpy as np


def mutation_vectors(X_int, ref, pmap):
    """X_int: (n,1273) → (n,L) uint8；殘基 ≠ 參考且 ≠ X 為 1。"""
    cols = pmap.spike_pos.values - 1
    sub = X_int[:, cols]
    return ((sub != ref[cols][None, :]) & (sub != 20)).astype(np.uint8)


def train_k(V_train):
    """訓練期（唯一序列）平均突變數，四捨五入（規模二預期 3、規模一預期 2）。"""
    return int(round(float(V_train.sum(1).mean())))
