"""預測命中指標（FR-018b，research R10）：最小 Hamming 距離、新位置命中率、頻繁位點召回率。"""
import numpy as np
import torch


def _onehot(X, n_cls=21):
    """(n,Lf) 整數編碼 → (n, Lf*n_cls) float32 one-hot。"""
    X = torch.as_tensor(X, dtype=torch.long)
    return torch.nn.functional.one_hot(X, n_cls).flatten(1).float()


def min_hamming(A, B, bs=250):
    """每列 A 到 B 的最小 Hamming 距離（逐位點殘基差異數）。以 one-hot 內積計算：d = Lf - <a,b>。"""
    A, B = np.asarray(A), np.asarray(B)
    Lf, Bo = A.shape[1], _onehot(B)
    out = np.empty(len(A), dtype=np.int32)
    for i in range(0, len(A), bs):
        out[i:i + bs] = (Lf - (_onehot(A[i:i + bs]) @ Bo.T).max(1).values).round().int().numpy()
    return out


def novel_hit_rate(pos, train_freq, test_mutated):
    """pos: 生成的突變位置（spike 1 起算，任意形狀）；分母為「訓練期從未突變」的位置次數，
    分子為其中在測試期出現突變者。分母為 0 回傳 None（不適用）。"""
    pos = np.asarray(pos).ravel() - 1
    novel = train_freq[pos] == 0
    return None if novel.sum() == 0 else float(test_mutated[pos][novel].mean())


def frequent_recall(pos, test_freq, candidates, thr=0.01):
    """測試期頻繁位點（頻率 ≥ thr，限於模型可選的 candidates）有多少比例被生成的位置集合涵蓋。無頻繁位點回傳 None。"""
    freq_sites = set(np.where(test_freq >= thr)[0] + 1) & set(candidates)
    if not freq_sites:
        return None
    return float(len(freq_sites & set(np.asarray(pos).ravel().tolist())) / len(freq_sites))
