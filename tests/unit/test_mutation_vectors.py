"""突變向量、k、lineage 代理標籤。"""
import numpy as np
import pandas as pd
from Research.StyleGAN.FE.lineage_proxy import lineage_labels
from Research.StyleGAN.FE.mutation_vectors import mutation_vectors, train_k
from Research.StyleGAN.FE.load_spike import AA


def test_vectors_and_k():
    ref = np.zeros(1273, np.uint8)
    X = np.tile(ref, (3, 1)); X[0, 4] = 1; X[1, 4] = 20; X[1, 6] = 2     # 20 = X 不計突變
    pm = pd.DataFrame({"spike_pos": [5, 7]})
    V = mutation_vectors(X, ref, pm)
    assert V.tolist() == [[1, 0], [0, 1], [0, 0]]
    assert train_k(V) == 1 and train_k(np.array([[1, 1, 1], [1, 0, 0]])) == 2   # 1.5→2（四捨五入到偶數）


def test_lineage_priority():
    ref = np.zeros(1273, np.uint8); X = np.tile(ref, (4, 1))
    def setm(i, name):
        X[i, int(name[1:-1]) - 1] = AA.index(name[-1])
    for n in ("L452R", "T478K"): setm(0, n)                       # Delta-like
    for n in ("D614G", "L452R"): setm(1, n)                       # Epsilon 蓋過 D614G
    for n in ("N501Y", "E484K", "A570D"): setm(2, n)              # Alpha 規則優先序較後
    lab = lineage_labels(X)
    assert lab[0].startswith("Delta") and lab[1].startswith("Epsilon") and lab[2].startswith("Alpha")
    assert lab[3].startswith("Wuhan")
