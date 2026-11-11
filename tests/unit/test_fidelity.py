"""保真度公式。"""
import numpy as np
from Research.utils.fidelity import fidelity_matrix, paper_check, summarize


def test_identity_orthogonal_and_square():
    a = np.array([[1., 1, 0, 0]]); b = np.array([[0., 0, 1, 1]])
    assert fidelity_matrix(a, a)[0, 0] == np.float64(1.0) or abs(fidelity_matrix(a, a)[0, 0] - 1) < 1e-12
    assert fidelity_matrix(a, b)[0, 0] == 0
    c = np.array([[1., 0, 1, 0]])
    assert np.allclose(fidelity_matrix(a, c, "eq1"), fidelity_matrix(a, c, "eq2") ** 2)


def test_hand_example():
    a = np.zeros((1, 40)); b = np.zeros((1, 40))
    a[0, :21] = 1; b[0, 6:27] = 1                 # 各 21 個突變，共享 15 個
    assert abs(fidelity_matrix(a, b, "eq1")[0, 0] - (15 / 21) ** 2) < 1e-9     # ≈0.51
    assert abs(fidelity_matrix(a, b, "eq2")[0, 0] - 15 / 21) < 1e-9            # ≈0.71


def test_zero_vector_is_zero_and_paper_check():
    assert fidelity_matrix(np.zeros((1, 3)), np.ones((1, 3)))[0, 0] == 0
    assert paper_check("train", "eq1", "model", "literal", 0.97) == (0.96, True)
    assert paper_check("test", "eq1", "model", "literal", 0.97) == (None, None)
    assert paper_check("train", "eq2", "model", "literal", 0.97) == (None, None)
    assert summarize(np.array([[0., 1.]])) == dict(min=0.0, mean=0.5, max=1.0)
