"""核矩陣與古典替身的單元測試（T015）。

C2（RFF 古典替身）是對 Schreiber et al. 的回應：若 PQK ≈ C2，量子部分可被古典取代。
"""
import numpy as np

from Research.QuSpike_PQK import kernels as K


def _phi(b=64, d=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (b, d))


# --- RBF 核的數學性質 ---

def test_kernel_is_symmetric_unit_diagonal_psd():
    """k_train 對稱、對角為 1、半正定。"""
    k = K.rbf(_phi(), gamma=0.5)
    assert np.allclose(k, k.T, atol=1e-6)
    assert np.allclose(np.diag(k), 1.0, atol=1e-6)
    ev = np.linalg.eigvalsh(k.astype(np.float64))
    assert ev.min() > -1e-6, f"最小特徵值 {ev.min():.2e} 顯示非半正定"


def test_eval_kernel_shape():
    """評估核形狀為 (n_eval, n_train)。"""
    assert K.rbf(_phi(20), _phi(50, seed=1), gamma=0.5).shape == (20, 50)


def test_auto_gamma_is_positive_and_scale_aware():
    """gamma='auto' 的中位數啟發式須為正，且隨特徵尺度變化。"""
    a = K.auto_gamma(_phi())
    b = K.auto_gamma(_phi() * 0.1)
    assert a > 0 and b > 0 and b > a, "特徵尺度變小時 gamma 應變大"


# --- 半正定修正（FR-015）---

def test_psd_violation_detected_and_corrected():
    """人工破壞半正定性後必須被偵測、修正並記錄，不得靜默通過。"""
    k = K.rbf(_phi(32), gamma=0.5)
    k[0, 1] = k[1, 0] = 5.0  # 破壞半正定
    fixed, diag = K.ensure_psd(k)
    assert diag["psd_violation"] is True
    assert diag["psd_correction"], "必須記錄修正方式"
    assert np.linalg.eigvalsh(fixed.astype(np.float64)).min() > -1e-8


def test_psd_ok_is_not_flagged():
    """健康的核不得被誤標為違反半正定。"""
    _, diag = K.ensure_psd(K.rbf(_phi(32), gamma=0.5))
    assert diag["psd_violation"] is False and diag["psd_correction"] is None


# --- 核值分佈診斷（FR-024；Huang 指出的核集中問題）---

def test_offdiagonal_diagnostics_reported():
    """必須輸出 off-diagonal 的平均與變異。"""
    d = K.diagnostics(K.rbf(_phi(), gamma=0.5))
    assert 0.0 <= d["offdiag_mean"] <= 1.0
    assert d["offdiag_std"] >= 0.0


def test_concentration_shows_up_as_low_offdiag_std():
    """核集中（所有核值趨同）應表現為 off-diagonal 變異極小。"""
    tight = K.diagnostics(K.rbf(_phi(), gamma=1e-6))   # gamma→0：所有核值趨近 1
    loose = K.diagnostics(K.rbf(_phi(), gamma=1.0))
    assert tight["offdiag_std"] < loose["offdiag_std"]


# --- C2：RFF 古典替身 ---

def test_rff_feature_dim_matches_quantum():
    """RFF 替身的特徵數須與量子讀出同維度（3n），以保證等容量比較。"""
    z = np.random.default_rng(0).uniform(-np.pi / 2, np.pi / 2, (40, 8))
    f_tr, f_ev = K.rff_surrogate(z, z[:10], n_features=12, seed=0)
    assert f_tr.shape == (40, 12) and f_ev.shape == (10, 12)


def test_rff_is_deterministic_given_seed():
    """同種子的 RFF 必須可重現。"""
    z = np.random.default_rng(1).uniform(-1, 1, (20, 8))
    a, _ = K.rff_surrogate(z, z, n_features=12, seed=5)
    b, _ = K.rff_surrogate(z, z, n_features=12, seed=5)
    assert np.allclose(a, b)


def test_rff_is_bounded_like_quantum_readout():
    """RFF 使用 cos／sin 基底，與量子期望值同樣有界於 [−1,1]。"""
    z = np.random.default_rng(2).uniform(-np.pi / 2, np.pi / 2, (64, 8))
    f, _ = K.rff_surrogate(z, z, n_features=12, seed=0)
    assert np.abs(f).max() <= 1.0 + 1e-9


# --- Nyström 後備 ---

def test_nystrom_approximates_exact_kernel():
    """Nyström 近似在 m 接近 n 時應趨近精確核。"""
    phi = _phi(120, 8, seed=4)
    exact = K.rbf(phi, gamma=0.5)
    approx = K.nystrom_kernel(phi, phi, gamma=0.5, m=100, seed=0)
    assert np.abs(approx - exact).mean() < 0.05
