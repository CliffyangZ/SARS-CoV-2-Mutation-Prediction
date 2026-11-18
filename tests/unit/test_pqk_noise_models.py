"""四種有資訊量的雜訊與終端去極化回歸（US5，FR-018／FR-019）。"""
import numpy as np
import pytest

from Research.QuSpike_PQK import noise, pqk


def _z(b=300, n=3, seed=0):
    return np.random.default_rng(seed).uniform(-1.5, 1.5, size=(b, 2 * n))


def test_terminal_depol_scales_pauli_uniformly_and_gram_invariant():
    """終端 DepolarizingChannel(p)：每個期望值乘 (1−4p/3)；重估 γ 後 Gram 不變（Prop. 4）。"""
    z, p = _z(), 0.02
    clean = pqk.features(z, n_qubits=3).numpy()
    noisy = pqk.features(z, n_qubits=3, noise_p=p).numpy()
    np.testing.assert_allclose(noisy, clean * (1 - 4 * p / 3), atol=1e-9)
    assert noise.gram_deviation(noisy, clean) < 1e-6


def test_readout_asym_matches_monte_carlo():
    """解析式與蒙地卡羅抽樣讀出在統計容差內相符。"""
    rng = np.random.default_rng(1)
    pa, pb = 0.08, 0.02
    e = 0.3   # 真實期望值
    n = 400000
    truth = np.where(rng.random(n) < (1 + e) / 2, 1, -1)
    flip = np.where(truth == 1, rng.random(n) < pa, rng.random(n) < pb)
    read = np.where(flip, -truth, truth)
    assert read.mean() == pytest.approx(float(noise.apply_readout_asym(np.array([e]), pa, pb)[0]), abs=0.01)


def test_uniform_readout_error_is_cancelled_by_median_gamma():
    """發現：所有位元相同的 (p_a, p_b)（即使不對稱）只是全域縮放＋平移，Gram 完全不變。"""
    z = _z()
    clean = pqk.features(z, n_qubits=3).numpy()
    assert noise.gram_deviation(noise.apply_readout_asym(clean, 0.05, 0.05), clean) < 1e-6
    assert noise.gram_deviation(noise.apply_readout_asym(clean, 0.10, 0.0), clean) < 1e-6


def test_per_qubit_readout_error_breaks_invariance():
    """各位元錯誤率不同（真實裝置）→ 不變性消失，掃描才有資訊量。"""
    z = _z()
    clean = pqk.features(z, n_qubits=3).numpy()
    per_q = noise.apply_readout_asym(clean, 0.10, 0.0, n_qubits=3)
    assert noise.gram_deviation(per_q, clean) > 1e-4
    np.testing.assert_allclose(noise.apply_readout_asym(clean, 0.0, 0.0, n_qubits=3), clean, atol=1e-12)


def test_amplitude_damping_shifts_z_up_and_breaks_invariance():
    z = _z(120)
    clean = pqk.features(z, n_qubits=3).numpy()
    noisy = noise.noisy_compute("amplitude_damping", 0.05)(z, n_qubits=3).numpy()
    zi = [3 * j + 2 for j in range(3)]
    assert (noisy[:, zi] - clean[:, zi]).mean() > 0.01       # ⟨Z⟩ 向 +1 漂移（加性項）
    assert noise.gram_deviation(noisy, clean, n=120) > 1e-4   # 不被中位數抵消


def test_interleaved_depol_breaks_invariance():
    z = _z(120)
    clean = pqk.features(z, n_qubits=3).numpy()
    noisy = noise.noisy_compute("interleaved_depol", 0.02)(z, n_qubits=3).numpy()
    assert noise.gram_deviation(noisy, clean, n=120) > 1e-4


def test_over_rotation_zero_equals_clean_and_nonzero_differs():
    z = _z(60)
    clean = pqk.features(z, n_qubits=3).numpy()
    np.testing.assert_allclose(noise.noisy_compute("over_rotation", 0.0)(z, n_qubits=3).numpy(), clean, atol=1e-9)
    assert noise.gram_deviation(noise.noisy_compute("over_rotation", 0.1)(z, n_qubits=3).numpy(), clean, n=60) > 1e-4


def test_zero_strength_noise_equals_clean_for_all_kinds():
    z = _z(40)
    clean = pqk.features(z, n_qubits=3).numpy()
    for kind in ("amplitude_damping", "interleaved_depol", "over_rotation", "readout_asym"):
        got = noise.noisy_compute(kind, 0.0)(z, n_qubits=3).numpy()
        np.testing.assert_allclose(got, clean, atol=1e-9, err_msg=kind)


def test_cache_key_distinguishes_noise_kinds():
    k1 = pqk.cache_key("h", "delta", "lda", 4, 4, 0.8, True, None, 0.0)
    assert noise.noise_cache_suffix("amplitude_damping", 0.05, 0.0) != noise.noise_cache_suffix("over_rotation", 0.05, 0.0)
    assert noise.noise_cache_suffix("amplitude_damping", 0.05, 0.0) != noise.noise_cache_suffix("amplitude_damping", 0.06, 0.0)
    assert k1  # v1 鍵不受影響
