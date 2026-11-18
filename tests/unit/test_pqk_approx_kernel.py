"""近似核（Q7，US4，FR-016／FR-017）。"""
import tracemalloc

import numpy as np
import pytest

from Research.QuSpike_PQK import experts, kernels as K


def _phi(n=400, d=8, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, size=(n, d))


def test_nystrom_features_reproduce_exact_kernel_when_m_equals_n():
    phi = _phi(150)
    g = 0.7
    f_tr, _ = K.nystrom_features(phi, phi[:5], g, m=150, seed=0)
    exact = K.rbf(phi, None, g).astype(np.float64)
    np.testing.assert_allclose(f_tr @ f_tr.T, exact, atol=1e-5)


def test_nystrom_does_not_materialise_n_by_n():
    n, m = 3000, 100
    phi = _phi(n)
    tracemalloc.start()
    mp = K.NystromMap(phi, 0.5, m=m, seed=0)
    mp.transform(phi)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < n * n * 8 / 4  # 遠小於 N×N 的 float64 矩陣


def test_nystrom_transform_arbitrary_rows_shape():
    phi = _phi(300)
    mp = K.NystromMap(phi, 0.5, m=64, seed=1)
    assert mp.transform(phi[:7]).shape[0] == 7 and mp.transform(phi[:7]).shape[1] <= 64


def test_auto_mode_small_group_chooses_exact_with_reason():
    mode, why = experts.resolve_kernel_mode("auto", 1500)
    assert mode == "exact" and "1500" in why
    assert experts.resolve_kernel_mode("auto", 9000)[0] == "nystrom"
    assert experts.resolve_kernel_mode("exact", 9000)[0] == "exact"


def test_nystrom_expert_close_to_exact_on_separable_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    phi = rng.normal(size=(600, 6)) * 0.3 + np.eye(3)[y].repeat(2, axis=1)
    ex = experts.fit_expert("kernel", phi, y, 3, kernel_mode="exact")
    ap = experts.fit_expert("kernel", phi, y, 3, kernel_mode="nystrom", nystrom_m=256)
    assert ap.diag["approximation"] == "nystrom"
    assert (ex.scores(phi).argmax(1) == ap.scores(phi).argmax(1)).mean() > 0.9


def test_exact_over_memory_budget_falls_back_to_nystrom(monkeypatch):
    monkeypatch.setattr(experts, "MEM_BUDGET_BYTES", 1000.0)
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 120)
    ex = experts.fit_expert("kernel", rng.normal(size=(120, 4)), y, 2, kernel_mode="exact")
    assert ex.diag["kernel_mode"] == "nystrom" and "預算" in ex.diag["kernel_mode_reason"]


def test_gram_cache_hits_on_second_call_and_matches(tmp_path):
    phi = _phi(200)
    k1, psd1, hit1 = K.gram_cached(phi, 0.5, tmp_path)
    k2, psd2, hit2 = K.gram_cached(phi, 0.5, tmp_path)
    assert (hit1, hit2) == (False, True)
    np.testing.assert_array_equal(k1, k2)
    assert psd1 == psd2
    # 不同 γ 或不同 φ 不得命中
    assert K.gram_cached(phi, 0.6, tmp_path)[2] is False
    assert K.gram_cached(_phi(200, seed=3), 0.5, tmp_path)[2] is False


def test_cholesky_psd_check_agrees_with_eigh_and_logs_violation():
    phi = _phi(120)
    k = K.rbf(phi, None, 0.5)
    fixed, d = K.ensure_psd(k)
    assert d["psd_violation"] is False
    bad = np.array(k, np.float64)
    bad[0, 1] = bad[1, 0] = 5.0          # 破壞半正定
    fixed, d = K.ensure_psd(bad)
    assert d["psd_violation"] is True and d["psd_correction"]
    assert np.linalg.eigvalsh(np.asarray(fixed, np.float64)).min() > -1e-4


def test_two_seeds_same_svc_result_with_gram_cache(tmp_path):
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 300)
    phi = rng.normal(size=(300, 5))
    a = experts.fit_expert("kernel", phi, y, 3, seed=0, cache_dir=tmp_path)
    b = experts.fit_expert("kernel", phi, y, 3, seed=1, cache_dir=tmp_path)
    assert a.diag["gram_cache_hit"] is False and b.diag["gram_cache_hit"] is True
    np.testing.assert_allclose(a.scores(phi[:20]), b.scores(phi[:20]), atol=1e-6)
