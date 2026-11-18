"""v2 特徵圖（US3，FR-012／FR-015）。"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from Research.QuSpike_PQK import features as F, pqk, pqk_v2


def _z(n, b=6, seed=0, lo=-1.5, hi=1.5):
    return np.random.default_rng(seed).uniform(lo, hi, size=(b, 2 * n))


@pytest.mark.parametrize("n", [2, 4])
def test_uncoupled_matches_closed_form(n):
    """耦合全為 0：線路輸出與解析閉式在 1e-8 內相符（含雙位元相關 = 1-RDM 乘積）。"""
    z = _z(n)
    got = pqk_v2.features(z, n_qubits=n, coupling_scale=0.0).numpy()
    want = pqk_v2.closed_form_uncoupled(z, n, 0.8, 0.5, 4)
    np.testing.assert_allclose(got, want, atol=1e-8)


@pytest.mark.parametrize("n,dim", [(4, 24), (2, 9)])
def test_output_dimension(n, dim):
    assert pqk_v2.n_features(n) == dim
    assert pqk_v2.features(_z(n, 3), n_qubits=n).shape == (3, dim)


def test_v2_breaks_magnetization_conservation_but_v1_keeps_it():
    """v1：⟨ΣZ⟩ ≡ 0（守恆）；v2：⟨ΣZ⟩ 隨資料變動（弱點 2 被破除）。"""
    n = 4
    z = _z(n, 12, seed=3)
    v1 = pqk.features(z, n_qubits=n, entangle=True).numpy()
    v2 = pqk_v2.features(z, n_qubits=n).numpy()
    zi = [3 * j + 2 for j in range(n)]
    assert np.abs(v1[:, zi].sum(1)).max() < 1e-9
    assert np.abs(v2[:, zi].sum(1)).max() > 1e-2


def test_no_qubit_insensitive_to_coupling_dims():
    """弱點 1：即使某耦合維度為 0（零填補），位元仍因 J0 受耦合影響（v1 該位元完全解耦）。"""
    n = 4
    z = _z(n, 5, seed=4)
    z[:, 6:] = 0.0   # 與 v1 的零填補相同：邊 (2,3)、(3,0) 的資料耦合為 0
    on = pqk_v2.features(z, n_qubits=n).numpy()
    off = pqk_v2.features(z, n_qubits=n, coupling_scale=0.0).numpy()
    q3 = [3 * 3, 3 * 3 + 1, 3 * 3 + 2]
    assert np.abs(on[:, q3] - off[:, q3]).max() > 1e-4
    v1_on = pqk.features(z, n_qubits=n, entangle=True).numpy()
    v1_off = pqk.features(z, n_qubits=n, entangle=False).numpy()
    np.testing.assert_allclose(v1_on[:, q3], v1_off[:, q3], atol=1e-9)  # v1 的位元 3 確實解耦


def test_cache_key_does_not_collide_with_v1():
    design = dict(n_qubits=4, trotter_steps=4, evolution_time=0.8, h=0.5, j0=0.1,
                  kappa_yy=0.4, kappa_zz=1.5)
    k2 = pqk_v2.cache_key_v2("h", "delta", "lda+pls", design, True)
    k1 = pqk.cache_key("h", "delta", "lda", 4, 4, 0.8, True, None, 0.0)
    assert k2 != k1 and k2.startswith("v2")
    assert k2 != pqk_v2.cache_key_v2("h", "delta", "lda+pls", dict(design, h=0.6), True)


def test_v1_circuit_unchanged_by_v2():
    """v2 另放檔案：pqk.py 內容雜湊沿用 test_pqk_v1_frozen 的常數。"""
    from tests.unit.test_pqk_v1_frozen import FROZEN_PQK_SHA

    assert hashlib.sha256(Path(pqk.__file__).read_bytes()).hexdigest()[:16] == FROZEN_PQK_SHA
