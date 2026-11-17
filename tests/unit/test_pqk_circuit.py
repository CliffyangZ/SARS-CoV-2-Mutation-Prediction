"""量子特徵圖的單元測試（T006／T032）。

最關鍵的一項是 test_no_entangle_matches_closed_form：關閉糾纏時特徵必須等於解析的
cos／sin 形式。對照組 C3（無糾纏）是「量子性是否有作用」唯一的證據來源，
若 C3 的實作本身有錯，該宣稱就不可信。見 specs/007-pqk-pennylane-ibm/research.md D6。
"""
import numpy as np
import pennylane as qml
import pytest
import torch

from Research.QuSpike_PQK import pqk


def _z(b=16, n=4, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.uniform(-np.pi / 2, np.pi / 2, (b, 2 * n)), dtype=torch.float64)


# --- 正確性：無糾纏的閉式解（SC-003／FR-008）---

def test_no_entangle_matches_closed_form():
    """關閉糾纏時：⟨X_j⟩=cos(2τz_j)、⟨Y_j⟩=sin(2τz_j)、⟨Z_j⟩=0。

    初態 |+⟩ 經總角度 θ=2τz_j 的 RZ 後即為此形式；與 T（Trotter 步數）無關，
    因為單位元項彼此可交換。
    """
    n, tau = 4, 0.8
    z = _z(32, n)
    for T in (1, 4, 10):
        phi = pqk.features(z, n_qubits=n, trotter_steps=T, evolution_time=tau, entangle=False)
        th = 2 * tau * z[:, :n]
        want = torch.stack([torch.cos(th), torch.sin(th), torch.zeros_like(th)], -1).reshape(len(z), 3 * n)
        assert torch.allclose(phi, want, atol=1e-10), f"T={T} 時與閉式解不符"


def test_no_entangle_ignores_coupling_features():
    """無糾纏時，耦合用的後 n 個特徵不影響輸出（確認 C3 真的切掉了耦合）。"""
    n = 4
    z = _z(8, n)
    z2 = z.clone()
    z2[:, n:] += 1.0
    a = pqk.features(z, n_qubits=n, entangle=False)
    b = pqk.features(z2, n_qubits=n, entangle=False)
    assert torch.allclose(a, b, atol=1e-12)


def test_entangle_changes_output():
    """開啟糾纏必須改變輸出，否則 MAIN 與 C3 無從區分。"""
    n = 4
    z = _z(8, n)
    a = pqk.features(z, n_qubits=n, entangle=True)
    b = pqk.features(z, n_qubits=n, entangle=False)
    assert not torch.allclose(a, b, atol=1e-3)


def _hamiltonian(z, n):
    """與 pqk._trotter 相同的 H（環狀 Heisenberg + 單位元 Z 場）。"""
    ops, coeffs = [], []
    for j in range(n):
        ops.append(qml.PauliZ(j))
        coeffs.append(float(z[j]))
    edges = [(j, (j + 1) % n) for j in range(n)]
    if n == 2:
        edges = edges[:1]  # n=2 時環上只有一條邊
    for j, k in edges:
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            ops.append(P(j) @ P(k))
            coeffs.append(float(z[n + j]))
    return qml.dot(coeffs, ops)


def _exact_expectations(z, n, tau):
    """直接對 exp(−i τ H) 取期望值，作為 ground truth。"""
    from scipy.linalg import expm

    hm = qml.matrix(_hamiltonian(z, n), wire_order=range(n))
    psi = expm(-1j * tau * hm) @ (np.ones(2 ** n) / np.sqrt(2 ** n))  # |+⟩^n
    out = []
    for w in range(n):
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            m = qml.matrix(P(w), wire_order=range(n))
            out.append(float(np.real(psi.conj() @ (m @ psi))))
    return np.asarray(out)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_trotter_converges_to_exact_evolution(n):
    """顯式 Ising 分解必須以 O(1/T) 收斂到精確演化 exp(−i τ H)。

    這是正確性的 ground truth。**不**以 qml.TrotterProduct 為基準：它在 0.45.1
    採 exp(+iHt) 慣例（與 qml.evolve 的 exp(−iHt) 相反），且一階 Trotter 的
    算符順序不同，兩者在有限 T 下本就不相等（見 test_trotter_product_sign_convention）。
    """
    tau = 0.7
    z = _z(1, n, seed=n)[0]
    exact = _exact_expectations(z, n, tau)
    errs = []
    for T in (4, 16, 64, 256):
        got = pqk.features(z.unsqueeze(0), n_qubits=n, trotter_steps=T,
                           evolution_time=tau, entangle=True)[0].numpy()
        errs.append(float(np.abs(got - exact).max()))
    assert errs[-1] < 1e-2, f"T=256 仍未收斂：誤差 {errs[-1]:.3e}"
    assert all(b < a for a, b in zip(errs, errs[1:])), f"誤差未單調下降：{errs}"


def test_trotter_product_sign_convention():
    """記錄 PennyLane 的符號慣例，避免有人把實作「修正」成符合 TrotterProduct 預設。

    qml.TrotterProduct(H, time=τ) 收斂到 exp(+i τ H)，故須傳 time=−τ 才與本實作
    （標準的 exp(−i τ H)）一致。
    """
    from scipy.linalg import expm

    n, tau = 3, 0.7
    z = _z(1, n, seed=3)[0]
    h = _hamiltonian(z, n)
    hm = qml.matrix(h, wire_order=range(n))
    u_tp = qml.matrix(qml.TrotterProduct(h, time=tau, n=512, order=1), wire_order=range(n))
    assert np.abs(u_tp - expm(1j * tau * hm)).max() < 1e-2, "TrotterProduct 應收斂到 exp(+iτH)"
    assert np.abs(u_tp - expm(-1j * tau * hm)).max() > 0.5, "若此斷言失效，表示上游改了慣例"

    u_neg = qml.matrix(qml.TrotterProduct(h, time=-tau, n=512, order=1), wire_order=range(n))
    assert np.abs(u_neg - expm(-1j * tau * hm)).max() < 1e-2, "time=−τ 才對應本實作的慣例"


# --- 形狀與界限 ---

def test_batch_shape_is_3n():
    """批次輸出形狀為 (B, 3n)。"""
    for n in (2, 4):
        z = _z(7, n)
        assert pqk.features(z, n_qubits=n).shape == (7, 3 * n)


def test_features_are_bounded():
    """|φ| ≤ 1：Pauli 期望值的數學界限。"""
    z = _z(64, 4, seed=3)
    phi = pqk.features(z, n_qubits=4, trotter_steps=6, evolution_time=1.3)
    assert float(phi.abs().max()) <= 1.0 + 1e-9


def test_padded_dims_are_zero_safe():
    """補零維度（z 為 0）不應產生 NaN，且對應位元仍落在界限內。"""
    z = torch.zeros(4, 8, dtype=torch.float64)
    phi = pqk.features(z, n_qubits=4)
    assert torch.isfinite(phi).all() and float(phi.abs().max()) <= 1.0 + 1e-9


# --- 快取鍵（研究 D2）---

def test_cache_key_excludes_seed():
    """cache_key 不得包含種子：φ 與種子無關，這是快取可跨種子重用的前提。"""
    base = dict(data_hash="abc123", group="delta", projection="lda", n_qubits=4,
                trotter_steps=4, evolution_time=0.8, entangle=True, shots=None,
                noise_p=0.0, backend="simulator")
    k1 = pqk.cache_key(**base)
    k2 = pqk.cache_key(**base)
    assert k1 == k2
    assert "seed" not in k1
    for field, other in (("n_qubits", 8), ("entangle", False), ("evolution_time", 1.0),
                         ("group", "b1"), ("shots", 1024), ("noise_p", 0.02)):
        assert pqk.cache_key(**{**base, field: other}) != k1, f"{field} 應影響 cache_key"


def test_cache_roundtrip_and_hit(tmp_path):
    """相同 key 必須命中快取而非重算。"""
    z = _z(8, 2)
    calls = {"n": 0}
    orig = pqk.features

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    phi1, hit1 = pqk.cached_features(z, cache_dir=tmp_path, key="k1", compute=counting, n_qubits=2)
    phi2, hit2 = pqk.cached_features(z, cache_dir=tmp_path, key="k1", compute=counting, n_qubits=2)
    assert calls["n"] == 1, "第二次應命中快取，不得重算"
    assert hit1 is False and hit2 is True
    assert np.allclose(phi1, phi2)


def test_cache_invalid_on_shape_mismatch(tmp_path):
    """key 相同但形狀不符時必須重算，不得靜默使用舊資料。"""
    pqk.cached_features(_z(8, 2), cache_dir=tmp_path, key="k", n_qubits=2)
    phi, hit = pqk.cached_features(_z(5, 2), cache_dir=tmp_path, key="k", n_qubits=2)
    assert hit is False and phi.shape == (5, 6)


# --- 有限取樣與噪聲（T032／US2）---

def test_finite_shots_uses_set_shots_transform():
    """有限取樣必須以 qml.set_shots 施加，不得在 device 設 shots（0.45.1 已棄用）。"""
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        phi = pqk.features(_z(8, 2), n_qubits=2, shots=512)
    assert phi.shape == (8, 6)
    msgs = " ".join(str(w.message) for w in rec)
    assert "Setting shots on device is deprecated" not in msgs, "仍在 device 層級設 shots"


def test_finite_shots_is_noisy_but_bounded():
    """有限取樣下輸出仍有界，且與解析值不同（確認 shots 真的生效）。"""
    z = _z(16, 3, seed=5)
    exact = pqk.features(z, n_qubits=3)
    sampled = pqk.features(z, n_qubits=3, shots=256)
    assert float(sampled.abs().max()) <= 1.0 + 1e-9
    assert not torch.allclose(exact, sampled, atol=1e-3)


def test_depolarizing_noise_shrinks_features():
    """去極化噪聲應使期望值收縮（趨向 0），且保持有界。"""
    z = _z(8, 3, seed=7)
    clean = pqk.features(z, n_qubits=3)
    noisy = pqk.features(z, n_qubits=3, noise_p=0.15)
    assert float(noisy.abs().max()) <= 1.0 + 1e-9
    assert float(noisy.abs().mean()) < float(clean.abs().mean())
