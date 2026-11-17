"""量子模擬器：與 PennyLane 對照、縮放不重疊、匹配古典頭、記憶體估算。"""
import numpy as np
import pytest
import torch

from Research.QuSpike_JEPA.encoder import Encoder
from Research.QuSpike_JEPA.qhead import QuantumHead, max_batch, sim_memory_bytes


def test_matches_pennylane():
    qml = pytest.importorskip("pennylane")
    n, L = 4, 2
    h = QuantumHead(n, L, seed=3, entangle=True)
    z = torch.rand(5, n) * 2 - 1
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ(zz):
        def block(l):
            for q in range(n):
                qml.RY(float(h.w[l, q, 1]), wires=q)
                qml.RZ(float(h.w[l, q, 0]), wires=q)
            for q in range(n):
                qml.CZ(wires=[q, (q + 1) % n])
        for l in range(L):
            block(l)
            for q in range(n):
                qml.RY(float(h.a[l, q]) * float(zz[q]) + float(h.b[l, q]), wires=q)
        block(L)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)] + [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n)) for i in range(n)]

    ref = np.array([[float(v) for v in circ(zi)] for zi in z])
    got = h.expectations(z).detach().numpy()
    assert np.abs(ref - got).max() < 1e-5


def test_scaling_keeps_extremes_distinct():
    zt = Encoder.scale(torch.tensor([-50.0, 50.0]))
    h = QuantumHead(4, 2, seed=1)
    out = h.expectations(zt.unsqueeze(1).repeat(1, 4))
    assert (out[0] - out[1]).abs().max() > 1e-3
    assert zt.abs().max() <= np.pi / 2 + 1e-6


def test_seeded_head_is_deterministic():
    z = torch.rand(3, 4)
    assert torch.equal(QuantumHead(4, 2, seed=9).expectations(z), QuantumHead(4, 2, seed=9).expectations(z))


def test_shots_converge_and_noise_shrinks():
    h = QuantumHead(4, 2, seed=2)
    z = torch.rand(2, 4)
    exact = h.expectations(z)
    est = h.expectations(z, shots=200000, generator=torch.Generator().manual_seed(0))
    assert (est - exact).abs().max() < 0.02
    assert torch.allclose(h.expectations(z, noise=0.1), 0.9 * exact, atol=1e-6)


def test_memory_estimate_grows_and_batch_shrinks():
    assert sim_memory_bytes(10, 3, 64) > sim_memory_bytes(6, 3, 64)
    assert max_batch(10, 3, 2 ** 26) < max_batch(6, 3, 2 ** 26)
    assert max_batch(20, 3, 2 ** 20) == 0
