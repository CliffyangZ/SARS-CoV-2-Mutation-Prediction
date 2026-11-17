"""預測頭：量子頭（PyTorch 批次狀態向量模擬）與參數量匹配的古典頭。

兩者都把縮放後的潛向量 z̃ ∈ ℝ^n 對應到 m=2n 個有界的「期望值」（|p| ≤ 1）；
後接共用的線性映射 C（見 pretrain.Predictor）才得到嵌入 q。

線路（文件 §3.2）：|0⟩ → W¹ → S₁ → W² → … → W^L → S_L → W^{L+1} → 量測
  S_l = ⊗ R_y(a_{l,i}·z̃_i + b_{l,i})；W = ⊗ R_z(α)·R_y(β)，後接 CZ 環。
讀出 ⟨Z_i⟩（n 個）與環上 ⟨Z_i Z_{i+1}⟩（n 個）。
"""
import math

import torch
from torch import nn


def sim_memory_bytes(n_qubits, n_layers, batch):
    """估算自動微分下模擬的記憶體（每個閘約保留一份 complex64 狀態）。"""
    n_gates = (n_layers + 1) * 3 * n_qubits + n_layers * n_qubits + n_layers + 2
    return int(batch) * (2 ** n_qubits) * 8 * n_gates * 2


def max_batch(n_qubits, n_layers, budget_bytes, cap=256):
    """在記憶體預算內的最大批次；連 1 也放不下則回 0（呼叫端標示略過）。"""
    per = sim_memory_bytes(n_qubits, n_layers, 1)
    return int(min(cap, budget_bytes // per))


class QuantumHead(nn.Module):
    """量子預測頭。a 固定為 1（頻譜為整數、度數 ≤ L），trainable_freq=True 時 a 可訓練。"""

    def __init__(self, n_qubits=6, layers=3, seed=None, entangle=True, trainable_freq=False):
        super().__init__()
        self.n, self.layers, self.entangle, self.trainable_freq = n_qubits, layers, entangle, trainable_freq
        g = torch.Generator().manual_seed(0 if seed is None else int(seed))
        self.w = nn.Parameter((torch.rand(layers + 1, n_qubits, 2, generator=g) * 2 - 1) * math.pi)  # (α, β)
        self.b = nn.Parameter((torch.rand(layers, n_qubits, generator=g) * 2 - 1) * 0.1)
        a0 = torch.ones(layers, n_qubits)
        if trainable_freq:
            self.a = nn.Parameter(a0)
        else:
            self.register_buffer("a", a0)
        self.register_buffer("_zmat", self._observables(n_qubits))
        self.register_buffer("_cz", self._cz_signs(n_qubits))

    @staticmethod
    def _bits(n):
        idx = torch.arange(2 ** n)
        return torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)], 1)  # (2^n, n)，第 0 位為最高位

    @classmethod
    def _observables(cls, n):
        z = 1 - 2 * cls._bits(n).float()
        zz = torch.stack([z[:, i] * z[:, (i + 1) % n] for i in range(n)], 1)
        return torch.cat([z, zz], 1)  # (2^n, 2n)

    @classmethod
    def _cz_signs(cls, n):
        bits = cls._bits(n)
        pairs = [(i, (i + 1) % n) for i in range(n)] if n >= 3 else ([(0, 1)] if n == 2 else [])
        s = torch.ones(2 ** n)
        for i, j in pairs:
            s = s * (1 - 2 * (bits[:, i] * bits[:, j]).float())
        return s

    def n_quantum_params(self):
        """量子線路可訓練參數量：3nL+2n（a 可訓練時 4nL+2n）。"""
        return sum(p.numel() for p in self.parameters())

    def _gate(self, state, U, q):
        """對第 q 個 qubit 作用 2×2（共用）或 (B,2,2)（逐樣本）矩陣。state: (B, 2,…,2)。"""
        B = state.shape[0]
        s = state.movedim(q + 1, -1)
        shp = s.shape
        if U.dim() == 2:
            out = s.reshape(-1, 2) @ U.T
        else:
            out = torch.bmm(s.reshape(B, -1, 2), U.transpose(1, 2))
        return out.reshape(shp).movedim(-1, q + 1)

    @staticmethod
    def _ry(theta):
        c, s = torch.cos(theta / 2), torch.sin(theta / 2)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)

    @staticmethod
    def _rz(alpha):
        e = torch.exp(-0.5j * alpha.to(torch.complex64))
        z = torch.zeros_like(e)
        return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)

    def _block(self, state, l):
        for q in range(self.n):
            state = self._gate(state, self._ry(self.w[l, q, 1]), q)
            state = self._gate(state, self._rz(self.w[l, q, 0]), q)
        if self.entangle:
            B = state.shape[0]
            state = (state.reshape(B, -1) * self._cz.to(state.dtype)).reshape(state.shape)
        return state

    def probabilities(self, z):
        """z: (B, n) → 計算基底機率 (B, 2^n)。"""
        B = z.shape[0]
        state = torch.zeros(B, *([2] * self.n), dtype=torch.complex64, device=z.device)
        state.view(B, -1)[:, 0] = 1.0
        for l in range(self.layers):
            state = self._block(state, l)
            for q in range(self.n):
                state = self._gate(state, self._ry(self.a[l, q] * z[:, q] + self.b[l, q]), q)
        state = self._block(state, self.layers)
        return (state.reshape(B, -1).abs() ** 2)

    def expectations(self, z, shots=None, noise=0.0, generator=None):
        """z: (B, n) → (B, 2n) 期望值。shots 給定時對機率做多項取樣；noise 為去極化縮減（乘 (1-noise)）。"""
        p = self.probabilities(z)
        if shots:
            p = p / p.sum(1, keepdim=True)
            counts = torch.multinomial(p.detach().clamp_min(0), shots, replacement=True, generator=generator)
            p = torch.zeros_like(p).scatter_add_(1, counts, torch.ones_like(counts, dtype=p.dtype)) / shots
        out = p @ self._zmat
        return out * (1.0 - noise) if noise else out

    def forward(self, z, **kw):
        """前向，等同 expectations。"""
        return self.expectations(z, **kw)


class ClassicalHead(nn.Module):
    """單隱藏層 MLP，輸入 n 維、輸出 m 維（tanh 有界）；隱藏寬度取使參數量最接近 target_params 者。"""

    def __init__(self, n, m, target_params, seed=None):
        super().__init__()
        self.n, self.m = n, m
        h = min(range(1, 4096), key=lambda k: abs(n * k + k + k * m + m - target_params))
        if seed is not None:
            torch.manual_seed(int(seed))
        self.net = nn.Sequential(nn.Linear(n, h), nn.Tanh(), nn.Linear(h, m), nn.Tanh())

    def forward(self, z, **kw):
        """前向，等同 expectations。"""
        return self.net(z)

    expectations = forward


class MultiQuantumHead(QuantumHead):
    """H 個參數獨立的量子頭（§3.2 修訂：並聯加寬取代加深）。

    向量化成 B·H 個狀態一起模擬：輸入 (B, H·n)（第 h 個頭取第 h 段 n 個角度），輸出 (B, H·2n)。
    每個頭的線路與 QuantumHead 相同；H=1 時兩者在相同參數下輸出一致。
    """

    def __init__(self, n_qubits=8, layers=2, n_heads=8, seed=None, entangle=False, trainable_freq=False):
        nn.Module.__init__(self)
        self.n, self.layers, self.H, self.entangle, self.trainable_freq = n_qubits, layers, n_heads, entangle, trainable_freq
        g = torch.Generator().manual_seed(0 if seed is None else int(seed))
        self.w = nn.Parameter((torch.rand(n_heads, layers + 1, n_qubits, 2, generator=g) * 2 - 1) * math.pi)
        self.b = nn.Parameter((torch.rand(n_heads, layers, n_qubits, generator=g) * 2 - 1) * 0.1)
        a0 = torch.ones(n_heads, layers, n_qubits)
        if trainable_freq:
            self.a = nn.Parameter(a0)
        else:
            self.register_buffer("a", a0)
        self.register_buffer("_zmat", self._observables(n_qubits))
        self.register_buffer("_cz", self._cz_signs(n_qubits))

    @staticmethod
    def _rep(v, B):
        """(H,) → (B·H,)：每個樣本重複一份各頭的參數。"""
        return v.unsqueeze(0).expand(B, -1).reshape(-1)

    def _block(self, state, l, B):
        for q in range(self.n):
            state = self._gate(state, self._ry(self._rep(self.w[:, l, q, 1], B)), q)
            state = self._gate(state, self._rz(self._rep(self.w[:, l, q, 0], B)), q)
        if self.entangle:
            N = state.shape[0]
            state = (state.reshape(N, -1) * self._cz.to(state.dtype)).reshape(state.shape)
        return state

    def probabilities(self, z):
        """H 個並聯頭的測量機率：z (B, H·n) → (B·H, 2^n)，把頭維度攤進批次維度一次模擬。"""
        B = z.shape[0]
        zz = z.reshape(B * self.H, self.n)
        state = torch.zeros(B * self.H, *([2] * self.n), dtype=torch.complex64, device=z.device)
        state.view(B * self.H, -1)[:, 0] = 1.0
        for l in range(self.layers):
            state = self._block(state, l, B)
            for q in range(self.n):
                th = self._rep(self.a[:, l, q], B) * zz[:, q] + self._rep(self.b[:, l, q], B)
                state = self._gate(state, self._ry(th), q)
        state = self._block(state, self.layers, B)
        return state.reshape(B * self.H, -1).abs() ** 2

    def expectations(self, z, shots=None, noise=0.0, generator=None):
        """H 個並聯頭的讀出：z (B, H·n) → (B, H·2n)，各頭的 2n 維期望值串接。"""
        B = z.shape[0]
        out = QuantumHead.expectations(self, z, shots=shots, noise=noise, generator=generator)
        return out.reshape(B, self.H * 2 * self.n)


class WideClassicalHead(nn.Module):
    """寬度匹配古典頭（§5 修訂）：in → m → m，tanh 有界；隱藏層寬度 = 讀出維度 m，不做參數量匹配。"""

    def __init__(self, in_dim, m, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.net = nn.Sequential(nn.Linear(in_dim, m), nn.Tanh(), nn.Linear(m, m), nn.Tanh())

    def forward(self, z, **kw):
        """前向。"""
        return self.net(z)

    expectations = forward
