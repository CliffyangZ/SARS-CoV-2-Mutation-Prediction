"""投影量子核的量子特徵圖（T011／T012）。

編碼：n 個量子位元，初態 |+⟩^n，一階 Trotter 演化
    H(z) = Σ_j z_j Z_j  +  Σ_j z_{n+j} (X_jX_{j+1} + Y_jY_{j+1} + Z_jZ_{j+1})   （環狀）
投影：每個位元取 1-RDM 的 ⟨X⟩,⟨Y⟩,⟨Z⟩ → φ ∈ [−1,1]^{3n}（Huang et al. 2021 的投影量子核）。

兩個實作決策（見 specs/007-pqk-pennylane-ibm/research.md）：

D1 **不用 qml.TrotterProduct**：它搭配資料相依的 Hamiltonian 係數無法參數廣播
   （批次輸入會丟 RuntimeError）。改以顯式 RZ + IsingXX/YY/ZZ 分解，閘角度成為
   可廣播參數。實測 default.qubit 0.671 ms/樣本，而 lightning.qubit 慢 15 倍
   （它對批次逐筆迴圈）。

D2 **φ 與種子無關**：本模組不訓練任何參數（FR-007），故 cache_key 不含種子，
   七組設定 × 3 種子共用同一份快取。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

DEFAULTS = dict(n_qubits=4, trotter_steps=4, evolution_time=0.8, entangle=True,
                shots=None, noise_p=0.0, device="default.qubit")


def _trotter(z, n, steps, tau, entangle):
    """一階 Trotter 演化的顯式閘分解。z 的後 n 維為耦合強度，僅在 entangle 時使用。"""
    dt = tau / steps
    for _ in range(steps):
        for j in range(n):
            qml.RZ(2 * dt * z[..., j], wires=j)
        if entangle and n > 1:
            for j in range(n):
                k = (j + 1) % n
                a = 2 * dt * z[..., n + j]
                qml.IsingXX(a, wires=[j, k])
                qml.IsingYY(a, wires=[j, k])
                qml.IsingZZ(a, wires=[j, k])
                if n == 2:
                    break  # n=2 時環上只有一條邊，避免重複施加


def _make_qnode(n, steps, tau, entangle, shots, noise_p, device):
    """建 QNode。噪聲走密度矩陣裝置；有限取樣用 set_shots 轉換（device 層級已棄用）。"""
    dev_name = "default.mixed" if noise_p > 0 else device
    dev = qml.device(dev_name, wires=n)
    obs = [P(w) for w in range(n) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]

    def circuit(z):
        """線路本體：|+⟩^n → Trotter 演化 →（可選）去極化 → 1-RDM 期望值。"""
        for w in range(n):
            qml.Hadamard(w)
        _trotter(z, n, steps, tau, entangle)
        if noise_p > 0:
            for w in range(n):
                qml.DepolarizingChannel(noise_p, wires=w)
        return [qml.expval(o) for o in obs]

    node = qml.QNode(circuit, dev, interface="torch", diff_method=None)
    return qml.set_shots(node, shots=shots) if shots else node


def features(z, n_qubits=None, trotter_steps=None, evolution_time=None, entangle=None,
             shots=None, noise_p=None, device=None, batch_size=4096):
    """計算 φ。z 形狀 (B, 2n)，回傳 (B, 3n) 的 torch tensor，值域 [−1,1]。"""
    cfg = dict(DEFAULTS)
    for k, v in dict(n_qubits=n_qubits, trotter_steps=trotter_steps, evolution_time=evolution_time,
                     entangle=entangle, shots=shots, noise_p=noise_p, device=device).items():
        if v is not None:
            cfg[k] = v
    n = int(cfg["n_qubits"])
    z = torch.as_tensor(np.asarray(z), dtype=torch.float64)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.shape[1] < 2 * n:
        z = torch.hstack([z, torch.zeros(len(z), 2 * n - z.shape[1], dtype=z.dtype)])

    node = _make_qnode(n, int(cfg["trotter_steps"]), float(cfg["evolution_time"]),
                       bool(cfg["entangle"]), cfg["shots"], float(cfg["noise_p"]), cfg["device"])
    out = []
    for s in range(0, len(z), batch_size):
        got = node(z[s:s + batch_size])
        out.append(torch.stack(list(got), -1) if isinstance(got, (list, tuple)) else got)
    phi = torch.cat(out, 0) if len(out) > 1 else out[0]
    return phi.reshape(len(z), 3 * n).to(torch.float64).clamp_(-1.0, 1.0)


def cache_key(data_hash, group, projection, n_qubits, trotter_steps, evolution_time,
              entangle, shots, noise_p, backend="simulator"):
    """量子特徵的快取鍵。**刻意不含種子**（研究 D2）。"""
    payload = json.dumps(dict(data_hash=data_hash, group=group, projection=projection,
                              n_qubits=int(n_qubits), trotter_steps=int(trotter_steps),
                              evolution_time=round(float(evolution_time), 8),
                              entangle=bool(entangle),
                              shots=None if not shots else int(shots),
                              noise_p=round(float(noise_p), 8), backend=backend),
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cached_features(z, cache_dir, key, compute=None, **kw):
    """讀／寫快取。回傳 (φ ndarray, 是否命中快取)。形狀不符時重算（不靜默使用舊資料）。"""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"phi_{key}.npz"
    want = (len(z), 3 * int(kw.get("n_qubits", DEFAULTS["n_qubits"])))
    if path.exists():
        try:
            got = np.load(path)["phi"]
            if got.shape == want:
                return got, True
        except Exception:
            pass
    fn = compute or features
    phi = np.asarray(fn(z, **kw).cpu().numpy(), np.float64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, phi=phi)
    return phi, False
