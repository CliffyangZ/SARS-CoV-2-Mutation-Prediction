"""v2 投影量子核特徵圖（US3，Q4／Q6）。

針對論文承認的三個結構弱點（FR-012）：

1. **無解耦位元**：v1 在投影維度 k<2n 時零填補，使某些環上邊的耦合恆為 0，該位元完全解耦。
   v2 加入固定、與資料無關的弱耦合 J0（每條邊都有），即使某輸入維度為常數，位元仍在環上。
2. **守恆磁化使讀出少一維**：v1 的 XX+YY+ZZ 同係數＋ Z 場使總磁化守恆（初態 |+⟩ⁿ 的 ⟨ΣZ⟩≡0），
   讀出實際少一維。v2 讓 XX／YY／ZZ 係數各自獨立（YY、ZZ 取固定比例）並加橫向場 RX，破壞 U(1)／SU(2)。
3. **讀出看不到雙位元關聯**：v1 只取 1-RDM。v2 另加環上相鄰位元的 XX、YY、ZZ 相關，
   φ 維度 = 3n + 3|E|。

v1（pqk.py）**不得**改動（FR-015，有雜湊測試）。本模組完全獨立。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

DEFAULTS = dict(n_qubits=4, trotter_steps=4, evolution_time=0.8, h=0.5, j0=0.1,
                kappa_yy=0.4, kappa_zz=1.5, coupling_scale=1.0, device="default.qubit")


def edges(n):
    """環上的邊 [(j, k, 耦合維度索引)]；n=2 時只有一條邊（避免重複施加）。"""
    if n < 2:
        return []
    if n == 2:
        return [(0, 1, n)]
    return [(j, (j + 1) % n, n + j) for j in range(n)]


def n_features(n):
    """φ 的維度：3n（1-RDM）+ 3|E|（相鄰雙位元相關）。"""
    return 3 * n + 3 * len(edges(n))


def _circuit_factory(n, steps, tau, h, j0, k_yy, k_zz, cs):
    dt = tau / steps
    edge = edges(n)

    def circuit(z):
        """|+⟩ⁿ → T 步演化（場 RZ、橫向場 RX、XYZ 環耦合）→ 1-RDM 與邊相關。"""
        for w in range(n):
            qml.Hadamard(w)
        for _ in range(steps):
            for j in range(n):
                qml.RZ(2 * dt * z[..., j], wires=j)
                qml.RX(2 * dt * h * z[..., j], wires=j)
            if cs:
                for a, b, d in edge:
                    g = cs * (z[..., d] + j0)
                    qml.IsingXX(2 * dt * g, wires=[a, b])
                    qml.IsingYY(2 * dt * k_yy * g, wires=[a, b])
                    qml.IsingZZ(2 * dt * k_zz * g, wires=[a, b])
        out = [qml.expval(P(w)) for w in range(n) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]
        for a, b, _ in edge:
            out += [qml.expval(P(a) @ P(b)) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]
        return out

    return circuit


def features(z, n_qubits=None, trotter_steps=None, evolution_time=None, h=None, j0=None,
             kappa_yy=None, kappa_zz=None, coupling_scale=None, device=None, batch_size=4096):
    """計算 v2 的 φ。z 形狀 (B, 2n)，回傳 (B, 3n+3|E|)，值域 [−1,1]。

    coupling_scale=0 → 去耦合版本（對照組 V2_nocouple；讀出維度不變，雙位元項退化為乘積）。
    """
    cfg = dict(DEFAULTS)
    for k, v in dict(n_qubits=n_qubits, trotter_steps=trotter_steps, evolution_time=evolution_time,
                     h=h, j0=j0, kappa_yy=kappa_yy, kappa_zz=kappa_zz,
                     coupling_scale=coupling_scale, device=device).items():
        if v is not None:
            cfg[k] = v
    n = int(cfg["n_qubits"])
    z = torch.as_tensor(np.asarray(z), dtype=torch.float64)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.shape[1] < 2 * n:
        z = torch.hstack([z, torch.zeros(len(z), 2 * n - z.shape[1], dtype=z.dtype)])
    dev = qml.device(cfg["device"], wires=n)
    circuit = _circuit_factory(n, int(cfg["trotter_steps"]), float(cfg["evolution_time"]),
                               float(cfg["h"]), float(cfg["j0"]), float(cfg["kappa_yy"]),
                               float(cfg["kappa_zz"]), float(cfg["coupling_scale"]))
    node = qml.QNode(circuit, dev, interface="torch", diff_method=None)
    out = []
    for s in range(0, len(z), batch_size):
        got = node(z[s:s + batch_size])
        out.append(torch.stack(list(got), -1) if isinstance(got, (list, tuple)) else got)
    phi = torch.cat(out, 0) if len(out) > 1 else out[0]
    return phi.reshape(len(z), n_features(n)).to(torch.float64).clamp_(-1.0, 1.0)


def _rot(axis, ang):
    """Bloch 向量繞 axis 轉 ang 的 3×3 旋轉矩陣（與 qml.RZ／RX 的方向一致）。"""
    c, s = np.cos(ang), np.sin(ang)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # x


def closed_form_uncoupled(z, n, tau, h, steps):
    """耦合為 0 時的解析閉式（延續論文 Prop. 2 的驗證方式）。

    每個位元獨立：Bloch 向量從 (1,0,0) 起，每個 Trotter 步依序繞 z 轉 2dt·z_j、繞 x 轉 2dt·h·z_j。
    雙位元相關為乘積態的相關：⟨P_a P_b⟩ = ⟨P_a⟩⟨P_b⟩。
    """
    z = np.atleast_2d(np.asarray(z, np.float64))
    dt = tau / steps
    out = np.zeros((len(z), n_features(n)))
    for i, zi in enumerate(z):
        bloch = []
        for j in range(n):
            v = np.array([1.0, 0.0, 0.0])
            for _ in range(steps):
                v = _rot("x", 2 * dt * h * zi[j]) @ (_rot("z", 2 * dt * zi[j]) @ v)
            bloch.append(v)
        out[i, :3 * n] = np.concatenate(bloch)
        for e, (a, b, _) in enumerate(edges(n)):
            out[i, 3 * n + 3 * e:3 * n + 3 * e + 3] = bloch[a] * bloch[b]
    return out


def cache_key_v2(data_hash, group, projection, design, entangle):
    """快取鍵：含設計的全部超參數與讀出，與 v1 鍵不碰撞（鍵前綴 v2）。"""
    keep = {k: design[k] for k in sorted(design) if k in
            ("n_qubits", "trotter_steps", "evolution_time", "h", "j0", "kappa_yy", "kappa_zz")}
    payload = json.dumps(dict(v="v2", data_hash=data_hash, group=group, projection=projection,
                              design=keep, entangle=bool(entangle)), sort_keys=True, default=str)
    return "v2" + hashlib.sha256(payload.encode()).hexdigest()[:14]


def cached_features(z, cache_dir, key, n_qubits, **kw):
    """讀／寫快取；形狀（列數、v2 維度）不符時重算，不靜默使用舊資料。回傳 (φ, 是否命中)。"""
    path = Path(cache_dir) / f"phi_{key}.npz"
    want = (len(z), n_features(int(n_qubits)))
    if path.exists():
        try:
            got = np.load(path)["phi"]
            if got.shape == want:
                return got, True
        except Exception:
            pass
    phi = np.asarray(features(z, n_qubits=n_qubits, **kw).cpu().numpy(), np.float64)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, phi=phi)
    return phi, False
