"""核、診斷與古典替身（T019–T021）。

`rff_surrogate` 是對照組 C2：依 Schreiber et al.（arXiv:2206.11740），大類
re-uploading／演化型量子模型存在古典替身。本實作以 Trotter 演化可達的頻率抽樣，
並用 cos／sin 基底（與量子期望值同樣有界），特徵數對齊量子讀出的 3n，
使「量子特徵有用」這個宣稱能被乾淨檢定。
"""
import numpy as np


def auto_gamma(phi):
    """中位數啟發式：gamma = 1 / (2 · median(‖φ_i − φ_j‖²))。"""
    x = np.asarray(phi, np.float64)
    idx = np.random.default_rng(0).permutation(len(x))[:512]
    s = x[idx]
    d2 = np.maximum(((s[:, None, :] - s[None, :, :]) ** 2).sum(-1), 0.0)
    med = float(np.median(d2[np.triu_indices(len(s), 1)])) if len(s) > 1 else 1.0
    return 1.0 / (2.0 * max(med, 1e-12))


def rbf(a, b=None, gamma=1.0):
    """RBF 核 exp(−γ‖a−b‖²)。b 為 None 時回傳 a 的自核（對稱、對角 1）。"""
    x = np.asarray(a, np.float64)
    y = x if b is None else np.asarray(b, np.float64)
    d2 = (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2.0 * (x @ y.T)
    return np.exp(-float(gamma) * np.maximum(d2, 0.0)).astype(np.float32)


def ensure_psd(k, tol=1e-8):
    """檢查並修正半正定性（FR-015）。回傳 (修正後的核, 診斷字典)。"""
    km = np.asarray(k, np.float64)
    km = 0.5 * (km + km.T)
    ev, vec = np.linalg.eigh(km)
    if ev.min() >= -tol:
        return np.asarray(k, np.float32), {"psd_violation": False, "psd_correction": None}
    fixed = (vec * np.clip(ev, 0.0, None)) @ vec.T
    d = np.sqrt(np.clip(np.diag(fixed), 1e-12, None))
    fixed = fixed / np.outer(d, d)  # 重新正規化，使對角回到 1
    return fixed.astype(np.float32), {
        "psd_violation": True,
        "psd_correction": f"特徵值截斷（最小特徵值 {ev.min():.3e} → 0）後重新正規化對角",
    }


def diagnostics(k):
    """核值分佈診斷：off-diagonal 的平均與變異（Huang 指出的核集中問題）。"""
    km = np.asarray(k, np.float64)
    n = len(km)
    if n < 2:
        return {"offdiag_mean": float("nan"), "offdiag_std": float("nan")}
    iu = np.triu_indices(n, 1)
    off = km[iu]
    return {"offdiag_mean": float(off.mean()), "offdiag_std": float(off.std())}


def rff_surrogate(z_train, z_eval, n_features, seed=0, evolution_time=0.8, trotter_steps=4):
    """對照組 C2：古典隨機傅立葉特徵替身。

    頻率取自 Trotter 演化可達的尺度（2·τ 為單位元項的總角度係數，見 pqk 的閉式解），
    以 cos／sin 成對填滿 n_features 維，使輸出有界於 [−1,1] 且維度與量子讀出一致。
    """
    rng = np.random.default_rng(int(seed))
    x_tr = np.asarray(z_train, np.float64)
    x_ev = np.asarray(z_eval, np.float64)
    d = x_tr.shape[1]
    half = (int(n_features) + 1) // 2
    scale = 2.0 * float(evolution_time)
    w = rng.normal(0.0, scale, size=(d, half))
    b = rng.uniform(0.0, 2 * np.pi, size=half)

    def feat(x):
        """隨機傅立葉特徵：cos／sin 成對，輸出有界於 [−1,1]。"""
        p = x @ w + b
        out = np.concatenate([np.cos(p), np.sin(p)], axis=1)
        return out[:, :int(n_features)].astype(np.float64)

    return feat(x_tr), feat(x_ev)


def nystrom_kernel(phi_train, phi_eval, gamma, m=2048, seed=0):
    """Nyström 近似（超出記憶體預算時的後備）。回傳近似的 eval×train 核。"""
    x = np.asarray(phi_train, np.float64)
    n = len(x)
    m = int(min(m, n))
    idx = np.random.default_rng(int(seed)).permutation(n)[:m]
    land = x[idx]
    kmm = rbf(land, land, gamma).astype(np.float64)
    ev, vec = np.linalg.eigh(kmm)
    keep = ev > 1e-10
    inv = (vec[:, keep] / ev[keep]) @ vec[:, keep].T
    k_ev = rbf(phi_eval, land, gamma).astype(np.float64)
    k_tr = rbf(x, land, gamma).astype(np.float64)
    return (k_ev @ inv @ k_tr.T).astype(np.float32)
