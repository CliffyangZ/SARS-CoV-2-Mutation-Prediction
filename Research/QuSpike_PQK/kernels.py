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
    """檢查並修正半正定性（FR-015）。回傳 (修正後的核, 診斷字典)。

    先用 Cholesky 檢查 K + tol·I 是否正定：成功 ⇔ 最小特徵值 > −tol（與舊版的判定準則等價），
    成本約為完整特徵分解的 1/10（008 Q7）。失敗才走特徵分解、截斷負特徵值並重新正規化——
    修正一律記錄在診斷中，不得靜默。
    """
    km = np.asarray(k, np.float64)
    km = 0.5 * (km + km.T)
    try:
        np.linalg.cholesky(km + tol * np.eye(len(km)))
        return np.asarray(k, np.float32), {"psd_violation": False, "psd_correction": None}
    except np.linalg.LinAlgError:
        pass
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


class RFFMap:
    """隨機傅立葉特徵映射（可對任意列求值，供推論時對所有評估列計算）。

    頻率取自 Trotter 演化可達的尺度（2·τ），cos／sin 成對填滿 n_features 維。
    抽樣順序為先 w 後 b，與舊版 `rff_surrogate` 完全一致，故相同種子輸出逐元相同。
    """

    def __init__(self, d, n_features, seed=0, evolution_time=0.8):
        rng = np.random.default_rng(int(seed))
        self.n_features = int(n_features)
        half = (self.n_features + 1) // 2
        self.w = rng.normal(0.0, 2.0 * float(evolution_time), size=(int(d), half))
        self.b = rng.uniform(0.0, 2 * np.pi, size=half)

    def __call__(self, x):
        """cos／sin 成對，輸出有界於 [−1,1]。"""
        p = np.asarray(x, np.float64) @ self.w + self.b
        out = np.concatenate([np.cos(p), np.sin(p)], axis=1)
        return out[:, :self.n_features].astype(np.float64)


def rff_surrogate(z_train, z_eval, n_features, seed=0, evolution_time=0.8, trotter_steps=4):
    """對照組 C2：古典隨機傅立葉特徵替身。回傳 (訓練特徵, 評估特徵)。"""
    m = RFFMap(np.asarray(z_train).shape[1], n_features, seed=seed, evolution_time=evolution_time)
    return m(z_train), m(z_eval)


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


class NystromMap:
    """顯式 Nyström 特徵映射（US4，研究 D8）：Φ(x) = K(x, 地標) · U Λ^{-1/2}。

    與舊 `nystrom_kernel` 的差別：舊版仍會算出完整的 N×N 近似核，沒有真的省記憶體；
    本版只產生 (N, m') 的特徵，後接線性 SVM，時間與記憶體都是 O(N·m)。
    地標抽樣只依 seed，不依資料標籤。
    """

    def __init__(self, phi_train, gamma, m=512, seed=0):
        x = np.asarray(phi_train, np.float64)
        n = len(x)
        m = int(min(m, n))
        idx = np.random.default_rng(int(seed)).permutation(n)[:m]
        self.land = x[idx]
        self.gamma = float(gamma)
        kmm = rbf(self.land, self.land, gamma).astype(np.float64)
        ev, vec = np.linalg.eigh(0.5 * (kmm + kmm.T))
        keep = ev > 1e-10
        self.proj = vec[:, keep] / np.sqrt(ev[keep])
        self.m = m

    def transform(self, phi):
        """對任意列求特徵，形狀 (n, m')。"""
        k = rbf(phi, self.land, self.gamma).astype(np.float64)
        return k @ self.proj


def nystrom_features(phi_train, phi_eval, gamma, m=512, seed=0):
    """回傳 (訓練特徵, 評估特徵)。"""
    mp = NystromMap(phi_train, gamma, m=m, seed=seed)
    return mp.transform(phi_train), mp.transform(phi_eval)


def phi_digest(phi):
    """φ 內容的短雜湊，供 Gram 快取當鍵（φ 與種子無關，故跨種子命中）。"""
    import hashlib

    return hashlib.blake2b(np.ascontiguousarray(phi, np.float64).tobytes(), digest_size=10).hexdigest()


GRAM_CACHE_MAX_N = 10000  # 單一 Gram 最大約 400 MB；更大者不落地（磁碟有限）


def gram_cached(phi, gamma, cache_dir=None):
    """精確 RBF Gram（已做半正定檢查）。回傳 (K, psd 診斷, 是否命中快取)。

    γ（中位數啟發式，固定 rng）與 φ 都與種子無關，所以 5 個種子只需算一次 Gram 與 O(N³) 的特徵分解
    （FR-017）。cache_dir=None 時不快取。
    """
    import json
    from pathlib import Path

    if cache_dir is not None:
        d = Path(cache_dir) / "gram"
        key = f"{phi_digest(phi)}_{float(gamma):.8g}"
        kp, jp = d / f"{key}.npy", d / f"{key}.json"
        if kp.exists() and jp.exists():
            try:
                return np.load(kp), json.loads(jp.read_text()), True
            except Exception:
                pass
    k, psd = ensure_psd(rbf(phi, None, gamma))
    if cache_dir is not None and len(k) <= GRAM_CACHE_MAX_N:
        import os

        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.{os.getpid()}.npy"   # 原子寫入：多個種子行程可能同時算同一份 Gram
        np.save(tmp, k)
        jp.write_text(json.dumps(psd))
        os.replace(tmp, kp)
    return k, psd, False
