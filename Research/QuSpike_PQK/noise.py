"""有限取樣與去極化噪聲的掉點曲線（T033–T036）。

在本階段不送機的決定下，這是**唯一**能回答「硬體上還成立嗎」的證據來源。

兩個實作約束（研究 D5）：
- 有限取樣必須以 `qml.set_shots` 轉換施加（device 層級設 shots 在 0.45.1 已棄用）；
  這點由 pqk.features 負責，本模組只傳參數。
- 密度矩陣模擬慢 33–70 倍（實測 46.6 ms/樣本），故只在**分層子集**上執行。
"""
import numpy as np
import pennylane as qml
import torch

from . import criteria, features as F, pipeline


def stratified_subset(y, size, seed=0):
    """分層抽樣：每類別按比例取樣，至少 1 筆。回傳索引。"""
    y = np.asarray(y)
    rng = np.random.default_rng(int(seed))
    classes, counts = np.unique(y, return_counts=True)
    size = min(int(size), len(y))
    take = np.maximum((counts / counts.sum() * size).round().astype(int), 1)
    out = []
    for c, k in zip(classes, take):
        idx = np.where(y == c)[0]
        out.append(rng.permutation(idx)[:min(k, len(idx))])
    return np.sort(np.concatenate(out))


def subset_featureset(fs, size, seed=0):
    """把 FeatureSet 縮成分層子集（train 與 test 各自分層）。"""
    tr = stratified_subset(fs.y_train, size, seed)
    te = stratified_subset(fs.y_test, max(size // 2, 1), seed + 1)
    return F.FeatureSet(**{**fs.__dict__, "x_train": fs.x_train[tr], "y_train": fs.y_train[tr],
                           "x_test": fs.x_test[te], "y_test": fs.y_test[te],
                           "month_train": fs.month_train[tr] if fs.month_train is not None else None})


def scan(fs, groups, cfg, cache_dir, shots_list=(None, 8192, 1024, 256),
         noise_list=(0.0, 0.005, 0.02), controls=("MAIN", "C3", "C0"), seeds=(0,),
         subset=2000, tune=False):
    """掉點掃描。回傳 (每個設定點的紀錄清單, 基準判準)。"""
    from Research.lineage_baselines.metrics import compute_metrics

    sub = subset_featureset(fs, subset)
    sub_groups = F.make_groups(sub)
    rows, by_point = [], {}
    for shots in shots_list:
        for p in noise_list:
            c = dict(cfg, shots=shots, noise_p=p)
            scores = {}
            for ctrl in controls:
                vals = []
                for seed in seeds:
                    pred, proba, _ = pipeline.evaluate_control(ctrl, sub, sub_groups, c, seed,
                                                               cache_dir, tune=tune)
                    vals.append(compute_metrics(sub.y_test, pred, proba, len(sub.labels))["f1_macro"])
                scores[ctrl] = vals
            recs = criteria.evaluate(scores)
            key = (shots, p)
            by_point[key] = recs
            for ctrl, vals in scores.items():
                m, s = criteria.mean_std(vals)
                rows.append(dict(shots=shots, noise_p=p, control=ctrl, f1_macro=m, std=s,
                                 subset_size=int(len(sub.y_test))))
    return rows, by_point


def drop_curve(rows, baseline=(None, 0.0)):
    """相對無噪聲基準的掉點。"""
    base = {r["control"]: r["f1_macro"] for r in rows
            if (r["shots"], r["noise_p"]) == baseline}
    out = []
    for r in rows:
        b = base.get(r["control"])
        out.append(dict(r, drop=None if b is None else r["f1_macro"] - b))
    return out


def min_shots_for_same_verdict(by_point, claim="量子性有用（糾纏有作用）", noise_p=0.0):
    """維持與無噪聲相同判定所需的最低取樣次數（SC-011）。

    回傳 (最低 shots 或 None, 說明)。達不到時回傳 None 並說明，不得靜默略過。
    """
    def v(recs):
        """從判準紀錄中取出指定宣稱的判定。"""
        for r in recs:
            if r["claim"] == claim:
                return r["verdict"]
        return None

    ref = v(by_point.get((None, noise_p), []))
    if ref is None:
        return None, "缺少無噪聲基準，無法判定"
    cands = sorted((s for (s, p) in by_point if p == noise_p and s is not None))
    for s in cands:
        if v(by_point[(s, noise_p)]) == ref:
            return int(s), f"在 shots={s} 時判定與無噪聲一致（{ref}）"
    top = cands[-1] if cands else None
    return None, (f"即使在最高取樣次數 shots={top} 下，判定仍與無噪聲不同"
                  f"（無噪聲為 {ref}）——此為必須明確說明的情況（SC-011）")


# --- 008 US5：有資訊量的雜訊型態（Q8，FR-018／FR-019）---
#
# 終端去極化讓每個 Pauli 期望值乘上同一個常數 (1−4p/3)；中位數啟發式重估 γ 後 Gram 完全不變，
# 所以掃描沒有資訊量（論文 Prop. 4）。下列四種型態不會被這個抵消：
#   amplitude_damping  加性偏移（⟨Z⟩ 向 +1 漂移），不是均勻縮放
#   readout_asym       E' = E(1−p_a−p_b) + (p_b−p_a)，含常數偏移項；解析後處理，不需密度矩陣
#   interleaved_depol  雜訊夾在糾纏閘之間，與後續閘不可交換
#   over_rotation      系統性酉偏差，改變 φ 的函數形式
# 為了不改動 v1 的 pqk.py（有雜湊凍結測試），線路在此重建，與 pqk._trotter 同一分解。

KINDS = ("terminal_depol", "amplitude_damping", "readout_asym", "interleaved_depol", "over_rotation")


def qubit_multipliers(n_qubits, spread=0.5):
    """各量子位元讀出錯誤率的倍率，線性分佈於 [1−spread, 1+spread]（硬體上各位元的讀出品質不同）。"""
    n = int(n_qubits)
    return np.ones(1) if n == 1 else np.linspace(1 - spread, 1 + spread, n)


def apply_readout_asym(phi, p_a, p_b, n_qubits=None):
    """非對稱讀出錯誤的解析後處理：P(讀到 −1|真 +1)=p_a、P(讀到 +1|真 −1)=p_b。

    E' = E(1−p_a−p_b) + (p_b−p_a)。

    **發現（008 實測，推翻 research D9 的預期）**：若所有位元的 (p_a, p_b) 相同，
    這只是對所有座標同一個縮放加同一個平移，RBF 核與中位數 γ 完全抵消它——
    和終端去極化一樣沒有資訊量。因此給定 n_qubits 時，各位元的錯誤率乘上
    `qubit_multipliers` 的倍率（符合真實裝置各位元不同），不變性才會消失。
    n_qubits=None 保留均勻版本，供回歸測試證明上述抵消。
    """
    phi = np.asarray(phi, np.float64)
    if n_qubits is None:
        return phi * (1.0 - p_a - p_b) + (p_b - p_a)
    m = np.repeat(qubit_multipliers(n_qubits), 3)       # (3n,)：每位元的 X、Y、Z 共用倍率
    return phi * (1.0 - m * (p_a + p_b)) + m * (p_b - p_a)


def _noisy_circuit(n, steps, tau, entangle, kind, strength):
    dt = tau / steps

    def circuit(z):
        """與 pqk 的 v1 線路相同的分解，並依 kind 插入雜訊。"""
        for w in range(n):
            qml.Hadamard(w)
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
                    if kind == "interleaved_depol":
                        qml.DepolarizingChannel(strength, wires=j)
                        qml.DepolarizingChannel(strength, wires=k)
                    if n == 2:
                        break
            if kind == "amplitude_damping":
                for w in range(n):
                    qml.AmplitudeDamping(strength, wires=w)
            if kind == "over_rotation":
                for w in range(n):
                    qml.RX(strength, wires=w)
        return [qml.expval(P(w)) for w in range(n) for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]

    return circuit


def shot_noise(phi, shots, seed=0):
    """有限取樣：每個 Pauli 以 shots 次 ±1 抽樣估計，E → (2·Binomial(N,(1+E)/2) − N)/N。

    讓雜訊型態（密度矩陣給精確期望值）也能掃取樣次數，回答「維持判定所需最低取樣次數」。
    """
    rng = np.random.default_rng(int(seed))
    p = np.clip((np.asarray(phi, np.float64) + 1.0) / 2.0, 0.0, 1.0)
    return (2.0 * rng.binomial(int(shots), p) - int(shots)) / int(shots)


def noisy_compute(kind, strength, readout_pb=0.0):
    """回傳可傳給 `pqk.cached_features(compute=...)` 的函式 fn(z, **kw) → torch (B, 3n)。

    kw 與 `pqk.features` 相同（n_qubits、trotter_steps、evolution_time、entangle…）。
    readout_asym 以 strength 為 p_a、readout_pb 為 p_b。
    """
    from . import pqk

    def fn(z, n_qubits=4, trotter_steps=4, evolution_time=0.8, entangle=True, device="default.qubit",
           batch_size=1024, shots=None, **_):
        """計算該雜訊型態下的 φ（torch (B, 3n)），可選有限取樣。"""
        out = _fn(z, n_qubits, trotter_steps, evolution_time, entangle, device, batch_size)
        if shots:
            out = torch.as_tensor(shot_noise(out.numpy(), shots)).clamp_(-1.0, 1.0)
        return out

    def _fn(z, n_qubits, trotter_steps, evolution_time, entangle, device, batch_size):
        if kind == "readout_asym":
            clean = pqk.features(z, n_qubits=n_qubits, trotter_steps=trotter_steps,
                                 evolution_time=evolution_time, entangle=entangle, device=device)
            return torch.as_tensor(apply_readout_asym(clean.numpy(), strength, readout_pb, n_qubits=n_qubits))
        n = int(n_qubits)
        z = torch.as_tensor(np.asarray(z), dtype=torch.float64)
        if z.shape[1] < 2 * n:
            z = torch.hstack([z, torch.zeros(len(z), 2 * n - z.shape[1], dtype=z.dtype)])
        mixed = kind in ("amplitude_damping", "interleaved_depol")
        dev = qml.device("default.mixed" if mixed else device, wires=n)
        node = qml.QNode(_noisy_circuit(n, int(trotter_steps), float(evolution_time), bool(entangle),
                                        kind, float(strength)), dev, interface="torch", diff_method=None)
        out = []
        for s in range(0, len(z), batch_size):
            got = node(z[s:s + batch_size])
            out.append(torch.stack(list(got), -1) if isinstance(got, (list, tuple)) else got)
        return torch.cat(out, 0).reshape(len(z), 3 * n).to(torch.float64).clamp_(-1.0, 1.0)

    return fn


def gram_deviation(phi_noisy, phi_clean, n=1500, seed=0):
    """相對無雜訊的 Gram 偏差：兩者各以自己的中位數 γ 建 RBF 核，取同一組列的最大絕對差。

    終端去極化應回傳 ≈ 0（FR-019 回歸檢查）；其餘型態應明顯 > 0。
    """
    from . import kernels as K

    idx = np.random.default_rng(seed).permutation(len(phi_clean))[:n]
    a, b = np.asarray(phi_noisy)[idx], np.asarray(phi_clean)[idx]
    ka = K.rbf(a, None, K.auto_gamma(a)).astype(np.float64)
    kb = K.rbf(b, None, K.auto_gamma(b)).astype(np.float64)
    return float(np.abs(ka - kb).max())


def kind_gram_report(z, kind, strength, n_qubits=4, readout_pb=0.0, **kw):
    """對給定輸入 z 計算某型態的 Gram 偏差（供 CLI 與回歸測試共用）。"""
    from . import pqk

    clean = pqk.features(z, n_qubits=n_qubits, **{k: v for k, v in kw.items()
                         if k in ("trotter_steps", "evolution_time", "entangle")}).numpy()
    if kind == "terminal_depol":
        noisy = pqk.features(z, n_qubits=n_qubits, noise_p=strength, **{k: v for k, v in kw.items()
                             if k in ("trotter_steps", "evolution_time", "entangle")}).numpy()
    else:
        noisy = noisy_compute(kind, strength, readout_pb)(z, n_qubits=n_qubits, **kw).numpy()
    return dict(kind=kind, strength=strength, gram_deviation=gram_deviation(noisy, clean))


def noise_cache_suffix(kind, strength, readout_pb=0.0):
    """快取鍵後綴：型態與強度（避免與 v1 的鍵或不同型態互相碰撞，研究 D9）。"""
    return f"{kind}_{round(float(strength), 8)}_{round(float(readout_pb), 8)}"


def scan_kind(fs, groups, cfg, cache_dir, kind, strengths, shots_list=(None,), readout_pb=0.0,
              controls=("MAIN", "C3", "C0"), seeds=(0,), subset=2000, tune=False):
    """對單一雜訊型態掃強度×取樣次數。回傳 (紀錄, by_point)；by_point 的鍵為 (shots, strength)。"""
    from Research.lineage_baselines.metrics import compute_metrics

    sub = subset_featureset(fs, subset)
    sub_groups = F.make_groups(sub)
    rows, by_point = [], {}
    for shots in shots_list:
        for st in strengths:
            c = dict(cfg, shots=shots, noise_p=0.0)
            if kind == "terminal_depol":
                c["noise_p"] = st
            elif st:
                c.update(noise_kind=kind, noise_params=dict(strength=st, readout_pb=readout_pb))
            scores = {}
            for ctrl in controls:
                vals = []
                for seed in seeds:
                    pred, proba, _ = pipeline.evaluate_control(ctrl, sub, sub_groups, c, seed, cache_dir,
                                                               tune=tune)
                    vals.append(compute_metrics(sub.y_test, pred, proba, len(sub.labels))["f1_macro"])
                scores[ctrl] = vals
            by_point[(shots, st)] = criteria.evaluate(scores)
            for ctrl, vals in scores.items():
                m, sd = criteria.mean_std(vals)
                rows.append(dict(kind=kind, shots=shots, noise_p=st, control=ctrl, f1_macro=m, std=sd,
                                 subset_size=int(len(sub.y_test))))
    return rows, by_point
