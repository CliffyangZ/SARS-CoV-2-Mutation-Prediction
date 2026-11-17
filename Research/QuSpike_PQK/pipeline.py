"""單一設定的評估流程，含 validation 上的混合權重選擇（T024／T029）。

為什麼需要在 validation 上調 `expert_scale`：專家輸出（SVM decision_function 或
log 機率）與參考線 logits 的尺度不同，固定為 1.0 等於隨機挑一個混合權重。
若不調，「框架有用」這項判準會被一個沒調過的權重拖累，負面結論就不可信。

公平性保證：**所有設定共用同一個權重網格、同一個 validation 切分、同一個選擇規則**，
所以 MAIN 與各對照組之間的比較不受此步驟偏惠。
"""
import time

import numpy as np
from sklearn.metrics import f1_score

from . import experts, features as F, kernels as K, pqk, router

SCALE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

# 每個設定的 (特徵圖, 是否糾纏, 是否分群)
SPEC = {
    "MAIN": ("pqk", True, True),
    "C0": ("none", None, False),
    "C1": ("classical_rbf", None, True),
    "C2": ("rff", None, True),
    "C3": ("pqk", False, True),
    "C4": ("mlp", None, True),
    "C5": ("pqk", True, False),
}


def softmax(x):
    """數值穩定的 softmax（逐列減去最大值）。"""
    z = np.asarray(x, np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def feature_map(kind, z_tr, z_ev, group_name, n_q, cfg, cache_dir, entangle, seed, fs_hash, proj_method):
    """取特徵圖輸出。PQK 走快取（φ 與種子無關）；古典替身即時算。"""
    if kind == "pqk":
        key = pqk.cache_key(fs_hash, group_name, proj_method, n_q, cfg["trotter_steps"],
                            cfg["evolution_time"], entangle, cfg["shots"], cfg["noise_p"])
        kw = dict(n_qubits=n_q, trotter_steps=cfg["trotter_steps"],
                  evolution_time=cfg["evolution_time"], entangle=entangle,
                  shots=cfg["shots"], noise_p=cfg["noise_p"], device=cfg["device"])
        tr, hit_a = pqk.cached_features(z_tr, cache_dir, f"{key}_tr", **kw)
        ev, hit_b = pqk.cached_features(z_ev, cache_dir, f"{key}_ev", **kw)
        return tr, ev, dict(cache_key=key, cache_hit=bool(hit_a and hit_b))
    if kind in ("classical_rbf", "mlp"):
        return np.asarray(z_tr), np.asarray(z_ev), {}
    if kind == "rff":
        tr, ev = K.rff_surrogate(z_tr, z_ev, n_features=3 * n_q, seed=seed,
                                 evolution_time=cfg["evolution_time"],
                                 trotter_steps=cfg["trotter_steps"])
        return tr, ev, {}
    raise ValueError(kind)


def _expert_blocks(kind, targets, phis, y_train_by_group, routed, eval_index, n_cls, cfg, seed, scale):
    """為每個目標群算出要加到參考線上的 logits 區塊。"""
    blocks, diags = [], {}
    for g in targets:
        phi_tr, phi_ev = phis[g.name]
        rows_global = np.where(routed == g.name)[0] if len(targets) > 1 or g.name != "all" else np.arange(len(routed))
        if g.name == "all":
            rows_global = np.arange(len(routed))
        pos = {int(v): i for i, v in enumerate(eval_index[g.name])}
        rows = np.array([r for r in rows_global if int(r) in pos], np.int64)
        if not len(rows):
            continue
        take = np.array([pos[int(r)] for r in rows], np.int64)
        block, diag = experts.fit_group_expert(
            "mlp" if kind == "mlp" else "kernel", phi_tr, phi_ev[take],
            y_train_by_group[g.name], g.class_ids, n_cls, len(rows),
            gamma=cfg["gamma"], seed=seed, scale=scale, hidden=3 * cfg["n_qubits"].get(g.name, 4))
        diags[g.name] = diag
        blocks.append((rows, block))
    return blocks, diags


def evaluate_control(control, fs, groups, cfg, seed, cache_dir, tune=True):
    """跑單一設定。回傳 (預測, 機率, 診斷)。"""
    t0 = time.time()
    kind, entangle, grouped = SPEC[control]
    n_cls = len(fs.labels)
    info = {"seconds": {}, "kernel_diagnostics": {}, "quantum": {}, "expert_scale": 0.0}

    # --- 參考線（= C0，也是結構下限）---
    ref_model, dev = router.fit_reference(fs.x_train, fs.y_train, n_cls, seed=seed)
    ref_ev = router.logits(ref_model, fs.x_test, dev)
    routed, _ = router.route(ref_ev, groups)
    info["router_accuracy"] = router.router_accuracy(routed, fs.y_test, groups)

    if kind == "none":
        info["seconds"]["total"] = time.time() - t0
        p = softmax(ref_ev)
        return p.argmax(1), p, info

    # --- 目標群 ---
    if grouped:
        targets = [g for g in groups if g.is_quantum]
    else:  # C5：不分群，單一專家吃全部類別
        targets = [F.Group("all", tuple(range(n_cls)), len(fs.y_train), len(fs.y_test), True,
                           np.arange(len(fs.y_train)), np.arange(len(fs.y_test)))]

    # --- 在 validation 上選混合權重（所有設定共用同一網格與規則）---
    best_scale = 1.0
    if tune:
        fit_idx, val_idx = F.split_validation(fs)
        sub = F.FeatureSet(**{**fs.__dict__, "x_test": fs.x_train[val_idx], "y_test": fs.y_train[val_idx],
                              "x_train": fs.x_train[fit_idx], "y_train": fs.y_train[fit_idx],
                              "month_train": fs.month_train[fit_idx] if fs.month_train is not None else None})
        sub_groups = F.make_groups(sub)
        sub_targets = ([g for g in sub_groups if g.is_quantum] if grouped else
                       [F.Group("all", tuple(range(n_cls)), len(sub.y_train), len(sub.y_test), True,
                                np.arange(len(sub.y_train)), np.arange(len(sub.y_test)))])
        vm, vdev = router.fit_reference(sub.x_train, sub.y_train, n_cls, seed=seed)
        v_ref = router.logits(vm, sub.x_test, vdev)
        v_routed, _ = router.route(v_ref, sub_groups)
        v_phis, v_y, v_idx = {}, {}, {}
        for g in sub_targets:
            pr = F.fit_projection(sub, g, cfg["n_qubits"].get(g.name, 4), method=cfg["projection"])
            v_phis[g.name] = feature_map(kind, pr.z_train, pr.z_test, g.name,
                                         cfg["n_qubits"].get(g.name, 4), cfg,
                                         cache_dir / "val", entangle, seed, fs.data_hash + "_val",
                                         pr.method)[:2]
            v_y[g.name] = sub.y_train[g.train_idx]
            v_idx[g.name] = g.test_idx
        # 專家的擬合與 scale 無關，只有輸出縮放有關 → 只擬合一次，再逐一套用網格
        raw_blocks, _ = _expert_blocks(kind, sub_targets, v_phis, v_y, v_routed, v_idx,
                                       n_cls, cfg, seed, 1.0)
        scored = []
        for s in SCALE_GRID:
            pred = experts.combine(v_ref, [(r, b * s) for r, b in raw_blocks]).argmax(1)
            scored.append((float(f1_score(sub.y_test, pred, average="macro", zero_division=0)), s))
        best = max(scored)
        best_scale = best[1]
        info["scale_search"] = [{"scale": s, "val_f1_macro": v} for v, s in scored]
    info["expert_scale"] = best_scale

    # --- 以選定的權重在 test 上評估 ---
    phis, y_by_group, idx_by_group = {}, {}, {}
    for g in targets:
        pr = F.fit_projection(fs, g, cfg["n_qubits"].get(g.name, 4), method=cfg["projection"])
        tq = time.time()
        tr, ev, qinfo = feature_map(kind, pr.z_train, pr.z_test, g.name,
                                    cfg["n_qubits"].get(g.name, 4), cfg, cache_dir,
                                    entangle, seed, fs.data_hash, pr.method)
        info["seconds"]["quantum_features"] = info["seconds"].get("quantum_features", 0.0) + (time.time() - tq)
        phis[g.name] = (tr, ev)
        y_by_group[g.name] = fs.y_train[g.train_idx]
        idx_by_group[g.name] = g.test_idx
        if qinfo:
            info["quantum"][g.name] = qinfo
        info.setdefault("projection_info", {})[g.name] = dict(n_components=pr.n_components,
                                                              n_padded=pr.n_padded)
    if best_scale == 0.0:  # validation 選到「不要專家」→ 等同參考線
        info["seconds"]["total"] = time.time() - t0
        p = softmax(ref_ev)
        return p.argmax(1), p, info

    blocks, diags = _expert_blocks(kind, targets, phis, y_by_group, routed, idx_by_group,
                                   n_cls, cfg, seed, best_scale)
    info["kernel_diagnostics"] = diags
    info["seconds"]["total"] = time.time() - t0
    p = softmax(experts.combine(ref_ev, blocks))
    return p.argmax(1), p, info
