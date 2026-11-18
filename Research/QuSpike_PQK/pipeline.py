"""單一設定的評估流程，含 validation 上的混合權重選擇（T024／T029）。

為什麼需要在 validation 上調 `expert_scale`：專家輸出（SVM decision_function 或
log 機率）與參考線 logits 的尺度不同，固定為 1.0 等於隨機挑一個混合權重。
若不調，「框架有用」這項判準會被一個沒調過的權重拖累，負面結論就不可信。

公平性保證：**所有設定共用同一個權重網格、同一個 validation 切分、同一個選擇規則**，
所以 MAIN 與各對照組之間的比較不受此步驟偏惠。

008 變更：推論改走 `inference.predict`（無標籤）。舊的洩漏行為只在 `legacy_leaky=True` 時重現。
"""
import time

import numpy as np
from sklearn.metrics import f1_score

from . import experts, features as F, inference, kernels as K, pqk, pqk_v2, router

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
    # v2（US3）：新的、事前登錄的假設；對照組 V2_nocouple／V2_classical 與 BEST_CLASSICAL_US2
    "V2": ("pqk_v2", True, True),
    "V2_nocouple": ("pqk_v2", False, True),
    "V2_classical": ("rff_v2", None, True),
}
V2_KINDS = ("pqk_v2", "rff_v2")


def softmax(x):
    """數值穩定的 softmax（逐列減去最大值）。"""
    return inference.softmax(x)


def _make_featurize(kind, entangle, cfg, cache_dir, fs_hash, seed, rff_maps, n_cls):
    """建立 (群名, z, tag) -> φ。PQK 走快取（φ 與種子無關）；古典替身即時算。"""
    def featurize(name, z, tag):
        """對 z（群內投影後的角度）算 φ；tag 為快取檔後綴（tr／ev）。"""
        nq = cfg["n_qubits"].get(name, 4)
        if kind == "pqk_v2":
            d = cfg["v2"]
            key = pqk_v2.cache_key_v2(fs_hash, name, featurize.proj_key[name],
                                      dict(d, n_qubits=nq, trotter_steps=cfg["trotter_steps"],
                                           evolution_time=cfg["evolution_time"]), entangle)
            phi, hit = pqk_v2.cached_features(
                z, cache_dir, f"{key}_{tag}", nq, trotter_steps=cfg["trotter_steps"],
                evolution_time=cfg["evolution_time"], h=d["h"], j0=d["j0"], kappa_yy=d["kappa_yy"],
                kappa_zz=d["kappa_zz"], coupling_scale=1.0 if entangle else 0.0, device=cfg["device"])
            featurize.cache_info[name] = dict(cache_key=key, cache_hit=bool(hit))
            return phi
        if kind == "rff_v2":
            if name not in rff_maps:
                rff_maps[name] = K.RFFMap(z.shape[1], pqk_v2.n_features(nq), seed=seed,
                                          evolution_time=cfg["evolution_time"])
            return rff_maps[name](z)
        if kind == "pqk":
            key = pqk.cache_key(fs_hash, name, featurize.proj_key[name], nq, cfg["trotter_steps"],
                                cfg["evolution_time"], entangle, cfg["shots"], cfg["noise_p"])
            kw = dict(n_qubits=nq, trotter_steps=cfg["trotter_steps"],
                      evolution_time=cfg["evolution_time"], entangle=entangle,
                      shots=cfg["shots"], noise_p=cfg["noise_p"], device=cfg["device"])
            nk = cfg.get("noise_kind")
            if nk and nk != "terminal_depol":  # US5：四種有資訊量的雜訊（v1 線路不動，見 noise.py）
                from . import noise

                par = cfg["noise_params"]
                key += "_" + noise.noise_cache_suffix(nk, par["strength"], par.get("readout_pb", 0.0))
                phi, hit = pqk.cached_features(z, cache_dir, f"{key}_{tag}",
                                               compute=noise.noisy_compute(nk, par["strength"],
                                                                           par.get("readout_pb", 0.0)), **kw)
            else:
                phi, hit = pqk.cached_features(z, cache_dir, f"{key}_{tag}", **kw)
            featurize.cache_info[name] = dict(cache_key=key, cache_hit=bool(hit))
            return phi
        if kind in ("classical_rbf", "mlp"):
            return np.asarray(z)
        if kind == "rff":
            if name not in rff_maps:
                rff_maps[name] = K.RFFMap(z.shape[1], 3 * nq, seed=seed,
                                          evolution_time=cfg["evolution_time"])
            return rff_maps[name](z)
        raise ValueError(kind)

    featurize.proj_key, featurize.cache_info = {}, {}
    return featurize


def _targets(grouped, groups, n_cls, n_train, n_test):
    """C5 不分群：單一專家吃全部類別。"""
    if grouped:
        return [g for g in groups if g.is_quantum]
    return [F.Group("all", tuple(range(n_cls)), n_train, n_test, True,
                    np.arange(n_train), np.arange(n_test))]


def fit_state(kind, entangle, grouped, fs, groups, cfg, seed, cache_dir, fs_hash,
              supplement="zero", fit_experts=True):
    """訓練參考線與投影，並（可選）擬合各群專家。回傳 (InferenceState, 診斷)。"""
    n_cls = len(fs.labels)
    model, dev = router.fit_reference(fs.x_train, fs.y_train, n_cls, seed=seed,
                                      kind=cfg.get("reference", "linear"))
    targets = _targets(grouped, groups, n_cls, len(fs.y_train), len(fs.y_test))
    rff_maps = {}
    feat = _make_featurize(kind, entangle, cfg, cache_dir, fs_hash, seed, rff_maps, n_cls)
    st = inference.InferenceState(ref_model=model, device=dev, groups=groups, targets=targets,
                                  n_classes=n_cls, featurize=feat, grouped=grouped)
    diag = {"seconds_quantum": 0.0, "projection_info": {}, "kernel_diagnostics": {}, "psd_events": [],
            "_phi_train": {}}
    for g in targets:
        nq = cfg["n_qubits"].get(g.name, 4)
        pr = F.fit_projection(fs, g, nq, method=cfg["projection"], supplement=supplement)
        st.projections[g.name] = pr
        feat.proj_key[g.name] = pr.method if pr.supplement == "zero" else f"{pr.method}+{pr.supplement}"
        tq = time.time()
        diag["_phi_train"][g.name] = feat(g.name, pr.z_train, "tr")
        diag["seconds_quantum"] += time.time() - tq
        diag["projection_info"][g.name] = dict(
            n_components=pr.n_components, n_padded=pr.n_padded, supplement=pr.supplement,
            n_supplement=pr.n_supplement,
            decoupled_qubits=F.decoupled_qubits(pr.z_train, nq))
    if fit_experts:
        fit_experts_into(st, fs, diag, kind, cfg, seed, cache_dir)
    return st, diag


def fit_experts_into(st, fs, diag, kind, cfg, seed, cache_dir=None):
    """擬合各群專家（與 scale 無關，只擬合一次）。"""
    for g in st.targets:
        phi_tr = diag["_phi_train"][g.name]
        y = fs.y_train[g.train_idx]
        ex = experts.fit_expert("mlp" if kind == "mlp" else "kernel", phi_tr, y, st.n_classes,
                                gamma=cfg["gamma"], seed=seed,
                                hidden=3 * cfg["n_qubits"].get(g.name, 4),
                                kernel_mode=cfg.get("kernel_mode", "exact"),
                                nystrom_m=cfg.get("nystrom_m") if isinstance(cfg.get("nystrom_m"), int) else 512,
                                cache_dir=cache_dir)
        st.experts[g.name] = ex
        diag["kernel_diagnostics"][g.name] = ex.diag
        diag["psd_events"] += [dict(e, group=g.name) for e in ex.psd_events]


def _split_featureset(fs):
    """train 內切出 fit／validation，回傳以 validation 當評估集的 FeatureSet。"""
    fit_idx, val_idx = F.split_validation(fs)
    sub = F.FeatureSet(**{**fs.__dict__, "x_test": fs.x_train[val_idx], "y_test": fs.y_train[val_idx],
                          "x_train": fs.x_train[fit_idx], "y_train": fs.y_train[fit_idx],
                          "month_train": fs.month_train[fit_idx] if fs.month_train is not None else None})
    return sub


def tune_hyperparams(kind, entangle, grouped, fs, cfg, seed, cache_dir, supplement, routing, thr_req,
         legacy_leaky=False):
    """在 validation 上選 expert_scale（與軟路由門檻）。回傳 (scale, thr, 搜尋紀錄)。

    專家的擬合與 scale、門檻都無關 → 只擬合一次、對全部 validation 列評分一次，再逐一套用網格。
    平手規則：F1 相同時取較小門檻（偏向硬路由），再取較大 scale（沿用 v1 的 max((f1, s))）。
    """
    sub = _split_featureset(fs)
    sub_groups = F.make_groups(sub)
    st, diag = fit_state(kind, entangle, grouped, sub, sub_groups, cfg, seed,
                         cache_dir / "val", fs.data_hash + "_val", supplement=supplement)
    ref = router.logits(st.ref_model, sub.x_test, st.device)
    blocks = inference.expert_blocks(st, sub.x_test, "ev", None)
    if routing == "soft":
        thrs = router.SOFT_GRID if thr_req in (None, "auto") else (float(thr_req),)
    else:
        thrs = (0.0,)
    scored = []
    for thr in thrs:
        weights, _ = inference._weights(st, ref, routing, thr)
        if legacy_leaky:
            for g in st.targets:
                if g.name != "all":
                    rows, w = weights[g.name]
                    keep = np.isin(sub.y_test[rows], list(g.class_ids))
                    weights[g.name] = (rows[keep], w[keep])
        for s in SCALE_GRID:
            pred = inference.assemble(st, ref, weights, blocks, s).argmax(1)
            f1 = float(f1_score(sub.y_test, pred, average="macro", zero_division=0))
            scored.append((f1, -thr, s))
    f1, nthr, s = max(scored)
    log = [dict(scale=sc, soft_threshold=-nt, val_f1_macro=v) for v, nt, sc in scored]
    return s, -nthr, log


def evaluate_control(control, fs, groups, cfg, seed, cache_dir, tune=True, legacy_leaky=False):
    """跑單一設定。回傳 (預測, 機率, 診斷)。推論不使用任何測試標籤（除非 legacy_leaky）。"""
    t0 = time.time()
    alias = None
    if control == "BEST_CLASSICAL_US2":  # 登錄檔指定的最佳古典設定（只在 validation 選出）
        alias, control = control, cfg["best_classical"]
    kind, entangle, grouped = SPEC[control]
    if kind in V2_KINDS:
        d = cfg["v2"]
        cfg = dict(cfg, n_qubits=d["n_qubits"], trotter_steps=d["trotter_steps"],
                   evolution_time=d["evolution_time"], supplement=d.get("supplement", "pls"))
    routing = cfg.get("routing", "hard")
    thr_req = cfg.get("soft_threshold", "auto" if routing == "soft" else 0.0)
    supplement = cfg.get("supplement", "zero")
    n_cls = len(fs.labels)
    info = {"seconds": {}, "kernel_diagnostics": {}, "quantum": {}, "expert_scale": 0.0,
            "inference_mode": "legacy_leaky" if legacy_leaky else "label_free"}
    if alias:
        info["alias_of"] = control

    if kind == "none":  # C0：只有參考線（也是結構下限）
        model, dev = router.fit_reference(fs.x_train, fs.y_train, n_cls, seed=seed,
                                          kind=cfg.get("reference", "linear"))
        ref_ev = router.logits(model, fs.x_test, dev)
        routed, _ = router.route(ref_ev, groups)
        info["router_accuracy"] = router.router_accuracy(routed, fs.y_test, groups)
        info["_routed"] = routed
        info["seconds"]["total"] = time.time() - t0
        p = softmax(ref_ev)
        return p.argmax(1), p, info

    best_scale, best_thr = 1.0, (0.0 if routing == "hard" else
                                 (0.0 if thr_req in (None, "auto") else float(thr_req)))
    # US4：Nyström 的地標數 m 也只在 validation 選（256／512／1024）
    tune_done = False
    if cfg.get("kernel_mode") in ("nystrom", "auto") and not isinstance(cfg.get("nystrom_m"), int):
        picks = []
        if tune:
            for m in (256, 512, 1024):
                sc, th, lg = tune_hyperparams(kind, entangle, grouped, fs, dict(cfg, nystrom_m=m), seed,
                                              cache_dir, supplement, routing, thr_req, legacy_leaky)
                picks.append((max(x["val_f1_macro"] for x in lg), -m, m, sc, th, lg))
        if picks:
            _, _, m, best_scale, best_thr, log = max(picks)
            info["scale_search"] = log
            info["nystrom_m_search"] = [dict(m=q[2], val_f1_macro=q[0]) for q in picks]
            tune_done = True
        else:
            m = 512
        cfg = dict(cfg, nystrom_m=m)
        info["nystrom_m_selected"] = m
    if tune and not tune_done:
        best_scale, best_thr, log = tune_hyperparams(kind, entangle, grouped, fs, cfg, seed, cache_dir,
                                                     supplement, routing, thr_req, legacy_leaky)
        info["scale_search"] = log
    info["expert_scale"] = best_scale
    info["soft_threshold_selected"] = best_thr

    st, diag = fit_state(kind, entangle, grouped, fs, groups, cfg, seed, cache_dir, fs.data_hash,
                         supplement=supplement, fit_experts=best_scale != 0.0)
    info["seconds"]["quantum_features"] = diag["seconds_quantum"]
    info["projection_info"] = diag["projection_info"]
    info["decoupled_qubits"] = {k: v["decoupled_qubits"] for k, v in diag["projection_info"].items()}
    info["quantum"] = dict(st.featurize.cache_info)
    info["kernel_diagnostics"] = diag["kernel_diagnostics"]
    info["psd_events"] = diag["psd_events"]
    st.scale, st.routing, st.soft_threshold = best_scale, routing, best_thr

    # 路由診斷：硬／軟路由下真實群別專家有機會修正的比例（只用於報告）
    ref_ev = router.logits(st.ref_model, fs.x_test, st.device)
    routed, _ = router.route(ref_ev, groups)
    info["router_accuracy"] = router.router_accuracy(routed, fs.y_test, groups)
    info["_routed"] = routed
    if grouped:
        wh, names = router.route_weights(ref_ev, groups, 0.0)
        info["route_coverage"] = {"hard": router.coverage(wh, names, fs.y_test, groups)}
        if routing == "soft":
            ws, _ = router.route_weights(ref_ev, groups, best_thr)
            info["route_coverage"]["soft"] = router.coverage(ws, names, fs.y_test, groups)

    if best_scale == 0.0:  # validation 選到「不要專家」→ 等同參考線
        info["seconds"]["total"] = time.time() - t0
        p = softmax(ref_ev)
        return p.argmax(1), p, info
    if legacy_leaky:
        pred, proba, _ = inference.predict_legacy_leaky(st, fs.x_test, fs.y_test)
    else:
        pred, proba, _ = inference.predict(st, fs.x_test)
    info["seconds"]["total"] = time.time() - t0
    return pred, proba, info


def feature_map(kind, z_tr, z_ev, group_name, n_q, cfg, cache_dir, entangle, seed, fs_hash, proj_method):
    """相容 007 的介面：回傳 (訓練 φ, 評估 φ, 快取資訊)。新程式碼請用 `_make_featurize`。"""
    feat = _make_featurize(kind, entangle, dict(cfg, n_qubits={**cfg["n_qubits"], group_name: n_q}),
                           cache_dir, fs_hash, seed, {}, 0)
    feat.proj_key[group_name] = proj_method
    tr, ev = feat(group_name, z_tr, "tr"), feat(group_name, z_ev, "ev")
    return tr, ev, dict(feat.cache_info.get(group_name, {}))
