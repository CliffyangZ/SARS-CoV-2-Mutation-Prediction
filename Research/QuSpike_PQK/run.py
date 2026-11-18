"""CLI（T004／T029）。

七組設定共用同一個 FeatureSet、路由、降維方式、分類器與混合權重網格，
只在「特徵圖」處不同（FR-011）。MAIN 與 C3 只差 entangle 一個欄位。

離開碼語意（FR-027）：任一判準不成立仍回傳 0——負面結果是有效結論，不是執行錯誤。
"""
import argparse
from pathlib import Path

import numpy as np

from Research.lineage_baselines.metrics import compute_metrics, per_class_report
from Research.lineage_baselines.results_io import environment_info, result_path, write_result

from . import criteria, features as F, pipeline, stats
from .pipeline import SPEC

DEFAULT_OUT = F.D.ROOT / "Results" / "quspike_pqk"
CONTROLS = ("MAIN", "C0", "C1", "C2", "C3", "C4", "C5")


def make_cfg(a):
    """把命令列參數轉成設定字典（七組設定共用）。"""
    nq = dict(x.split("=") for x in a.n_qubits.split(","))
    return dict(n_qubits={k: int(v) for k, v in nq.items()},
                trotter_steps=a.trotter_steps, evolution_time=a.evolution_time,
                gamma=None if a.gamma == "auto" else float(a.gamma),
                projection=a.projection, shots=a.shots, noise_p=a.noise_p, device=a.device,
                routing=getattr(a, "routing", "hard"),
                soft_threshold=getattr(a, "soft_threshold", "auto"),
                supplement=getattr(a, "supplement", "zero"),
                kernel_mode=getattr(a, "kernel", "exact"),
                nystrom_m=getattr(a, "nystrom_m", None))


V1_KEYS = ("n_qubits", "trotter_steps", "evolution_time", "gamma", "projection", "shots",
           "noise_p", "device")


def v1_quantum_cfg(cfg):
    """只取 v1 量子超參數（FR-015 凍結檢查用）。"""
    return {k: cfg[k] for k in V1_KEYS}


def group_scores(fs, groups, pred):
    """分群 macro F1（FR-024）。量子只應改變難群。"""
    from sklearn.metrics import f1_score

    out = {}
    for g in groups:
        ids = np.asarray(g.class_ids)
        mask = np.isin(fs.y_test, ids)
        if mask.sum():
            out[g.name] = float(f1_score(fs.y_test[mask], pred[mask], labels=ids,
                                         average="macro", zero_division=0))
    return out


def subsample(fs, limit):
    """隨機取子集（僅供除錯用的 --limit）。"""
    rng = np.random.default_rng(0)
    tr = rng.permutation(len(fs.y_train))[:limit]
    te = rng.permutation(len(fs.y_test))[:max(limit // 2, 1)]
    return F.FeatureSet(**{**fs.__dict__, "x_train": fs.x_train[tr], "y_train": fs.y_train[tr],
                           "x_test": fs.x_test[te], "y_test": fs.y_test[te],
                           "month_train": fs.month_train[tr] if fs.month_train is not None else None})


def build_parser():
    """建立主管線的參數解析器。"""
    p = argparse.ArgumentParser(description="QuSpike-PQK：投影量子核分類管線")
    p.add_argument("--data", default=str(F.DEFAULT_DATA))
    p.add_argument("--features", choices=["mutation", "lm"], default=None)
    p.add_argument("--embedding-cache", default=None)
    p.add_argument("--controls", default=",".join(CONTROLS))
    p.add_argument("--n-qubits", default="delta=4,b1=2,all=4")
    p.add_argument("--trotter-steps", type=int, default=4)
    p.add_argument("--evolution-time", type=float, default=0.8)
    p.add_argument("--gamma", default="auto")
    p.add_argument("--projection", choices=["lda", "pca", "raw"], default="lda")
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--noise-p", type=float, default=0.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--all-seeds", type=int, nargs="+", default=None,
                   help="完整種子清單（記錄於結果檔）。種子平行跑時，每個行程只跑 --seeds 的一部分")
    p.add_argument("--routing", choices=["hard", "soft"], default="hard")
    p.add_argument("--soft-threshold", default="auto",
                   help="軟路由門檻；auto = 在 validation 上選（research D5）")
    p.add_argument("--supplement", choices=["zero", "pls"], default="zero",
                   help="投影維度不足時的補充方式；zero = v1 行為")
    p.add_argument("--kernel", choices=["exact", "nystrom", "auto"], default="exact")
    p.add_argument("--nystrom-m", type=int, default=None)
    p.add_argument("--legacy-leaky", action="store_true",
                   help="重現 v1 的標籤洩漏推論（僅供比對舊數字；結果不得作為最終數字）")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cache", default=None)
    p.add_argument("--device", default="default.qubit")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-tune", action="store_true", help="跳過 validation 上的混合權重搜尋")
    return p


def cmd_noise(argv):
    """noise 子命令（US5／007 US2）：掉點曲線。

    不給 --noise-kind：沿用 007 的有限取樣＋終端去極化掃描。
    給 --noise-kind（可重複）：每種型態各掃強度×取樣次數，並輸出 Gram 偏差與最低取樣次數。
    terminal_depol 附 Gram 不變性斷言（FR-019），違反 → 離開碼 3。
    """
    import json

    from . import noise, pqk

    p = build_parser()
    p.add_argument("--shots-list", default="null,8192,1024,256")
    p.add_argument("--noise-list", default="0,0.005,0.02")
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--noise-kind", action="append", choices=list(noise.KINDS), default=None)
    p.add_argument("--strengths", default="0,0.005,0.02")
    p.add_argument("--readout-pb", type=float, default=0.0)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(a)
    fs = F.load_features(source=a.features or "mutation", data=a.data,
                         embedding_cache=a.embedding_cache)
    groups = F.make_groups(fs)
    shots_list = [None if s.strip() in ("null", "none", "") else int(s)
                  for s in a.shots_list.split(",")]
    cache = Path(a.cache) if a.cache else out / "cache"
    controls = tuple(a.controls.split(","))
    if not a.noise_kind:
        noise_list = [float(x) for x in a.noise_list.split(",")]
        print(f"噪聲掃描：shots={shots_list}　noise_p={noise_list}　分層子集={a.subset}", flush=True)
        print("（密度矩陣模擬慢 33–70 倍，故不跑全量；見 research.md D5）", flush=True)
        rows, by_point = noise.scan(fs, groups, cfg, cache, shots_list=shots_list,
                                    noise_list=noise_list, controls=controls, seeds=tuple(a.seeds),
                                    subset=a.subset, tune=not a.no_tune)
        curve = noise.drop_curve(rows)
        for r in curve:
            d = "—" if r["drop"] is None else f"{r['drop']:+.4f}"
            print(f"  shots={str(r['shots']):>5s} p={r['noise_p']:<6g} {r['control']:5s} "
                  f"F1={r['f1_macro']:.4f} 掉點={d}", flush=True)
        ms, note = noise.min_shots_for_same_verdict(by_point)
        print(f"\n最低取樣次數（維持相同判定）：{ms}　{note}", flush=True)
        (out / "noise_scan.json").write_text(json.dumps(
            {"curve": curve, "min_shots_for_same_verdict": ms, "note": note},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    strengths = [float(x) for x in a.strengths.split(",")]
    sub = noise.subset_featureset(fs, a.subset)
    g_delta = [g for g in F.make_groups(sub) if g.name == "delta"][0]
    pr = F.fit_projection(sub, g_delta, cfg["n_qubits"]["delta"], method=cfg["projection"])
    z = pr.z_test[:1500]
    report, code = {}, 0
    for kind in a.noise_kind:
        print(f"\n== 雜訊型態 {kind} ==", flush=True)
        grams = [noise.kind_gram_report(z, kind, st, n_qubits=cfg["n_qubits"]["delta"],
                                        readout_pb=a.readout_pb,
                                        trotter_steps=cfg["trotter_steps"],
                                        evolution_time=cfg["evolution_time"])
                 for st in strengths if st > 0]
        for gr in grams:
            print(f"  強度 {gr['strength']:<7g} Gram 偏差 {gr['gram_deviation']:.3e}", flush=True)
        if kind == "terminal_depol" and any(g["gram_deviation"] >= 1e-6 for g in grams):
            print("[失敗] 終端去極化的 Gram 不變性被破壞（FR-019 回歸檢查）", flush=True)
            code = 3
        rows, by_point = noise.scan_kind(fs, groups, cfg, cache, kind, strengths,
                                         shots_list=shots_list, readout_pb=a.readout_pb,
                                         controls=controls, seeds=tuple(a.seeds), subset=a.subset,
                                         tune=not a.no_tune)
        curve = noise.drop_curve([dict(r, noise_p=r["noise_p"]) for r in rows])
        for r in curve:
            d = "—" if r["drop"] is None else f"{r['drop']:+.4f}"
            print(f"  shots={str(r['shots']):>5s} 強度={r['noise_p']:<6g} {r['control']:5s} "
                  f"F1={r['f1_macro']:.4f} 掉點={d}", flush=True)
        ms, note = noise.min_shots_for_same_verdict(by_point)
        print(f"  最低取樣次數（維持相同判定）：{ms}　{note}", flush=True)
        report[kind] = dict(gram=grams, curve=curve, min_shots_same_verdict=ms, note=note)
    (out / "noise_scan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return code


def cmd_hardware(argv):
    """hardware 子命令（US3）：**預設乾跑，不送機**。"""
    import json

    from . import hardware as H

    p = argparse.ArgumentParser(description="QuSpike-PQK 硬體乾跑（預設不送機）")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--budget-seconds", type=float, default=600.0)
    p.add_argument("--backend", default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--estimate-only", action="store_true", default=True)
    p.add_argument("--submit", action="store_true",
                   help="明確啟用送機。本階段列為 Out of Scope，會消耗裝置配額")
    p.add_argument("--allow-network", action="store_true",
                   help="明確允許連線查詢候選裝置與指令時長。預設完全離線估算")
    a = p.parse_args(argv)
    try:
        est = H.plan_run(n_samples=a.samples, n_qubits=a.n_qubits, shots=a.shots,
                         budget_seconds=a.budget_seconds, submit=a.submit, backend=a.backend,
                         allow_network=a.allow_network)
    except H.CredentialError as e:
        print(f"[失敗] {e}", flush=True)
        print("未寫出任何結果檔（不得產生聲稱來自硬體的結果）。", flush=True)
        return 2
    for k in ("chosen_backend", "n_samples", "n_circuits", "n_jobs", "shots",
              "estimated_device_seconds", "duration_source", "estimate_is_rough",
              "budget_seconds", "exceeds_budget", "max_samples_in_budget", "submitted_jobs",
              "network_used"):
        print(f"  {k:26s} {est[k]}", flush=True)
    if est["backend_listing_failed"]:
        print("  [注意] 未取得候選裝置清單。預設為完全離線估算；要連線查詢須加 --allow-network"
              "（會把憑證送往外部服務）。", flush=True)
    if est["exceeds_budget"]:
        print(f"  [超出預算] 預算內可處理的最大樣本數 = {est['max_samples_in_budget']}", flush=True)
    print(f"  {est['submit_hint']}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardware_estimate.json").write_text(json.dumps(est, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return 0


def _shuffled(fs, seed):
    """評估標籤隨機打亂的 FeatureSet 副本（洩漏檢查用）。"""
    y = fs.y_test.copy()
    np.random.default_rng(int(seed)).shuffle(y)
    return F.FeatureSet(**{**fs.__dict__, "y_test": y})


def cmd_leakcheck(argv):
    """leakcheck 子命令（FR-002、SC-001）：打亂評估標籤後，所有設定的預測必須逐筆不變。

    離開碼：0 全部相同；3 偵測到洩漏。先以同標籤重跑一次當對照——若連它都不相同，
    代表裝置非決定性，檢查無法下結論（印出並以離開碼 4 回報，而不是誤判為洩漏）。
    """
    p = build_parser()
    a = p.parse_args(argv)
    cfg = make_cfg(a)
    fs = F.load_features(source=a.features or "mutation", data=a.data,
                         embedding_cache=a.embedding_cache)
    if a.limit:
        fs = subsample(fs, a.limit)
    cache = Path(a.cache) if a.cache else Path(a.out) / "cache"
    seed = a.seeds[0]
    leaked, inconclusive = [], []
    for control in a.controls.split(","):
        base = pipeline.evaluate_control(control, fs, F.make_groups(fs), cfg, seed, cache,
                                         tune=not a.no_tune)[0]
        again = pipeline.evaluate_control(control, fs, F.make_groups(fs), cfg, seed, cache,
                                          tune=not a.no_tune)[0]
        sh_fs = _shuffled(fs, 12345)
        shuf = pipeline.evaluate_control(control, sh_fs, F.make_groups(sh_fs), cfg, seed, cache,
                                         tune=not a.no_tune)[0]
        deterministic = bool(np.array_equal(base, again))
        same = bool(np.array_equal(base, shuf))
        print(f"  [{control:5s}] identical={str(same).lower():5s} "
              f"(同標籤重跑相同={str(deterministic).lower()}，差異筆數={int((base != shuf).sum())})",
              flush=True)
        if not deterministic:
            inconclusive.append(control)
        elif not same:
            leaked.append(control)
    if leaked:
        print(f"[洩漏] 打亂標籤後預測改變：{leaked}", flush=True)
        return 3
    if inconclusive:
        print(f"[無法判定] 裝置非決定性：{inconclusive}", flush=True)
        return 4
    print("全部設定的預測在打亂標籤後逐筆相同（SC-001）。", flush=True)
    return 0


def cmd_ceiling(argv):
    """ceiling 子命令（FR-006、SC-004）：只依資料，不需任何模型。"""
    import json

    from . import ceiling

    p = build_parser()
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = ceiling.run_report(a.data, out / "pqk_C0__mut__seed0.json" if (out / "pqk_C0__mut__seed0.json").exists() else None)
    (out / "ceiling.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "ceiling.md").write_text(ceiling.to_markdown(rep), encoding="utf-8")
    for g, r in rep["groups"].items():
        print(f"  群 {g:6s} 衝突序列比例(test)={r['conflict_fraction_test']:.3f} "
              f"群內 macro-F1 上限={r['macro_f1_upper_bound']:.3f}", flush=True)
    return 0


def cmd_baselines22(argv):
    """baselines22 子命令（FR-005）：在與 PQK 相同的 22 類切分上重跑古典基線。"""
    from . import baselines22

    p = build_parser()
    p.add_argument("--models", default="lr,rf,mlp")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = F.load_features(source=a.features or "mutation", data=a.data,
                         embedding_cache=a.embedding_cache)
    if a.limit:
        fs = subsample(fs, a.limit)
    baselines22.run_all(fs, a.models.split(","), a.seeds, out)
    print("（CNN 基線為 11 類、單種子，與 22 類 PQK 不可比，不重跑。）", flush=True)
    return 0


def result_record(control, seed, fs, groups, cfg, a, pred, m, info, mode, extra_config=None):
    """單一（設定，種子）的結果檔內容（欄位見 contracts/result-schema.md）。"""
    spec = SPEC["V2" if control == "BEST_CLASSICAL_US2" else control] if control in SPEC else SPEC["C0"]
    config = dict(name=f"pqk_{control}", control=control, features=fs.source,
                  features_selection_reason=fs.selection_reason,
                  n_qubits=cfg["n_qubits"], trotter_steps=cfg["trotter_steps"],
                  evolution_time=cfg["evolution_time"], entangle=spec[1],
                  projection=cfg["projection"], gamma=cfg["gamma"],
                  kernel_kind=spec[0], grouped=spec[2],
                  expert_scale=info.get("expert_scale"),
                  shots=cfg["shots"], noise_p=cfg["noise_p"], backend="simulator",
                  data=Path(a.data).name, n_classes=len(fs.labels),
                  inference_mode=info.get("inference_mode", mode),
                  legacy_leaky=bool(getattr(a, "legacy_leaky", False)), routing=cfg["routing"],
                  soft_threshold=cfg["soft_threshold"],
                  soft_threshold_selected=info.get("soft_threshold_selected"),
                  supplement=cfg["supplement"], kernel_mode=cfg["kernel_mode"],
                  nystrom_m=cfg["nystrom_m"], nystrom_m_selected=info.get("nystrom_m_selected"),
                  seeds=list(getattr(a, "all_seeds", None) or a.seeds),
                  noise_kind=cfg.get("noise_kind"), noise_params=cfg.get("noise_params"),
                  alias_of=info.get("alias_of"))
    config.update(extra_config or {})
    return dict(
        config=config, seed=seed, split_version=fs.split_version, status="complete",
        metrics_test=m,
        per_class_test=per_class_report(fs.y_test, pred, len(fs.labels), fs.labels),
        group_metrics_test=group_scores(fs, groups, pred),
        router_accuracy=info.get("router_accuracy"),
        route_coverage=info.get("route_coverage"),
        decoupled_qubits=info.get("decoupled_qubits", {}),
        projection_info=info.get("projection_info", {}),
        kernel_diagnostics=info.get("kernel_diagnostics", {}),
        psd_events=info.get("psd_events", []),
        ood_mutation_fraction=fs.ood_fraction,
        scale_search=info.get("scale_search"), val_log=info.get("scale_search"),
        nystrom_m_search=info.get("nystrom_m_search"),
        quantum_feature_cache_key=info.get("quantum", {}),
        seconds=info["seconds"], environment=environment_info(),
        test_eval_index=(extra_config or {}).get("test_eval_index"))


def _v2_cfg(a, design):
    """把登錄的 v2 設計併入設定。supplement／routing 取自設計，使對照組共用同一個古典改進。"""
    cfg = make_cfg(a)
    cfg.update(v2=dict(design), supplement=design.get("supplement", "pls"),
               routing=design.get("routing", "hard"),
               soft_threshold=design.get("soft_threshold", "auto"),
               best_classical=design.get("best_classical", "C1"))
    return cfg


def cmd_v2_select(argv):
    """v2-select（FR-014）：**只用 validation** 選設計。ValidationOnlyView 上沒有 x_test／y_test。"""
    import itertools
    import json

    from . import registry

    p = argparse.ArgumentParser(description="v2 設計選擇（僅 validation）")
    p.add_argument("--draft", required=True)
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--data", default=str(F.DEFAULT_DATA))
    p.add_argument("--out", default=str(DEFAULT_OUT.parent / "quspike_pqk_v2"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    draft = json.loads(Path(a.draft).read_text(encoding="utf-8"))
    fs = F.load_features(source="mutation", data=a.data)
    if a.limit:
        fs = subsample(fs, a.limit)
    view = registry.ValidationOnlyView(fs)
    del fs  # 此後只有 view（無測試陣列）
    sub = view.as_featureset()
    cache = Path(a.out) / "cache_select"
    grid = draft["candidate_grid"]
    keys = sorted(grid)
    cands = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    if len(cands) > a.grid_size:
        raise SystemExit(f"候選組合 {len(cands)} > 上限 {a.grid_size}（避免在 validation 上過擬合）")
    base = dict(draft["design"])
    log = []
    ns = argparse.Namespace(**vars(build_parser().parse_args([])))
    for c in cands:
        design = {**base, **c}
        cfg = _v2_cfg(ns, design)
        cfg.update(n_qubits=design["n_qubits"], trotter_steps=design["trotter_steps"],
                   evolution_time=design["evolution_time"])
        sc, th, lg = pipeline.tune_hyperparams("pqk_v2", True, True, sub, cfg, a.seed, cache,
                                               design.get("supplement", "pls"), cfg["routing"],
                                               cfg["soft_threshold"])
        f1 = max(x["val_f1_macro"] for x in lg)
        log.append(dict(candidate=c, val_f1_macro=f1, scale=sc))
        print(f"  候選 {c} → validation macro F1 {f1:.4f}", flush=True)
    best = max(log, key=lambda r: (r["val_f1_macro"], json.dumps(r["candidate"], sort_keys=True)))
    draft["design"] = {**base, **best["candidate"]}
    # 最佳古典設定：同樣只在 validation 上選
    cands_c = {}
    for ctrl in ("C1", "C2", "C4"):
        cfg = _v2_cfg(ns, draft["design"])
        sc, th, lg = pipeline.tune_hyperparams(SPEC[ctrl][0], SPEC[ctrl][1], True, sub,
                                               dict(cfg, n_qubits=make_cfg(ns)["n_qubits"],
                                                    trotter_steps=4, evolution_time=0.8),
                                               a.seed, cache, cfg["supplement"], cfg["routing"],
                                               cfg["soft_threshold"])
        cands_c[ctrl] = max(x["val_f1_macro"] for x in lg)
        cands_c["C0"] = next(x["val_f1_macro"] for x in lg if x["scale"] == 0.0)
    draft["design"]["best_classical"] = max(cands_c, key=cands_c.get)
    draft["selection_log"] = log + [dict(best_classical_validation=cands_c)]
    Path(a.draft).write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"選定設計：{best['candidate']}；最佳古典（validation）：{draft['design']['best_classical']}",
          flush=True)
    print("草稿已更新 selection_log；尚未鎖定、未評估測試集。", flush=True)
    return 0


def cmd_v2_register(argv):
    """v2-register：驗證草稿；--lock 時寫入唯讀登錄檔。"""
    import json

    from . import registry

    p = argparse.ArgumentParser(description="v2 事前登錄")
    p.add_argument("--draft", required=True)
    p.add_argument("--lock", action="store_true")
    p.add_argument("--root", default=str(registry.DEFAULT_ROOT))
    a = p.parse_args(argv)
    draft = json.loads(Path(a.draft).read_text(encoding="utf-8"))
    try:
        registry.validate_draft(draft)
        if not a.lock:
            print(f"草稿格式有效；內容雜湊 {registry.content_hash(draft)}（尚未鎖定）", flush=True)
            return 0
        path = registry.lock(draft, root=a.root)
    except registry.RegistryError as e:
        print(f"[拒絕] {e}", flush=True)
        return 2
    print(f"已鎖定 {draft['version']} → {path}", flush=True)
    return 0


def cmd_v2_eval(argv):
    """v2-eval：只有通過 assert_locked 才會載入測試資料（FR-011）。只評估一次；重複評估會被計數與警告。"""
    import json
    import uuid

    from . import registry

    p = build_parser()
    p.add_argument("--version", required=True)
    p.add_argument("--root", default=str(registry.DEFAULT_ROOT))
    a = p.parse_args(argv)
    if a.out == str(DEFAULT_OUT):
        a.out = str(DEFAULT_OUT.parent / "quspike_pqk_v2")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(a.seeds)

    def loader():
        """延遲載入測試資料——只有通過登錄檢查後才會被呼叫。"""
        fs = F.load_features(source="mutation", data=a.data)
        return subsample(fs, a.limit) if a.limit else fs

    def runner(fs, rec):
        """以登錄的設計跑 V2 與三個對照，套用登錄檔內的判準。"""
        groups = F.make_groups(fs)
        cfg = _v2_cfg(a, rec["design"])
        cache = out / "cache"
        scores = {}
        idx = _current_index[0]
        for seed in seeds:
            for control in ("V2", "V2_classical", "V2_nocouple", "BEST_CLASSICAL_US2"):
                pred, proba, info = pipeline.evaluate_control(control, fs, groups, cfg, seed, cache)
                m = compute_metrics(fs.y_test, pred, proba, len(fs.labels))
                scores.setdefault(control, []).append(m["f1_macro"])
                a.all_seeds = seeds
                res = result_record(control, seed, fs, groups, cfg, a, pred, m, info, "label_free",
                                    extra_config=dict(registry_version=rec["version"],
                                                      test_eval_index=idx, v2_design=rec["design"]))
                write_result(result_path(out, f"pqk_{control}", "mut", seed), res)
                print(f"[{control:18s} seed{seed}] macro F1={m['f1_macro']:.4f}", flush=True)
        fl = stats.estimate_noise_floor(scores)
        print(f"\n雜訊下限 {fl['floor']:.4f}（{fl['floor_source']}）", flush=True)
        verdicts = []
        for c in rec["criteria"]:
            d, v = criteria.verdict(stats.aggregate(scores[c["a"]]), stats.aggregate(scores[c["b"]]),
                                    fl["floor"])
            verdicts.append(dict(claim=c["claim"], delta=d, verdict=v))
            print(f"  {c['claim']:24s} {c['a']} − {c['b']} = {d:+.4f} → {v}", flush=True)
        (out / f"verdicts_{rec['version']}.json").write_text(
            json.dumps(dict(version=rec["version"], test_eval_index=idx, floor=fl, verdicts=verdicts),
                       ensure_ascii=False, indent=2), encoding="utf-8")
        if not all(v["verdict"] == "holds" for v in verdicts):
            print("\n停止規則：三項判準未全部成立 → 負面結論，不得在同一假設下重調 (h, J0, τ, T)。", flush=True)
        return scores

    _current_index = [0]
    try:
        rec = registry.assert_locked(a.version, a.root)
        _current_index[0] = sum(1 for e in registry._evals(a.root) if e["version"] == a.version) + 1
        if _current_index[0] > 1:
            print(f"[警告] 這是 {a.version} 的第 {_current_index[0]} 次測試評估；正式報告只採第 1 次。", flush=True)
        registry.evaluate_on_test(a.version, loader, runner, run_id=str(uuid.uuid4())[:8], root=a.root)
    except registry.RegistryError as e:
        print(f"[拒絕] {e}", flush=True)
        return 2
    return 0


def cmd_approx(argv):
    """approx（FR-016、SC-008）：每個設定同時跑精確與 Nyström，回報 ΔF1 與時間比。"""
    import json

    p = build_parser()
    p.add_argument("--m", default="auto", help="256,512,1024 或 auto（validation 選）")
    p.add_argument("--exact-from", default=None, help="沿用既有精確版結果目錄（省去重跑）")
    a = p.parse_args(argv)
    if a.out == str(DEFAULT_OUT):
        a.out = str(DEFAULT_OUT.parent / "quspike_pqk_approx")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) if a.cache else out / "cache"
    fs = F.load_features(source="mutation", data=a.data)
    if a.limit:
        fs = subsample(fs, a.limit)
    groups = F.make_groups(fs)
    exact_ref = {}
    if a.exact_from:
        from Research.lineage_baselines.results_io import read_results

        for r in read_results(a.exact_from):
            exact_ref[(r["config"]["control"], r["seed"])] = (r["metrics_test"]["f1_macro"],
                                                              r["seconds"].get("total", 0.0))
    res = {}
    for control in a.controls.split(","):
        if SPEC[control][0] == "none":
            continue
        d_f1, t_ex, t_ap, ms = [], [], [], []
        for seed in a.seeds:
            if (control, seed) in exact_ref:
                f_e, t_e = exact_ref[(control, seed)]
            else:
                pe, pr_, ie = pipeline.evaluate_control(control, fs, groups, dict(make_cfg(a), kernel_mode="exact"),
                                                        seed, cache, tune=not a.no_tune)
                f_e, t_e = compute_metrics(fs.y_test, pe, pr_, len(fs.labels))["f1_macro"], ie["seconds"]["total"]
            cfg_n = dict(make_cfg(a), kernel_mode="nystrom",
                         nystrom_m=None if a.m == "auto" else int(a.m.split(",")[0]))
            pn, prn, inn = pipeline.evaluate_control(control, fs, groups, cfg_n, seed, cache, tune=not a.no_tune)
            f_n = compute_metrics(fs.y_test, pn, prn, len(fs.labels))["f1_macro"]
            d_f1.append(f_n - f_e)
            t_ex.append(t_e)
            t_ap.append(inn["seconds"]["total"])
            ms.append(inn.get("nystrom_m_selected"))
            print(f"[{control} seed{seed}] 精確 {f_e:.4f}/{t_e:.0f}s　Nyström {f_n:.4f}/{t_ap[-1]:.0f}s", flush=True)
        ratio = float(np.mean(t_ap) / max(np.mean(t_ex), 1e-9))
        delta = float(np.mean(d_f1))
        usable = bool(abs(delta) <= 0.005 and ratio <= 0.5)
        why = ("通過" if usable else
               "；".join(x for x in (f"|ΔF1|={abs(delta):.4f} > 0.005" if abs(delta) > 0.005 else "",
                                     f"時間比 {ratio:.2f} > 0.5（未縮短 ≥50%）" if ratio > 0.5 else "") if x)
               + " → approx_unusable，採精確版")
        res[control] = dict(delta_f1=delta, time_ratio=ratio, usable=usable, reason=why,
                            nystrom_m=ms, seconds_exact=float(np.mean(t_ex)), seconds_nystrom=float(np.mean(t_ap)))
    (out / "approx_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_report(argv):
    """report：Q1–Q9 彙整與論文待更新清單。"""
    from . import opt_report

    return opt_report.main(argv)


def main(argv=None):
    """命令列入口。子命令：（預設）模擬器管線、noise、hardware、leakcheck、ceiling、baselines22。"""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "noise":
        return cmd_noise(argv[1:])
    if argv and argv[0] == "hardware":
        return cmd_hardware(argv[1:])
    if argv and argv[0] == "leakcheck":
        return cmd_leakcheck(argv[1:])
    if argv and argv[0] == "ceiling":
        return cmd_ceiling(argv[1:])
    if argv and argv[0] == "baselines22":
        return cmd_baselines22(argv[1:])
    table = {"v2-select": cmd_v2_select, "v2-register": cmd_v2_register, "v2-eval": cmd_v2_eval,
             "approx": cmd_approx, "report": cmd_report}
    if argv and argv[0] in table:
        return table[argv[0]](argv[1:])
    a = build_parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) if a.cache else out / "cache"
    cfg = make_cfg(a)

    fs = F.load_features(source=a.features or "mutation", data=a.data,
                         embedding_cache=a.embedding_cache)
    if a.limit:
        fs = subsample(fs, a.limit)
    groups = F.make_groups(fs)
    print(f"特徵 {fs.source}（{fs.x_train.shape[1]} 維）；理由：{fs.selection_reason}", flush=True)
    for g in groups:
        print(f"  群 {g.name:7s} 類別 {len(g.class_ids):2d} train {g.n_train:6d} "
              f"test {g.n_test:5d} 量子={'是' if g.is_quantum else '否'}", flush=True)
    mode = "legacy_leaky" if a.legacy_leaky else "label_free"
    print(f"推論模式：{mode}　路由：{a.routing}　補維：{a.supplement}　種子：{list(a.seeds)}", flush=True)

    scores = {}
    runs = {}
    for seed in a.seeds:
        for control in a.controls.split(","):
            pred, proba, info = pipeline.evaluate_control(control, fs, groups, cfg, seed, cache,
                                                          tune=not a.no_tune,
                                                          legacy_leaky=a.legacy_leaky)
            m = compute_metrics(fs.y_test, pred, proba, len(fs.labels))
            scores.setdefault(control, []).append(m["f1_macro"])
            runs[(control, seed)] = (pred, info.pop("_routed", None))
            res = result_record(control, seed, fs, groups, cfg, a, pred, m, info, mode)
            write_result(result_path(out, f"pqk_{control}", "mut", seed), res)
            print(f"[{control:5s} seed{seed}] macro F1={m['f1_macro']:.4f} acc={m['accuracy']:.4f} "
                  f"scale={info.get('expert_scale')} 路由={info.get('router_accuracy', 0):.4f} "
                  f"{info['seconds'].get('total', 0):.1f}s", flush=True)

        # SC-004（008 改述）：**被路由到** others 的列，預測必須與 C0 逐筆相同
        if ("C0", seed) in runs and ("MAIN", seed) in runs:
            c0_pred, routed = runs[("C0", seed)]
            main_pred = runs[("MAIN", seed)][0]
            if routed is not None:
                mask = routed == "others"
                diff = int((main_pred[mask] != c0_pred[mask]).sum())
                print(f"  [檢查] 路由到 others 的列與 C0 的差異筆數 = {diff}（要求 0）", flush=True)

    floor = stats.estimate_noise_floor(scores)
    print(f"\n=== 雜訊下限：論文 δ={floor['delta']:.3f}　本資料 {len(a.seeds)} 種子合併 σ={floor['sigma_pooled']:.4f}"
          f"　採用 {floor['floor']:.4f}（{floor['floor_source']}）===", flush=True)
    print("=== 判準（事前訂定）===", flush=True)
    recs = criteria.evaluate(scores, floor=floor["floor"])
    for r in recs:
        print(f"  {r['claim']:22s} {r['a']} − {r['b']:4s} = {r['delta']:+.4f}  → {r['verdict']}",
              flush=True)
    if criteria.stop_condition_met(recs):
        print("\n停止條件已觸發：量子特徵與量子性皆無增量。依提案 §4，不得加大 n 或調參求轉正。",
              flush=True)
    return 0  # FR-027


if __name__ == "__main__":
    import sys

    sys.exit(main())
