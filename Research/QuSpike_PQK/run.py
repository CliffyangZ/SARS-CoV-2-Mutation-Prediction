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

from . import criteria, features as F, pipeline
from .pipeline import SPEC

DEFAULT_OUT = F.D.ROOT / "Results" / "quspike_pqk"
CONTROLS = ("MAIN", "C0", "C1", "C2", "C3", "C4", "C5")


def make_cfg(a):
    """把命令列參數轉成設定字典（七組設定共用）。"""
    nq = dict(x.split("=") for x in a.n_qubits.split(","))
    return dict(n_qubits={k: int(v) for k, v in nq.items()},
                trotter_steps=a.trotter_steps, evolution_time=a.evolution_time,
                gamma=None if a.gamma == "auto" else float(a.gamma),
                projection=a.projection, shots=a.shots, noise_p=a.noise_p, device=a.device)


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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--cache", default=None)
    p.add_argument("--device", default="default.qubit")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-tune", action="store_true", help="跳過 validation 上的混合權重搜尋")
    return p


def cmd_noise(argv):
    """noise 子命令（US2）：有限取樣與去極化的掉點曲線。"""
    import json

    from . import noise

    p = build_parser()
    p.add_argument("--shots-list", default="null,8192,1024,256")
    p.add_argument("--noise-list", default="0,0.005,0.02")
    p.add_argument("--subset", type=int, default=2000)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(a)
    fs = F.load_features(source=a.features or "mutation", data=a.data,
                         embedding_cache=a.embedding_cache)
    groups = F.make_groups(fs)
    shots_list = [None if s.strip() in ("null", "none", "") else int(s)
                  for s in a.shots_list.split(",")]
    noise_list = [float(x) for x in a.noise_list.split(",")]
    print(f"噪聲掃描：shots={shots_list}　noise_p={noise_list}　分層子集={a.subset}", flush=True)
    print("（密度矩陣模擬慢 33–70 倍，故不跑全量；見 research.md D5）", flush=True)
    rows, by_point = noise.scan(fs, groups, cfg, Path(a.cache) if a.cache else out / "cache",
                                shots_list=shots_list, noise_list=noise_list,
                                controls=tuple(a.controls.split(",")), seeds=tuple(a.seeds),
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


def main(argv=None):
    """命令列入口。子命令：（預設）模擬器管線、noise、hardware。"""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "noise":
        return cmd_noise(argv[1:])
    if argv and argv[0] == "hardware":
        return cmd_hardware(argv[1:])
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

    scores, c0_pred = {}, {}
    for seed in a.seeds:
        for control in a.controls.split(","):
            pred, proba, info = pipeline.evaluate_control(control, fs, groups, cfg, seed, cache,
                                                          tune=not a.no_tune)
            m = compute_metrics(fs.y_test, pred, proba, len(fs.labels))
            scores.setdefault(control, []).append(m["f1_macro"])
            if control == "C0":
                c0_pred[seed] = pred
            res = dict(
                config=dict(name=f"pqk_{control}", control=control, features=fs.source,
                            features_selection_reason=fs.selection_reason,
                            n_qubits=cfg["n_qubits"], trotter_steps=cfg["trotter_steps"],
                            evolution_time=cfg["evolution_time"], entangle=SPEC[control][1],
                            projection=cfg["projection"], gamma=cfg["gamma"],
                            kernel_kind=SPEC[control][0], grouped=SPEC[control][2],
                            expert_scale=info.get("expert_scale"),
                            shots=cfg["shots"], noise_p=cfg["noise_p"], backend="simulator",
                            data=Path(a.data).name, n_classes=len(fs.labels)),
                seed=seed, split_version=fs.split_version, status="complete",
                metrics_test=m,
                per_class_test=per_class_report(fs.y_test, pred, len(fs.labels), fs.labels),
                group_metrics_test=group_scores(fs, groups, pred),
                router_accuracy=info.get("router_accuracy"),
                projection_info=info.get("projection_info", {}),
                kernel_diagnostics=info.get("kernel_diagnostics", {}),
                scale_search=info.get("scale_search"),
                quantum_feature_cache_key=info.get("quantum", {}),
                seconds=info["seconds"], environment=environment_info())
            write_result(result_path(out, f"pqk_{control}", "mut", seed), res)
            print(f"[{control:5s} seed{seed}] macro F1={m['f1_macro']:.4f} acc={m['accuracy']:.4f} "
                  f"scale={info.get('expert_scale')} 路由={info.get('router_accuracy', 0):.4f} "
                  f"{info['seconds'].get('total', 0):.1f}s", flush=True)

        # SC-004：易分群的預測必須與 C0 逐筆相同
        if "C0" in a.controls and "MAIN" in a.controls and seed in c0_pred:
            others = [g for g in groups if not g.is_quantum]
            if others:
                mask = np.isin(fs.y_test, np.concatenate([np.asarray(g.class_ids) for g in others]))
                main_pred = pipeline.evaluate_control("MAIN", fs, groups, cfg, seed, cache,
                                                      tune=not a.no_tune)[0]
                diff = int((main_pred[mask] != c0_pred[seed][mask]).sum())
                print(f"  [檢查] 易分群預測與 C0 的差異筆數 = {diff}（SC-004 要求 0）", flush=True)

    print("\n=== 判準（事前訂定，雜訊下限 0.02）===", flush=True)
    recs = criteria.evaluate(scores)
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
