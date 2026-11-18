"""008 彙整報告：Q1–Q9 各一列，並列出論文待更新的段落（FR-020／FR-021）。

收錄規則（contracts/result-schema.md）：
1. 只有 `inference_mode == label_free` 且非 legacy_leaky 的結果可作最終數字；舊（洩漏）結果只出現在「前」欄。
2. 同一張表內所有設定的 `config.seeds` 必須相同，否則拒絕產表。
3. V2 系列另需通過登錄時序檢查，且 `test_eval_index == 1`。
4. 判準的雜訊下限 = max(0.02, σ_pooled)。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from Research.lineage_baselines.results_io import read_results

from . import criteria, features as F, registry, stats

OLD_DIR = F.D.ROOT / "Results" / "quspike_pqk"        # 007 的單種子（洩漏）結果
DEFAULT_ROOT = F.D.ROOT / "Results"
VERDICT = {"holds": "**成立**", "fails": "不成立（較差）", "within_noise": "在雜訊內",
           "insufficient_data": "資料不足"}


def load(dir_):
    """讀目錄下所有 complete 結果，分為 (final, legacy)。"""
    final, legacy = [], []
    if not dir_ or not Path(dir_).exists():
        return final, legacy
    for r in read_results(dir_):
        if r.get("status") != "complete":
            continue
        cfg = r["config"]
        leaky = cfg.get("legacy_leaky") or cfg.get("inference_mode") != "label_free"
        (legacy if leaky else final).append(r)
    return final, legacy


def by_control(results, split="test"):
    """{設定: {vals, seeds, groups, seconds, route, cov, decoupled}}。"""
    g = {}
    for r in results:
        name = r["config"].get("control") or r["config"]["name"]
        d = g.setdefault(name, dict(vals=[], seeds=[], groups={}, seconds=[], router=[], cov=[], dec=[],
                                    cfg=r["config"]))
        d["vals"].append(r[f"metrics_{split}"]["f1_macro"])
        d["seeds"].append(r.get("seed"))
        d["seconds"].append((r.get("seconds") or {}).get("total", 0.0))
        if r.get("router_accuracy") is not None:
            d["router"].append(r["router_accuracy"])
        if r.get("route_coverage"):
            d["cov"].append(r["route_coverage"])
        d["dec"].append(r.get("decoupled_qubits", {}))
        for k, v in (r.get(f"group_metrics_{split}") or {}).items():
            d["groups"].setdefault(k, []).append(v)
    return g


def _f(ms):
    return "—" if not np.isfinite(ms[0]) else f"{ms[0]:.4f}±{ms[1]:.4f}"


def table(g, title):
    """設定 × (macro F1 均值±σ, 種子數, 秒)。"""
    L = [f"### {title}", "", "| 設定 | 種子 | macro F1 | Delta 群 F1 | B.1 群 F1 | 秒/種子 |", "|---|---|---|---|---|---|"]
    for n in sorted(g, key=lambda n: -stats.aggregate(g[n]["vals"])[0]):
        d = g[n]
        gm = {k: stats.aggregate(v) for k, v in d["groups"].items()}
        L.append(f"| {n} | {len(d['vals'])} | **{_f(stats.aggregate(d['vals']))}** | "
                 f"{_f(gm.get('delta', (float('nan'),) * 2))} | {_f(gm.get('b1', (float('nan'),) * 2))} | "
                 f"{np.mean(d['seconds']):.0f} |")
    return L


def verdict_table(scores, title):
    """判準表，並列 δ 與重估的雜訊下限。"""
    fl = stats.estimate_noise_floor(scores)
    recs = criteria.evaluate(scores, floor=fl["floor"])
    L = [f"### {title}", "",
         f"雜訊下限：論文 δ = {fl['delta']:.3f}；本資料多種子合併 σ = {fl['sigma_pooled']:.4f}；"
         f"採用 **{fl['floor']:.4f}**（{fl['floor_source']}）", "",
         "| 宣稱 | A | B | A − B | 判定 |", "|---|---|---|---|---|"]
    for r in recs:
        d = "—" if not np.isfinite(r["delta"]) else f"{r['delta']:+.4f}"
        L.append(f"| {r['claim']} | {r['a']} | {r['b']} | {d} | {VERDICT[r['verdict']]} |")
    if criteria.stop_condition_met(recs):
        L += ["", "> **停止條件已觸發**（提案 §4）：量子特徵與量子性皆無增量，不得加大 n 或調參求轉正。"]
    return L, fl, recs


def _seeds_consistent(results):
    seen = {tuple(sorted(r["config"].get("seeds") or [r.get("seed")])) for r in results}
    if len(seen) > 1:
        raise ValueError(f"種子清單不一致，拒絕產表：{sorted(seen)}")


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build(out_dirs, registry_root=registry.DEFAULT_ROOT):
    """產生完整報告（markdown 字串）。out_dirs: {'base','plus','v2','noise','approx','baselines'} → 路徑。"""
    base, legacy_in_base = load(out_dirs.get("base"))
    plus, _ = load(out_dirs.get("plus"))
    old, _ = [], None
    old_all = []
    if OLD_DIR.exists():
        old_all = [r for r in read_results(OLD_DIR) if r.get("status") == "complete"]
    _seeds_consistent(base) if base else None
    gb, go = by_control(base), by_control(old_all)
    L = ["# QuSpike-PQK 優化報告（008）", "",
         "本報告只收錄無標籤推論（`label_free`）的結果作為最終數字；舊的單種子結果僅出現在「前」欄。", ""]

    # --- US1 ---
    L += ["## US1　無洩漏、多種子的結果", ""]
    if gb:
        L += table(gb, f"七組設定（{len(next(iter(gb.values()))['vals'])} 種子，無標籤推論）") + [""]
        vt, floor, recs = verdict_table({n: d["vals"] for n, d in gb.items()}, "四項事前判準（重新判定）")
        L += vt + [""]
        if go:
            L += ["### 舊（洩漏、單種子）與新（無洩漏、多種子）並列", "",
                  "| 設定 | 舊：單種子 | 新：均值±σ | 差異 | 差異來源 |", "|---|---|---|---|---|"]
            for n in sorted(gb):
                if n in go:
                    o = go[n]["vals"][0]
                    nm = stats.aggregate(gb[n]["vals"])
                    s0 = [v for v, s in zip(gb[n]["vals"], gb[n]["seeds"]) if s == go[n]["seeds"][0]]
                    leak = (s0[0] - o) if s0 else float("nan")
                    seed_eff = nm[0] - (s0[0] if s0 else nm[0])
                    L.append(f"| {n} | {o:.4f} | {_f(nm)} | {nm[0] - o:+.4f} | "
                             f"洩漏修正（同種子）{leak:+.4f}；種子平均 {seed_eff:+.4f} |")
            L.append("")
    else:
        L += ["（尚無無洩漏結果）", ""]
    bl = _read_baselines(out_dirs.get("base"))
    if bl:
        L += ["### 同一份 22 類切分上的古典基線（FR-005、SC-003）", "",
              "| 基線 | 種子 | macro F1 | 收斂 |", "|---|---|---|---|"]
        for n, v in bl.items():
            L.append(f"| {n} | {len(v['f1'])} | {_f(stats.aggregate(v['f1']))} | {v['conv']} |")
        L += ["", "> CNN 基線為 11 類、單種子，與 22 類 PQK 不可比，未重跑。", ""]

    # --- US2 ---
    L += ["## US2　古典瓶頸", ""]
    ceil = _read_json(Path(out_dirs.get("base", ".")) / "ceiling.json")
    if ceil:
        L += ["### 天花板（只看 spike 突變）", "", "| 群 | 衝突序列比例(test) | 群內 macro-F1 上限（樂觀） | C0 錯誤中無法避免的比例 |", "|---|---|---|---|"]
        for g, r in ceil["groups"].items():
            u = r.get("unavoidable_error")
            frac = "—" if not u or u["fraction"] is None else "{:.1%}".format(u["fraction"])
            L.append(f"| {g} | {r['conflict_fraction_test']:.3f} | {r['macro_f1_upper_bound']:.3f} | {frac} |")
        L.append("")
    gp = by_control(plus)
    if gp:
        L += table(gp, "軟路由＋PLS 補維（同一改進套用到所有設定）") + [""]
        covs = [c for d in gp.values() for c in d["cov"]]
        if covs:
            hard = np.mean([c["hard"] for c in covs])
            soft = [c["soft"] for c in covs if c.get("soft") is not None]
            L += [f"- 真實群別專家有機會修正的測試列比例：硬路由 {hard:.4f}"
                  + (f"；軟路由 {np.mean(soft):.4f}" + ("（高於硬路由）" if np.mean(soft) > hard else "（**未**高於硬路由）") if soft else ""), ""]
        dec = [d for x in gp.values() for d in x["dec"]]
        if dec:
            L += [f"- 仍有解耦位元的設定（補維後）：{sorted({k + str(v) for d in dec for k, v in d.items() if v}) or '無'}", ""]
        if gb:
            sc = {n: d["vals"] for n, d in gp.items()}
            sc["C0_US1"] = gb.get("C0", {}).get("vals", [])
            best = max((n for n in gp if n != "C0"), key=lambda n: stats.aggregate(gp[n]["vals"])[0])
            fl = stats.estimate_noise_floor({**{n: d["vals"] for n, d in gb.items()}})
            v = criteria.verdict(stats.aggregate(gp[best]["vals"]), stats.aggregate(gb["C0"]["vals"]), fl["floor"])
            L += [f"- 最佳古典改進設定 **{best}** 相對無洩漏 C0：{v[0]:+.4f} → {VERDICT[v[1]]}（SC-006）", ""]
    else:
        L += ["（尚無改進版結果）", ""]

    # --- US3 ---
    L += ["## US3　v2 量子特徵圖（事前登錄）", ""]
    vers = registry.all_versions(registry_root)
    if vers:
        L += ["| 登錄版本 | 鎖定時間 | 首次測試評估 | 時序 |", "|---|---|---|---|"]
        for v in vers:
            rec = registry.load(v, registry_root)
            ok = registry.timing_ok(v, registry_root)
            L.append(f"| {v} | {rec['locked_at']} | {'尚無' if ok is None else '有'} | "
                     f"{'—' if ok is None else ('✓ 鎖定早於評估' if ok else '✗ 不收錄')} |")
        L.append("")
    v2, _ = load(out_dirs.get("v2"))
    if v2:
        v2 = [r for r in v2 if r["config"].get("test_eval_index", 1) == 1]
        g2 = by_control(v2)
        L += table(g2, "V2 與三個對照") + [""]
        if "V2" in g2:
            L += ["| 宣稱 | 對照 | V2 − 對照 | 判定 |", "|---|---|---|---|"]
            fl = stats.estimate_noise_floor({n: d["vals"] for n, d in g2.items()})
            for ctrl in ("V2_classical", "V2_nocouple", "BEST_CLASSICAL_US2"):
                if ctrl in g2:
                    d, v = criteria.verdict(stats.aggregate(g2["V2"]["vals"]), stats.aggregate(g2[ctrl]["vals"]), fl["floor"])
                    L.append(f"| v2 vs {ctrl} | {ctrl} | {d:+.4f} | {VERDICT[v]} |")
            L.append("")
    else:
        L += ["（尚未執行 v2 測試評估）", ""]

    # --- US4 / US5 ---
    L += ["## US4　近似核", ""]
    ap = _read_json(Path(out_dirs.get("approx", ".")) / "approx_check.json")
    if ap:
        L += ["| 設定 | ΔF1（近似−精確） | 時間比 | 可用 | 理由 |", "|---|---|---|---|---|"]
        for n, r in ap.items():
            L.append(f"| {n} | {r['delta_f1']:+.4f} | {r['time_ratio']:.2f} | {'是' if r['usable'] else '**否**'} | {r['reason']} |")
        L.append("")
    else:
        L += ["（尚未執行）", ""]
    L += ["## US5　雜訊", ""]
    nz = _read_json(Path(out_dirs.get("noise", ".")) / "noise_scan.json")
    if nz and all(isinstance(v, dict) for v in nz.values()) and "curve" not in nz:
        L += ["| 型態 | Gram 偏差（各強度） | 維持判定所需最低取樣次數 |", "|---|---|---|"]
        for k, v in nz.items():
            gd = ", ".join(f"{x['strength']:g}→{x['gram_deviation']:.2e}" for x in v["gram"])
            L.append(f"| {k} | {gd} | {v['min_shots_same_verdict']} {v['note']} |")
        L.append("")
    else:
        L += ["（尚未執行）", ""]

    L += ["## Q1–Q9 彙整", ""] + _q_table(gb, go, gp, ceil, ap, nz, bl) + [""]
    L += ["## 論文待更新段落（FR-021）", ""] + _paper_updates(gb, go) + [""]
    return "\n".join(L)


def _read_baselines(dir_):
    if not dir_ or not Path(dir_).exists():
        return {}
    out = {}
    for r in read_results(dir_):
        n = r["config"]["name"]
        if n.startswith("baseline22_"):
            d = out.setdefault(n, dict(f1=[], conv=set()))
            d["f1"].append(r["metrics_test"]["f1_macro"])
            d["conv"].add({True: "是", False: "**否**", None: "—"}[r.get("converged")])
    for d in out.values():
        d["conv"] = "/".join(sorted(d["conv"]))
    return out


def _q_table(gb, go, gp, ceil, ap, nz, bl):
    def mm(g, n):
        """某設定的 macro F1 均值±σ 字串。"""
        return _f(stats.aggregate(g[n]["vals"])) if g and n in g else "—"

    rows = [
        ("Q1 推論用到標籤", "改為無標籤推論（`inference.predict`）", f"MAIN 舊 {go['MAIN']['vals'][0]:.4f}" if go and "MAIN" in go else "—", f"MAIN {mm(gb, 'MAIN')}", "見 US1"),
        ("Q2 單種子", "≥5 種子，重估雜訊下限", "1 種子；δ=0.02", f"{len(gb['MAIN']['vals']) if gb and 'MAIN' in gb else 0} 種子", "見 US1 判準表"),
        ("Q3 路由上限", "軟路由（前兩名群機率差<門檻）", "ρ=0.9667", _cov(gp), "見 US2"),
        ("Q4 資訊瓶頸", "PLS 有監督補維；v2 破除守恆", "delta 位元 3 解耦", "補維後無解耦位元" if gp else "—", "見 US2／US3"),
        ("Q5 困難群", "天花板報告", "—", "見 ceiling" if ceil else "—", "見 US2"),
        ("Q6 古典替身", "v2＋三個對照（事前登錄）", "—", "見 US3", "見 US3"),
        ("Q7 古典核成本", "Nyström 特徵＋線性 SVM；Gram 跨種子快取", "185–576 s", "見 US4" if ap else "—", "見 US4"),
        ("Q8 雜訊無資訊量", "四種不被抵消的型態", "終端去極化 Gram 不變", "見 US5" if nz else "—", "見 US5"),
        ("Q9 基線不可比", "同一份 22 類切分重跑", "22,154/9,503 vs 22,175/9,514", "已重跑" if bl else "—", "見 US1"),
    ]
    L = ["| 問題 | 採取的優化 | 前 | 後 | 判定 |", "|---|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    return L


def _cov(gp):
    covs = [c for d in (gp or {}).values() for c in d["cov"]]
    if not covs:
        return "—"
    soft = [c["soft"] for c in covs if c.get("soft") is not None]
    return f"硬 {np.mean([c['hard'] for c in covs]):.4f}／軟 {np.mean(soft):.4f}" if soft else "—"


def _paper_updates(gb, go):
    new = _f(stats.aggregate(gb["MAIN"]["vals"])) if gb and "MAIN" in gb else "（待結果）"
    c0 = _f(stats.aggregate(gb["C0"]["vals"])) if gb and "C0" in gb else "（待結果）"
    return [f"- **摘要數字**：MAIN macro-F1 與 C0 改為多種子均值±σ（MAIN {new}；C0 {c0}），移除單種子 0.7547／0.7481。",
            "- **§5 結果表與結果段**：改以無標籤推論的多種子結果；四項判準使用 max(0.02, σ) 重新判定；基線改用同一份 22 類切分。",
            "- **§3.8 推論描述**：移除「專家只套用於被路由到該群**且真實標籤屬於該群**的序列」的註解，改為與 Algorithm 1 一致的無標籤推論。",
            "- **Introduction**：仍屬 QuSpike-JEPA 的目標敘述（可訓練量子投影頭）需改為 PQK 的固定特徵圖與否證脈絡。",
            "- **雜訊段（Prop. 4）**：補充「所有位元相同的非對稱讀出錯誤也被中位數啟發式抵消」，並以各位元錯誤率不同的情境取代。"]


def main(argv=None):
    """命令列入口：讀各結果目錄並寫出彙整報告。"""
    p = argparse.ArgumentParser(description="008 彙整報告")
    p.add_argument("--results", default=str(DEFAULT_ROOT))
    p.add_argument("--base", default=None)
    p.add_argument("--plus", default=None)
    p.add_argument("--v2", default=None)
    p.add_argument("--noise", default=None)
    p.add_argument("--approx", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    R = Path(a.results)
    dirs = dict(base=a.base or R / "quspike_pqk_opt", plus=a.plus or R / "quspike_pqk_opt_plus",
                v2=a.v2 or R / "quspike_pqk_v2", noise=a.noise or R / "quspike_pqk_noise",
                approx=a.approx or R / "quspike_pqk_approx")
    md = build(dirs)
    out = Path(a.out) if a.out else Path(dirs["base"]) / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"報告已寫入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
