"""彙整報告：與 005 baseline 同表、消融、時間漂移、有限取樣、假說結論。

只彙整 status=complete 且未塌陷的結果；塌陷與略過者另列。結論只依「差異是否超過種子標準差」判斷。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from Research.lineage_baselines.results_io import read_results

from .run import DEFAULT_OUT

KEYS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "roc_auc_ovr"]


def _ms(vals):
    v = np.asarray(vals, float)
    return (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0) if len(v) else (float("nan"), float("nan"))


def _fmt(ms):
    return f"{ms[0]:.3f}±{ms[1]:.3f}" if np.isfinite(ms[0]) else "—"


def group(results):
    """依設定名分組，回傳 {name: dict(ok=[...], collapsed=[...], skipped=[...])}。"""
    g = defaultdict(lambda: dict(ok=[], collapsed=[], skipped=[]))
    for r in results:
        name = r["config"]["name"]
        if r.get("status") == "skipped":
            g[name]["skipped"].append(r)
        elif r.get("collapsed"):
            g[name]["collapsed"].append(r)
        else:
            g[name]["ok"].append(r)
    return g


def agg(rs, split, key="f1_macro", suffix=""):
    """對一組結果取某指標的 (平均, 標準差)。"""
    return _ms([r[f"metrics_{split}{suffix}"][key] for r in rs if f"metrics_{split}{suffix}" in r])


def differs(a, b):
    """兩個 (mean, std) 的差是否超過兩者標準差中較大者。"""
    return abs(a[0] - b[0]) > max(a[1], b[1], 1e-12)


def table_main(g, baselines):
    """主結果表（含 005 baseline）。"""
    out = ["## 1. 主結果（與 005 baseline 同表，macro F1 為主）", "",
           "| 方法 | 種子 | accuracy | weighted F1 | **macro F1** | ROC AUC | 訓練秒 | 有效秩 | 資料 |", "|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(n for n in g if "__" not in n):
        rs = g[name]["ok"]
        for split in ("val", "test"):
            sub = [r for r in rs if f"metrics_{split}" in r]
            if not sub:
                continue
            m = {k: agg(sub, split, k) for k in KEYS}
            sec = _ms([r["train_seconds"] for r in sub])[0]
            rank = _ms([r.get("effective_rank") for r in sub if r.get("effective_rank") is not None])[0]
            out.append(f"| {name} | {len(sub)} | {_fmt(m['accuracy'])} | {_fmt(m['f1_weighted'])} | **{_fmt(m['f1_macro'])}** | "
                       f"{_fmt(m['roc_auc_ovr'])} | {sec:.0f} | {rank:.1f} | {split} |")
    by = defaultdict(list)
    for r in baselines:
        if r.get("status") == "complete" and "metrics" in r:
            by[(r["model"], r["representation"])].append(r["metrics"])
    for (mod, rep), ms in sorted(by.items()):
        m = {k: _ms([x[k] for x in ms]) for k in KEYS}
        out.append(f"| 005：{mod}／{rep} | {len(ms)} | {_fmt(m['accuracy'])} | {_fmt(m['f1_weighted'])} | **{_fmt(m['f1_macro'])}** | {_fmt(m['roc_auc_ovr'])} | — | — | test |")
    bad = [(n, len(v["collapsed"]), len(v["skipped"])) for n, v in g.items() if v["collapsed"] or v["skipped"]]
    if bad:
        out += ["", "塌陷或略過的執行（不入平均）：" + "；".join(f"{n}（塌陷 {c}、略過 {s}）" for n, c, s in bad)]
    return out


def table_ablation(g):
    """各消融軸的表格。"""
    axes = defaultdict(list)
    for n in g:
        if "__" in n:
            base, tag = n.split("__", 1)
            axes[tag.split("=")[0]].append((tag, n))
    out = ["", "## 2. 消融（單因素；macro F1，validation，平均±標準差）", ""]
    for ax, items in sorted(axes.items()):
        out += [f"### {ax}", "", "| 取值 | 種子 | val macro F1 | test macro F1 | 與前一取值差異是否超過標準差 |", "|---|---|---|---|---|"]
        prev = None
        def _key(t):
            v = t[0].split("=", 1)[1]
            try:
                return (0, float(v), v)
            except ValueError:
                return (1, 0.0, v)

        for tag, n in sorted(items, key=_key):
            rs = g[n]["ok"]
            v, t = agg(rs, "val"), agg(rs, "test")
            flag = "—" if prev is None else ("是" if differs(prev, v) else "否（在雜訊內）")
            out.append(f"| {tag.split('=')[1]} | {len(rs)} | {_fmt(v)} | {_fmt(t)} | {flag} |")
            prev = v
        if ax in ("entangle", "trainable_obs", "noise"):
            ref = agg(g["jepa_VA_quantum_n6L3"]["ok"], "val") if "jepa_VA_quantum_n6L3" in g else (float("nan"), float("nan"))
            out.append(f"| （預設設定 jepa_VA_quantum_n6L3） | {len(g.get('jepa_VA_quantum_n6L3', {}).get('ok', []))} | {_fmt(ref)} | — | — |")
        best = max(items, key=lambda t: np.nan_to_num(agg(g[t[1]]["ok"], "val")[0], nan=-1))
        out += ["", f"最佳取值：{best[0].split('=')[1]}", ""]
    return out


def table_drift(g):
    """validation 對 test 的落差與最差譜系。"""
    out = ["", "## 3. 時間漂移與最差譜系", "", "| 方法 | val macro F1 | test macro F1 | 落差（val−test） | 最差譜系（test F1）|", "|---|---|---|---|---|"]
    for n in sorted(x for x in g if "__" not in x):
        rs = [r for r in g[n]["ok"] if "metrics_test" in r]
        if not rs:
            continue
        v, t = agg(rs, "val"), agg(rs, "test")
        pc = defaultdict(list)
        for r in rs:
            for c in r.get("per_class_test", []):
                pc[c["lineage"]].append(c["f1"])
        worst = min(pc.items(), key=lambda kv: np.mean(kv[1])) if pc else None
        out.append(f"| {n} | {_fmt(v)} | {_fmt(t)} | {v[0] - t[0]:+.3f} | {worst[0] + f' ({np.mean(worst[1]):.2f})' if worst else '—'} |")
    return out


def table_noise(g):
    """有限取樣與雜訊的掉點表。"""
    out = ["", "## 4. 有限取樣／雜訊的掉點（val macro F1）", "", "| 設定 | 無雜訊 | 有雜訊 | 掉點 |", "|---|---|---|---|"]
    n_rows = 0
    for n in sorted(g):
        rs = [r for r in g[n]["ok"] if "metrics_val_noisy" in r and (r["config"].get("shots") or r["config"].get("noise"))]
        if rs:
            c = _ms([r["metrics_val"]["f1_macro"] for r in rs]); z = _ms([r["metrics_val_noisy"]["f1_macro"] for r in rs])
            sh, no = rs[0]["config"].get("shots"), rs[0]["config"].get("noise")
            out.append(f"| {n}（shots={sh}, noise={no}） | {_fmt(c)} | {_fmt(z)} | {c[0] - z[0]:+.3f} |"); n_rows += 1
    return out if n_rows else out + ["| （本次未執行有限取樣設定） | | | |"]


def _pick(g, split, **kw):
    n = kw["name"]
    return agg(g[n]["ok"], split) if n in g and g[n]["ok"] else (float("nan"), float("nan"))


def verdict(a, b, name_a, name_b, split="test"):
    """兩個 (平均, 標準差) 的差異是否超過標準差。"""
    if not (np.isfinite(a[0]) and np.isfinite(b[0])):
        return "不確定（缺少資料）"
    if differs(a, b):
        return f"{name_a if a[0] > b[0] else name_b} 較佳，差 {abs(a[0] - b[0]):.3f}，超過標準差"
    return f"差異 {abs(a[0] - b[0]):.3f} 在種子標準差內 → 不確定"


def hypotheses(g):
    """四項假說的結論。"""
    split = "test" if any("metrics_test" in r for v in g.values() for r in v["ok"]) else "val"
    out = ["", f"## 5. 假說結論（依 {split} macro F1；差異超過種子標準差才算有差）", ""]
    va, vb, vc = (_pick(g, split, name=f"jepa_{v}_classical") for v in ("VA", "VB", "VC"))
    out.append(f"- **正樣本變體**：同譜系 {_fmt(va)}／親緣鄰近 {_fmt(vb)}／單序列破壞 {_fmt(vc)}。同譜系 vs 單序列：{verdict(va, vc, '同譜系', '單序列')}；鄰近 vs 單序列：{verdict(vb, vc, '鄰近', '單序列')}。")
    q = _pick(g, split, name="jepa_VA_quantum_n6L3")
    q1 = _pick(g, split, name="jepa_VA_quantum_1layer_n6L3")
    out.append(f"- **編碼層數**：多層 {_fmt(q)} vs 單層 {_fmt(q1)}：{verdict(q, q1, '多層', '單層')}。")
    cm = _pick(g, split, name="jepa_VA_classical_matched")
    out.append(f"- **參數效率**：量子 {_fmt(q)} vs 參數量匹配古典 {_fmt(cm)}：{verdict(q, cm, '量子', '古典')}（不主張量子優勢）。注意：匹配到約 66 個參數時古典頭的隱藏層只有約 3 個單元，差距可能來自古典頭太窄，而不是量子結構；未以更寬的古典頭驗證。")
    bal = [n for n in g if n.endswith("anchor_sampling=balanced")]
    nat = _pick(g, split, name="jepa_VA_classical")
    nat_v = agg(g["jepa_VA_classical"]["ok"], "val") if "jepa_VA_classical" in g else (float("nan"), float("nan"))
    out.append("- **類別不平衡**（消融只跑 validation，故以 validation 比較）：" + ("；".join(f"自然 {_fmt(nat_v)} vs 平衡 {_fmt(agg(g[n]['ok'], 'val'))}：{verdict(nat_v, agg(g[n]['ok'], 'val'), '自然', '平衡')}" for n in bal) or "不確定（未執行類別平衡消融）") + "。")
    return out


def provenance(results):
    """設定來源（文件或自訂）與偏離說明。"""
    r = next((x for x in results if "config_origin" in x and "variant" in x["config_origin"]), None)
    if not r:
        return []
    doc = sorted(k for k, v in r["config_origin"].items() if v == "doc")
    ch = sorted(k for k, v in r["config_origin"].items() if v == "chosen")
    return ["", "## 6. 設定來源", "", f"- 來自文件：{', '.join(doc)}", f"- 自行決定：{', '.join(ch)}",
            "- 與文件的偏離：突變以對參考的序列比對取得（不截斷）；位置層級遮蔽數量依突變數比例；親緣鄰近距離為突變集合對稱差。"]


def notes():
    """報告的注意事項。"""
    return ["", "## 7. 注意事項", "",
            "- `jepa_VA_classical` 與 `jepa_VA_classical_matched` 是同一個實作（古典頭寬度本來就匹配到量子頭參數量），兩者為重複測量，不是兩個不同的對照。",
            "- ESM-2 的嵌入只算一次，三個「種子」只差 probe 的隨機性，標準差 0.000 不代表模型穩定，只代表一次測量。",
            "- 有效秩檢查只看維度使用數（正規化奇異值，與尺度無關），抓不到「變異趨近 0 但仍有殘餘雜訊」的塌陷：`whitening=identity` 與 `none` 的 macro F1 接近隨機（約 0.09–0.12）卻未被標示為塌陷。",
            "- 同一設定同一種子重跑在 GPU 上並非逐位元一致，validation macro F1 曾相差到約 0.06（`jepa_VA_quantum_n6L3` seed 0：0.562、0.605、0.614、0.621），解讀消融差異時應以 3 種子平均與標準差為準。",
            "- test 為較晚時間區段，但多數 JEPA 設定的 test macro F1 反而高於 validation（落差為負），與「時間漂移造成掉點」的預期相反；是否因為 test 譜系組成較易分尚未查證。",
            "- 所有 JEPA 變體與 ESM-2 的 macro F1 都低於 005 的 onehot 基準（0.82–0.83）；本報告未顯示任何 JEPA 設定勝過簡單基準。"]


def main(argv=None):
    """命令列入口：彙整結果並寫出報告。"""
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(DEFAULT_OUT))
    p.add_argument("--baselines", default="Results/lineage_baselines")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    results = [r for r in read_results(a.results) if "config" in r]
    base = read_results(a.baselines) if Path(a.baselines).exists() else []
    g = group(results)
    lines = ["# QuSpike-JEPA 彙整報告", "", "資料與切分同 005；validation = train 內各譜系最晚 15%；test 為較晚時間區段。不主張量子優勢。", ""]
    lines += table_main(g, base) + table_ablation(g) + table_drift(g) + table_noise(g) + hypotheses(g) + provenance(results) + notes()
    out = Path(a.out or Path(a.results) / "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"報告寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
