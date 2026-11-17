"""報告產出（T013／T026–T028／T052）。

三個錨點必須並列（FR-025），且標籤歧義上界必須註明為**樂觀估計**——
研究報告 §7.2 的實測顯示它含大量記憶成分（57.3% 的測試序列有獨一無二的突變簽章）。

FR-028 守衛：本階段不送機，報告中不得出現任何聲稱來自真實硬體的分數。
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from Research.lineage_baselines.results_io import read_results

from . import criteria

# 三個錨點（見 Doc/QuSpike-JEPA-研究報告.md §7.2）
ANCHORS = (
    ("精確簽章查表（train→test）", 0.411, "可實現，但 58% 的測試序列查不到"),
    ("監督線性參考線", 0.747, "可實現；本框架的結構下限"),
    ("標籤歧義上界（簽章 oracle）", 0.930, "**鬆上界**：偷看標籤，57.3% 測試序列簽章獨一無二"),
)
CONTROL_LABEL = {
    "MAIN": "完整框架（PQK + 分群 + 跳接）",
    "C0": "C0 僅參考線（結構下限）",
    "C1": "C1 群內古典 RBF 核",
    "C2": "C2 古典 RFF 替身",
    "C3": "C3 無糾纏 PQK（量子性消融）",
    "C4": "C4 群內寬度匹配 MLP",
    "C5": "C5 不分群 PQK",
}


def collect(out_dir, split="test"):
    """讀結果檔，依設定彙整。"""
    g = defaultdict(lambda: dict(vals=[], accs=[], f1w=[], auc=[], seconds=[], cfg=None,
                                 groups=defaultdict(list), router=[], scales=[], diags=[],
                                 per_class=None, backends=set()))
    for r in read_results(out_dir):
        if r.get("status") != "complete":
            continue
        m = r.get(f"metrics_{split}")
        if not m:
            continue
        name = r["config"].get("control", r["config"]["name"])
        d = g[name]
        d["vals"].append(m["f1_macro"])
        d["accs"].append(m.get("accuracy"))
        d["f1w"].append(m.get("f1_weighted"))
        d["auc"].append(m.get("roc_auc_ovr"))
        d["seconds"].append((r.get("seconds") or {}).get("total", 0.0))
        d["cfg"] = r["config"]
        d["backends"].add(r["config"].get("backend", "simulator"))
        if r.get("router_accuracy") is not None:
            d["router"].append(r["router_accuracy"])
        if r["config"].get("expert_scale") is not None:
            d["scales"].append(r["config"]["expert_scale"])
        for k, v in (r.get(f"group_metrics_{split}") or {}).items():
            d["groups"][k].append(v)
        if r.get("kernel_diagnostics"):
            d["diags"].append(r["kernel_diagnostics"])
        if r.get(f"per_class_{split}"):
            d["per_class"] = r[f"per_class_{split}"]
    return g


def _ms(v):
    return criteria.mean_std([x for x in v if x is not None])


def _fmt(ms):
    return "—" if not np.isfinite(ms[0]) else f"{ms[0]:.3f}±{ms[1]:.3f}"


def assert_no_hardware_claims(g):
    """FR-028：本階段不得出現聲稱來自真實硬體的分數。"""
    bad = [n for n, d in g.items() if d["backends"] - {"simulator"}]
    if bad:
        raise ValueError(f"FR-028 違規：下列設定聲稱來自非模擬器後端：{bad}")


def table_main(g, split="test"):
    """主表：各設定的分數、混合權重與耗時。"""
    L = [f"## 主結果（{split} macro F1）", "",
         "| 設定 | 種子 | macro F1 | accuracy | F1-weighted | ROC-AUC | 混合權重 | 秒 |",
         "|---|---|---|---|---|---|---|---|"]
    for name in sorted(g, key=lambda n: -_ms(g[n]["vals"])[0]):
        d = g[name]
        sc = ",".join(str(x) for x in sorted(set(d["scales"]))) or "—"
        L.append(f"| {CONTROL_LABEL.get(name, name)} | {len(d['vals'])} | **{_fmt(_ms(d['vals']))}** | "
                 f"{_fmt(_ms(d['accs']))} | {_fmt(_ms(d['f1w']))} | {_fmt(_ms(d['auc']))} | "
                 f"{sc} | {np.mean(d['seconds']):.0f} |")
    return L


def table_criteria(g):
    """判準表：四項宣稱的增量與三態判定，並標示停止條件是否觸發。"""
    recs = criteria.evaluate({n: d["vals"] for n, d in g.items()})
    L = ["## 判準（事前訂定，雜訊下限 0.02）", "",
         "| 宣稱 | A | B | A − B | 判定 |", "|---|---|---|---|---|"]
    mark = {"holds": "**成立**", "fails": "不成立（較差）", "within_noise": "在雜訊內",
            "insufficient_data": "資料不足"}
    for r in recs:
        d = "—" if not np.isfinite(r["delta"]) else f"{r['delta']:+.4f}"
        L.append(f"| {r['claim']} | {r['a']} | {r['b']} | {d} | {mark[r['verdict']]} |")
    if criteria.stop_condition_met(recs):
        L += ["", "> **停止條件已觸發**（提案 §4）：量子特徵與量子性皆無增量。"
                  "不得以加大 n、改 τ／T／gamma 的方式重試求轉正。"]
    return L


def table_groups(g, split="test"):
    """分群表：量子只應改變難群（SC-004）。"""
    names = [n for n in sorted(g) if g[n]["groups"]]
    if not names:
        return []
    keys = sorted({k for n in names for k in g[n]["groups"]})
    L = [f"## 分群 macro F1（{split}）", "",
         "| 設定 | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for n in names:
        row = [_fmt(_ms(g[n]["groups"][k])) if k in g[n]["groups"] else "—" for k in keys]
        L.append(f"| {CONTROL_LABEL.get(n, n)} | " + " | ".join(row) + " |")
    L += ["", "> 量子只應改變難群（delta／b1）。若 others 也變了，代表路由或兜底有問題（SC-004）。"]
    return L


def table_anchors(g):
    """錨點表：三個已知基準並列，並註明上界為樂觀估計。"""
    best = max((_ms(d["vals"])[0] for d in g.values()), default=float("nan"))
    L = ["## 錨點對照", "", "| 錨點 | macro F1 | 與本輪最佳的差距 | 性質 |", "|---|---|---|---|"]
    for label, v, note in ANCHORS:
        gap = "—" if not np.isfinite(best) else f"{best - v:+.3f}"
        L.append(f"| {label} | {v:.3f} | {gap} | {note} |")
    L += ["", "> 上界為**樂觀估計**，不是可達成的目標（見研究報告 §7.2）。"]
    return L


def table_diagnostics(g):
    """核值分佈與半正定診斷（核集中是量子核的已知退化模式）。"""
    rows = []
    for n in sorted(g):
        for diag in g[n]["diags"][:1]:
            for grp, d in sorted(diag.items()):
                rows.append((CONTROL_LABEL.get(n, n), grp, d))
    if not rows:
        return []
    L = ["## 核值分佈與半正定診斷", "",
         "| 設定 | 群 | gamma | off-diag 平均 | off-diag 變異 | 近似 | 半正定違反 |",
         "|---|---|---|---|---|---|---|"]
    for name, grp, d in rows:
        L.append(f"| {name} | {grp} | {d.get('gamma', float('nan')):.4g} | "
                 f"{d.get('offdiag_mean', float('nan')):.4f} | {d.get('offdiag_std', float('nan')):.4f} | "
                 f"{d.get('approximation', '—')} | {'是：' + str(d.get('psd_correction')) if d.get('psd_violation') else '否'} |")
    L += ["", "> off-diag 變異極小代表核值集中（Huang et al. 指出的退化模式）。"]
    return L


def table_router(g):
    """路由準確率：量子專家能改善的上限。"""
    acc = [x for d in g.values() for x in d["router"]]
    if not acc:
        return []
    ms = _ms(acc)
    return ["## 路由準確率", "",
            f"樣本被分到其真實類別所屬群的比例：**{_fmt(ms)}**。",
            "", "> 這是量子專家能改善的上限——被路錯群的樣本，專家根本看不到。"]


def build(out_dir, split="test"):
    """產生報告文字。任一判準不成立仍完整產出（FR-027）。"""
    g = collect(out_dir, split)
    assert_no_hardware_claims(g)
    L = ["# QuSpike-PQK 結果", "",
         f"資料來源：`{out_dir}`；{split} macro F1；只計入 status=complete。",
         "量子部分不訓練線路參數，故量子特徵與種子無關（可跨種子快取重用）。",
         "**本階段全部為模擬器結果，未送出任何硬體作業。**", ""]
    if not g:
        return "\n".join(L + ["（尚無結果）"]) + "\n"
    for fn in (table_main, table_criteria, table_groups, table_router, table_anchors,
               table_diagnostics):
        part = fn(g) if fn is not table_main else fn(g, split)
        if part:
            L += part + [""]
    return "\n".join(L) + "\n"


def main(argv=None):
    """命令列入口。"""
    p = argparse.ArgumentParser(description="QuSpike-PQK 報告")
    from .run import DEFAULT_OUT

    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--write", default=None)
    a = p.parse_args(argv)
    txt = build(a.out, a.split)
    print(txt)
    if a.write:
        Path(a.write).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
