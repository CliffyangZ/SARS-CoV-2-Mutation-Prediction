"""彙整報告：所有 baseline 的平均±標準差、與論文 Table 2 的對照、最差譜系與偏離清單。

用法：
    python -m Research.lineage_baselines.report --results Results/lineage_baselines
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import results_io as R

PAPER_JSON = Path(__file__).with_name("paper_table2.json")
KEYS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "roc_auc_ovr"]
SHORT = {"accuracy": "Acc", "precision_weighted": "Prec(w)", "recall_weighted": "Rec(w)",
         "f1_weighted": "F1(w)", "f1_macro": "F1(macro)", "roc_auc_ovr": "AUC"}


def aggregate(results):
    """依（模型，表示）彙整已完成結果：平均、標準差（ddof=1）、種子數、是否決定性、耗時、最差譜系。"""
    groups = collections.defaultdict(list)
    for r in results:
        if r.get("status") == "complete":
            groups[(r["model"], r["representation"])].append(r)
    out = {}
    for key, rs in groups.items():
        mean = {k: float(np.mean([r["metrics"][k] for r in rs])) for k in KEYS}
        std = None if len(rs) < 2 else {k: float(np.std([r["metrics"][k] for r in rs], ddof=1)) for k in KEYS}
        f1 = collections.defaultdict(list)
        for r in rs:
            for c in r.get("per_class", []):
                f1[c["lineage"]].append(c["f1"])
        worst = min(((l, float(np.mean(v))) for l, v in f1.items()), key=lambda t: t[1]) if f1 else None
        out[key] = {"n_seeds": len(rs), "deterministic": bool(rs[0].get("deterministic")), "mean": mean, "std": std,
                    "train_seconds": float(np.mean([r["train_seconds"] for r in rs])), "worst": worst,
                    "pretrained": any(r.get("hyperparameters", {}).get("pretrained_imagenet") for r in rs)}
    return out


def _fmt(a, k):
    """格式化一個指標：多種子為 平均±標準差，單一或決定性只顯示平均。"""
    m = a["mean"][k]
    return f"{m:.3f}" if a["std"] is None else f"{m:.3f}±{a['std'][k]:.3f}"


def _secs(s):
    """把秒數格式化成易讀字串。"""
    return f"{s:.1f}s" if s < 120 else (f"{s / 60:.1f}min" if s < 7200 else f"{s / 3600:.2f}h")


def build_report(results, paper):
    """產生報告 Markdown 全文。"""
    agg = aggregate(results)
    pre = next((r["preprocessing"] for r in results if r.get("preprocessing")), {})
    desc = (f"NCBI 來源、{len(pre['label_map'])} 類、train {pre['n_train']}／test {pre['n_test']} 條、時間切分"
            if pre else "NCBI 來源、時間切分")
    lines = ["# Baseline 重現報告", "", f"資料：{desc}，與論文的 GISAID Spike7k（7,000 條、22 類）不同，"
             "**數值不可直接比較，只比相對排名**。", ""]
    lines += ["## 彙整表（平均±標準差；決定性模型或單一種子只列平均）", "",
              "| 模型 | 表示 | 種子數 | 決定性 | " + " | ".join(SHORT[k] for k in KEYS) + " | 耗時 |",
              "|" + "---|" * (5 + len(KEYS))]
    for (m, rep), a in sorted(agg.items()):
        lines.append(f"| {m} | {rep} | {a['n_seeds']} | {'是' if a['deterministic'] else '否'} | "
                     + " | ".join(_fmt(a, k) for k in KEYS) + f" | {_secs(a['train_seconds'])} |")
    lines += ["", "## 與論文 Table 2 的對照", ""]
    prow = {(r["model"], r["representation"]): r for r in paper["rows"]}
    lines += ["| 模型 | 表示 | 本重現 Acc | 論文 Acc | 本重現 F1(macro) | 論文 F1(macro) | 本重現 AUC | 論文 AUC |", "|---|---|---|---|---|---|---|---|"]
    shared = []
    for (m, rep), a in sorted(agg.items()):
        p = prow.get((m, rep))
        if p is None:
            lines.append(f"| {m} | {rep} | {a['mean']['accuracy']:.3f} | 論文未提供 | {a['mean']['f1_macro']:.3f} | 論文未提供 | {a['mean']['roc_auc_ovr']:.3f} | 論文未提供 |")
            continue
        shared.append((m, rep, a, p))
        lines.append(f"| {m} | {rep} | {a['mean']['accuracy']:.3f} | {p['accuracy']:.3f} | {a['mean']['f1_macro']:.3f} | {p['f1_macro']:.3f} | {a['mean']['roc_auc_ovr']:.3f} | {p['roc_auc_ovr']:.3f} |")
    lines += ["", "### 排名一致性", ""]
    if len(shared) >= 3:
        for k, name in (("accuracy", "Acc"), ("f1_macro", "F1(macro)")):
            ours = [s[2]["mean"][k] for s in shared]
            theirs = [s[3][k] for s in shared]
            rho = spearmanr(ours, theirs).statistic
            lines.append(f"- {name}：{len(shared)} 個共同模型的 Spearman 排名相關 = {rho:.2f}")
    else:
        lines.append("- 共同模型少於 3 個，無法計算排名相關（Spearman）。")
    lines += ["", "不一致處需以資料差異解釋（NCBI 對 GISAID、11 類對 22 類、類別分布、時間切分、自編碼器細節未知、NTK 未重現），"
              "無法解釋者如實保留，不調整實驗去配合。", "", "## 各模型學得最差的譜系（各類別 F1 對種子平均）", "",
              "| 模型 | 表示 | 最差譜系 | F1 |", "|---|---|---|---|"]
    for (m, rep), a in sorted(agg.items()):
        w = a["worst"]
        lines.append(f"| {m} | {rep} | {w[0] if w else '-'} | {w[1]:.3f} |" if w else f"| {m} | {rep} | - | - |")
    inc = [r for r in results if r.get("status") != "complete"]
    lines += ["", "## 未完成的訓練", ""]
    lines += [f"- {r['model']}/{r['representation']}/seed{r['seed']}：已完成 {r.get('epochs_done')} 輪" for r in inc] or ["- 無"]
    pre = [k for k, a in agg.items() if a["pretrained"]]
    lines += ["", "## 偏離清單", "",
              "- **NTK 未重現**：論文未提供公式、寬度或深度；傳統分類器吃的是自編碼器嵌入，不是 NTK。",
              "- 自編碼器層寬、各分類器與各深度模型的超參數、批次大小、學習率皆為自行選用（結果檔 `provenance` 標為 chosen）。",
              "- 論文 k=5 折不重現，改為單一時間切分、多種子；傳統模型 5 個種子，深度模型 3 個種子。",
              "- 論文 Table 2 沒有 MTF 的列，MTF 結果無論文對照；GAF 列的演算法名稱為轉錄時推斷。",
              f"- ResNet34 {'使用 ImageNet 預訓練權重（' + ', '.join(f'{m}/{r}' for m, r in pre) + '）' if pre else '未使用 ImageNet 預訓練權重（隨機初始化），與論文的預訓練不同'}。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    """命令列入口：讀取結果並寫出 report.md。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="Results/lineage_baselines")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = build_report(R.read_results(a.results), json.loads(PAPER_JSON.read_text()))
    out = Path(a.out) if a.out else Path(a.results) / "report.md"
    out.write_text(text)
    print(f"已寫出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
