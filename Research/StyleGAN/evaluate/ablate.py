"""多種子彙總與兩輪消融（002 FR-004a、FR-011、FR-012；research R8）。

第 1 輪：legacy 與每種新編碼單獨比較（相同種子、相同步數）；
第 2 輪：只把第 1 輪判定「有效」的編碼組合起來再比一次。
已存在的 run 不會重訓（沿用其 metrics.json），也不會覆寫。

用法：python -m Research.StyleGAN.evaluate.ablate --round {1,2} --seeds 0 1 2 --results Results/StyleGAN/run03
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "legacy"
ROUND1 = ["legacy", "drop_const", "nonorm", "freq_order"]
EFFECTIVE_RATIO = 2.0        # 「有效」：相異突變集平均 ≥ legacy 最大值的 2 倍（初始值，R8）
SC = dict(distinct=100, seed_min=30, coverage=16, count_tol=1.0, corr=0.7)   # 成功標準門檻（spec SC-002、003、003a）
COLS = ["round", "encoding", "seed", "distinct_mutation_sets", "position_coverage", "freq_correlation",
        "random_freq_correlation", "mean_mutation_count", "train_mean_count", "min_hamming_model",
        "min_hamming_test_k_diagnostic", "min_hamming_freq_baseline", "loss_g_final", "collapse_warning"]


def run_dir_for(results, encoding, seed, stage="A", scale=1, loss="logistic"):
    """與 Config.run_dir 相同的目錄規則（legacy 不加後綴）。"""
    suffix = "" if encoding == "legacy" else f"_{encoding}"
    return Path(results) / stage / f"scale{scale}" / f"classical_{loss}{suffix}_seed{seed}"


def _ensure_dataset(encoding):
    """該編碼的資料集不存在時先建立（組合編碼在第 2 輪才第一次用到）。"""
    from Research.StyleGAN.FE.build_dataset import build, dataset_dir
    if not (dataset_dir(1, encoding) / "vectors.npz").exists():
        build(scale=1, encoding=encoding)


def run_one(encoding, seed, results, steps=3500, device="auto", n=1000):
    """訓練 → 生成 → 評估一個 (編碼, 種子)；已有 metrics.json 就沿用，不重訓也不覆寫。回傳 run 目錄。"""
    from Research.StyleGAN.evaluate.evaluate import evaluate
    from Research.StyleGAN.evaluate.generate import generate
    from Research.StyleGAN.train import train_StyleGAN as tc
    rd = run_dir_for(results, encoding, seed)
    if (rd / "metrics.json").exists():
        return rd
    _ensure_dataset(encoding)
    if not (rd / "checkpoint.pt").exists():
        try:
            tc.main(["--scale", "1", "--encoding", encoding, "--seed", str(seed), "--steps", str(steps),
                     "--results", str(results), "--wandb", "off", "--device", device])
        except SystemExit as e:                       # 訓練失敗（NaN 等）：留下 status.json，回報後略過
            if e.code:
                print(f"⚠ 訓練失敗：{encoding} seed={seed}（見 {rd / 'status.json'}）")
                return rd
    generate(rd, n=n)
    evaluate(rd)
    return rd


def row_from_run(rd, round_no, encoding, seed):
    """從一個 run 目錄讀出消融彙總的一列（AblationRow）。"""
    rd = Path(rd)
    div = json.loads((rd / "diversity.json").read_text())
    pred = json.loads((rd / "metrics.json").read_text())["prediction"]
    pick = lambda src, kind: next(r["min_hamming_mean"] for r in pred if r["source"] == src and r["k_source"] == kind)
    cfg = json.loads((rd / "config.json").read_text())
    from Research.StyleGAN.train.train_StyleGAN import load_train_vectors
    train_mean = float(load_train_vectors(1, None, encoding).sum(1).mean())
    loss = pd.read_csv(rd / "losses.csv").loss_g.iloc[-1]
    return dict(round=round_no, encoding=encoding, seed=seed, distinct_mutation_sets=div["distinct_mutation_sets"],
                position_coverage=div["position_coverage"], freq_correlation=div["freq_correlation"],
                random_freq_correlation=div["reference"]["random"]["freq_correlation"],
                mean_mutation_count=div["mean_mutation_count"], train_mean_count=train_mean,
                min_hamming_model=pick("model", "model"), min_hamming_test_k_diagnostic=pick("model", "test_mean_diagnostic"),
                min_hamming_freq_baseline=pick("freq", "model"), loss_g_final=float(loss),
                collapse_warning=bool(div["collapse_warning"]))


def summarize(df):
    """每個編碼的各指標平均、最小、最大（FR-011）。回傳寬表：每個編碼一列。"""
    cols = ["distinct_mutation_sets", "position_coverage", "freq_correlation", "mean_mutation_count",
            "min_hamming_model", "min_hamming_freq_baseline"]
    g = df.groupby("encoding", sort=False)[cols].agg(["mean", "min", "max"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_seeds"] = df.groupby("encoding", sort=False).size()
    return g


def judge(summary, base=BASE, ratio=EFFECTIVE_RATIO):
    """逐編碼判定「有效／無效／變差」（R8）。
    有效：相異突變集平均 ≥ ratio × base 最大值，且位點覆蓋平均 > base 覆蓋最大值。
    變差：相異突變集平均 < base 最小值 且 位點覆蓋平均 < base 覆蓋最小值。其餘為無效。"""
    b = summary.loc[base]
    out = {}
    for enc, r in summary.iterrows():
        if enc == base:
            continue
        if r["distinct_mutation_sets_mean"] >= ratio * b["distinct_mutation_sets_max"] \
                and r["position_coverage_mean"] > b["position_coverage_max"]:
            out[enc] = "有效"
        elif r["distinct_mutation_sets_mean"] < b["distinct_mutation_sets_min"] \
                and r["position_coverage_mean"] < b["position_coverage_min"]:
            out[enc] = "變差"
        else:
            out[enc] = "無效"
    return out


def plan_round2(verdicts):
    """第 2 輪要組合的編碼名稱。有效項目少於 2 個時沒有可組合的對象，回傳 None。"""
    eff = [e for e, v in verdicts.items() if v == "有效"]
    return "+".join(eff) if len(eff) >= 2 else None


def success_table(df):
    """逐編碼對照成功標準 SC-002、003、003a、004（SC-004 只用模型自決 k 的正式結果）。"""
    rows = []
    for enc, d in df.groupby("encoding", sort=False):
        if enc == BASE:
            continue
        gap = (d.min_hamming_freq_baseline - d.min_hamming_model).mean()
        rows.append(dict(
            encoding=enc,
            SC002=bool(d.distinct_mutation_sets.mean() >= SC["distinct"] and d.distinct_mutation_sets.min() >= SC["seed_min"]),
            SC003=bool(d.position_coverage.mean() >= SC["coverage"]
                       and abs(d.mean_mutation_count.mean() - d.train_mean_count.mean()) <= SC["count_tol"]),
            SC003a=bool(d.freq_correlation.mean() >= SC["corr"] and d.freq_correlation.mean() > d.random_freq_correlation.mean()),
            SC004=bool(gap > d.min_hamming_model.std(ddof=0)), gap_hamming=float(gap)))
    return pd.DataFrame(rows)


def write_report(df, out_dir, round_no):
    """寫 ablation_summary.csv 與 ablation_report.md（含判定、依據、SC 對照與結論，SC-007）。"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(out / "ablation_summary.csv", index=False)
    summ = summarize(df)
    verdicts = judge(summ) if BASE in summ.index else {}
    sc = success_table(df) if len(verdicts) else pd.DataFrame()
    L = [f"# 消融報告（第 {round_no} 輪）", "",
         f"每個編碼 {int(summ.n_seeds.max())} 個種子；「有效」＝相異突變集平均 ≥ {EFFECTIVE_RATIO:g}× legacy 最大值，且位點覆蓋平均高於 legacy 最大值（初始門檻，可修訂並註明理由）。", "",
         "## 各編碼的多樣性（平均［最小–最大］）", "",
         "| 編碼 | 相異突變集 | 位點覆蓋 | 頻率相關 | 平均突變數 | 最小 Hamming（模型 / 依頻率） | 判定 |", "|---|---|---|---|---|---|---|"]
    for enc, r in summ.iterrows():
        f = lambda c, p=1: f"{r[c + '_mean']:.{p}f}［{r[c + '_min']:.{p}f}–{r[c + '_max']:.{p}f}］"
        L.append(f"| {enc} | {f('distinct_mutation_sets')} | {f('position_coverage')} | {f('freq_correlation', 2)} | "
                 f"{f('mean_mutation_count', 2)} | {r['min_hamming_model_mean']:.2f} / {r['min_hamming_freq_baseline_mean']:.2f} | "
                 f"{'基準' if enc == BASE else verdicts.get(enc, '—')} |")
    if len(sc):
        L += ["", "## 成功標準對照（SC-004 取模型自決的突變數；測試期 k 只作診斷）", "",
              "| 編碼 | SC-002 相異突變集 | SC-003 覆蓋與突變數 | SC-003a 頻率相關 | SC-004 預測 | Hamming 差距（依頻率−模型） |", "|---|---|---|---|---|---|"]
        mark = lambda b: "✅" if b else "❌"
        for _, r in sc.iterrows():
            L.append(f"| {r.encoding} | {mark(r.SC002)} | {mark(r.SC003)} | {mark(r.SC003a)} | {mark(r.SC004)} | {r.gap_hamming:.2f} |")
    eff = [e for e, v in verdicts.items() if v == "有效"]
    L += ["", "## 結論", ""]
    if eff and not len(sc) == 0 and sc.SC002.any():
        L.append(f"- 有效編碼：{', '.join(eff)}。至少一種編碼達到 SC-002，**支持「編碼是模式塌縮的主因（之一）」**。")
    elif eff:
        L.append(f"- 有效編碼：{', '.join(eff)}，但未達 SC-002 的絕對門檻；編碼有幫助，但不足以單獨解決塌縮。")
    else:
        L.append("- **沒有任何編碼判定為有效：encoding 並非唯一主因（SC-007）。** 下一步假設：判別器過強（rahinge＋R1）、學習率與漸進式切換、改用共現矩陣表示。")
    worse = [e for e, v in verdicts.items() if v == "變差"]
    if worse:
        L.append(f"- 變差的編碼：{', '.join(worse)}（如實列入失敗紀錄）。")
    L.append("- 判定與 SC 對照皆以上表為準；門檻若修訂須註明理由。")
    (out / "ablation_report.md").write_text("\n".join(L) + "\n")
    return verdicts, sc


def main(argv=None):
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results", required=True)
    ap.add_argument("--encodings", nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    root = Path(a.results)
    if a.round == 1:
        encs = a.encodings or ROUND1
    else:                                                    # 第 2 輪：只組合第 1 輪有效的項目
        prev = pd.read_csv(root / "A" / "ablation" / "round1" / "ablation_summary.csv")
        verdicts = judge(summarize(prev))
        combo = plan_round2(verdicts)
        if combo is None and not a.encodings:
            print("第 1 輪有效的編碼少於 2 個，沒有可組合的對象；第 2 輪略過。", {k: v for k, v in verdicts.items()})
            return
        encs = [BASE] + (a.encodings or [combo])
    rows = []
    for e in encs:
        for s in a.seeds:
            rd = run_one(e, s, root, a.steps, a.device)
            if (rd / "metrics.json").exists():
                rows.append(row_from_run(rd, a.round, e, s))
    df = pd.DataFrame(rows)
    verdicts, _ = write_report(df, root / "A" / "ablation" / f"round{a.round}", a.round)
    print({k: v for k, v in verdicts.items()}, "→", root / "A" / "ablation" / f"round{a.round}")


if __name__ == "__main__":
    main()
