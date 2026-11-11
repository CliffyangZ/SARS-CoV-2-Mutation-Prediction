"""串接資料建置流程，輸出 Research/FE/processed/aa_scale{N}/（FR-003~005）。

用法：python -m Research.FE.build_dataset --scale {1,2}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Research.FE import load_spike as ls
from Research.FE.lineage_proxy import lineage_labels
from Research.FE.mutation_trend import monthly_mut_freq, train_trend, trend_table
from Research.FE.mutation_vectors import mutation_vectors, train_k
from Research.FE.position_map import build_position_map
from Research.FE.time_split import TEST_START, TRAIN_END, split_labels
from Research.utils.config import SCALES

PROCESSED = Path(__file__).resolve().parent / "processed"
EXPECTED = dict(raw=137132, filtered=132301, unique=13638, train=8161, test=3066)


def build(scale, fasta=None, out_dir=None, check_cache=True, expected=EXPECTED):
    """建資料集並寫檔；回傳 dict（供測試使用）。expected=None 時不核對論文數量。"""
    L = SCALES[scale]["L"]
    df, uniq, ref_seq, cnt = ls.load_all(fasta)
    X_all, X_u = ls.to_int(df.seq.tolist()), ls.to_int(uniq.seq.tolist())
    ref = ls.to_int([ref_seq])[0]
    split = split_labels(uniq.first_date)
    lineage = lineage_labels(X_u)
    tr = split == "train"
    w = uniq["count"].values.astype(float)

    # 位點選取與 k：只用訓練期資料（SC-001）
    pmap = build_position_map(X_u[tr], ref, w[tr], L)
    V = mutation_vectors(X_u, ref, pmap)
    k = train_k(V[tr])

    # 突變趨勢：以原始序列的月頻率；斜率只用訓練期月份
    month = df.date.dt.to_period("M")
    months = sorted(month.unique()); midx = month.map({m: i for i, m in enumerate(months)}).values
    F, n_m = monthly_mut_freq(X_all, ref, midx, len(months))
    train_months = [i for i, m in enumerate(months) if m.end_time <= TRAIN_END + pd.Timedelta(days=1)]
    slope, last = train_trend(F, n_m, train_months)
    test_m = next((i for i, m in enumerate(months) if m.start_time >= TEST_START), len(months) - 1)
    pred = np.clip(last + slope * (test_m - train_months[-1]), 0, 1)   # 外推到測試期起點月
    trend = trend_table(np.arange(1, ls.L_FULL + 1), slope, last, pred)

    out = Path(out_dir or PROCESSED / f"aa_scale{scale}"); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "vectors.npz", X=V, X_int=X_u, count=uniq["count"].values,
                        first_date=uniq.first_date.values.astype("datetime64[D]"),
                        split=split.astype(str), lineage_proxy=lineage.astype(str), ref=ref)
    meta = uniq.drop(columns="seq").assign(split=split, lineage_proxy=lineage, n_mut=((X_u != ref) & (X_u != 20)).sum(1))
    meta.to_csv(out / "unique_meta.csv", index=False)
    pmap.to_csv(out / "position_map.csv", index=False)
    trend.to_csv(out / "trend.csv", index=False)

    # 測試期原始序列中，訓練期從未出現過的比例（預測難度）
    train_set = set(uniq.seq[tr])
    te = df[df.date >= TEST_START]
    novel_frac = float((~te.seq.isin(train_set)).mean())
    n_split = {s: int((split == s).sum()) for s in ("train", "gap", "test")}
    mean_mut = {s: float(meta.n_mut[split == s].mean()) for s in ("train", "test")}
    res = dict(counts=cnt, split=n_split, k=k, novel_frac=novel_frac, mean_mut=mean_mut, out=out)

    problems = []
    if check_cache and expected:
        problems = _check(cnt, n_split, expected)
        problems += _check_cache(uniq, X_u)
    _report(out, scale, L, res, lineage, meta, pmap, problems)
    res["problems"] = problems
    return res


def _check(cnt, n_split, exp):
    """與論文 / notebook 預期數量比對。"""
    got = dict(cnt, train=n_split["train"], test=n_split["test"])
    return [f"{k} 預期 {v:,} 實得 {got[k]:,}" for k, v in exp.items() if got[k] != v]


def _check_cache(uniq, X_u):
    """與 notebook 產物 spike_uniq.npz 比對（序列集合與計數）。"""
    p = PROCESSED / "spike_uniq.npz"
    if not p.exists():
        return []
    c = np.load(p)
    a = {r.tobytes(): n for r, n in zip(X_u, uniq["count"].values)}
    b = {r.tobytes(): n for r, n in zip(c["X"], c["count"])}
    return [] if a == b else ["與 spike_uniq.npz 的序列／計數不一致"]


def _report(out, scale, L, r, lineage, meta, pmap, problems):
    """寫 dataset_report.md（FR-005）。"""
    c, s = r["counts"], r["split"]
    lab = pd.Series(lineage).value_counts()
    lines = [f"# 資料集報告（規模 {scale}，L={L}）", "",
             "## 各階段序列數", f"- 原始：{c['raw']:,}", f"- 品質過濾後：{c['filtered']:,}", f"- 去重後唯一序列：{c['unique']:,}", "",
             "## 時間切分（唯一序列，以 first_date 判定）",
             f"- 訓練期（≤2021-02-28）：{s['train']:,}", f"- 空白期：{s['gap']:,}（不參與訓練與評估）",
             f"- 測試期（≥2021-04-01）：{s['test']:,}", "",
             "## lineage 代理標籤（唯一序列數；近似標籤，非 Pango）"] + [f"- {k}：{v:,}" for k, v in lab.items()] + [
             "", f"## 測試期原始序列中「訓練期從未出現」的比例：{r['novel_frac']:.1%}", "",
             "## 突變數與 k",
             f"- 訓練期每條唯一序列平均突變數（全 1273 位點）：{r['mean_mut']['train']:.2f}；測試期：{r['mean_mut']['test']:.2f}",
             f"- 訓練期於 {L} 位點上的平均突變數四捨五入 → **k = {r['k']}**",
             f"- 位點來源：{pmap.origin.value_counts().to_dict()}", "",
             "## 與論文資料的差異",
             "- 論文用核苷酸資料與 GISAID/Omicron；本專案用 NCBI Virus 胺基酸 spike（快照 2021-10-12），無 Omicron。",
             "- 以時間切分做預測：訓練期 ≤2021-02-28，測試期 ≥2021-04-01；Delta-like 於訓練期為 0 條，故為完全外推。", "",
             "## 檢查", *([f"- ❌ {p}" for p in problems] or ["- ✅ 數量與 notebook 產物 spike_uniq.npz 一致"])]
    (out / "dataset_report.md").write_text("\n".join(lines) + "\n")


def load_dataset(scale, data_dir=None):
    """讀回已建好的資料集：dict(X, X_int, count, split, lineage, ref, pmap, trend, meta)。"""
    d = Path(data_dir) if data_dir else PROCESSED / f"aa_scale{scale}"
    v = np.load(d / "vectors.npz")
    return dict(X=v["X"], X_int=v["X_int"], count=v["count"], split=v["split"], lineage=v["lineage_proxy"],
                ref=v["ref"], pmap=pd.read_csv(d / "position_map.csv"), trend=pd.read_csv(d / "trend.csv"),
                first_date=v["first_date"])


def main():
    """命令列入口。"""
    ap = argparse.ArgumentParser(); ap.add_argument("--scale", type=int, choices=[1, 2], default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = build(a.scale)
    print(f"完成 → {r['out']}  k={r['k']}  split={r['split']}")
    if r["problems"]:
        print("資料檢查失敗：", *r["problems"], sep="\n  ", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
