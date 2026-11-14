"""資料多樣性上限分析（004 US1，FR-001～FR-003）：不訓練，估計資料本身能支持多少種相異突變集。

口徑與生成評估相同：在編碼後 L 維 0/1 向量上，相異向量＝相異突變集（見 Clarifications）。
四個數字：A 訓練期相異向量數；B 依真實分佈（以序列出現次數加權）抽樣的相異數；
C 各位點依訓練期頻率獨立抽樣的相異數；D 測試期相異向量數與未見比例。

用法：python -m Research.StyleGAN.evaluate.ceiling --scale 1 --encoding drop_const --out Results/StyleGAN/run04/ceiling
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

TARGET = 100                 # 原目標：相異突變集平均 ≥ 100
REASONABLE_RATIO = 1.2       # 抽樣平均 ≥ 目標的 1.2 倍 → 目標合理
REVISE_RATIO = 0.7           # 目標偏高時，修訂門檻 = 抽樣平均的 70% 取整
BEST_KNOWN = {"run02 drop_const（編碼）": 55, "run03 fade_frac=1.0（訓練設定）": 77}


def n_distinct(X):
    """0/1 矩陣 (n,L) 的相異列數。"""
    X = np.asarray(X)
    return int(len(np.unique(X, axis=0))) if len(X) else 0


def effective_n(w):
    """依出現次數加權的有效樣本數（Kish）：(Σw)² / Σw²。"""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def resample_distinct(X, w, n_draw, n_rep, seed):
    """依權重 w 有放回抽 n_draw 列，重複 n_rep 次，回傳每次的相異列數陣列。"""
    rng = np.random.default_rng(seed)
    p = np.asarray(w, float) / np.sum(w)
    # 先把相同的列合併成群，抽樣只需對群編號（相異列數 = 抽到的相異群編號數）
    _, gid = np.unique(np.asarray(X), axis=0, return_inverse=True)
    gp = np.bincount(gid.ravel(), weights=p)
    return np.array([len(np.unique(rng.choice(len(gp), size=n_draw, p=gp))) for _ in range(n_rep)])


def independent_distinct(freq, n_draw, n_rep, seed):
    """各位點依頻率 freq 獨立 Bernoulli 抽樣 n_draw 列，重複 n_rep 次的相異列數。"""
    rng = np.random.default_rng(seed)
    freq = np.asarray(freq, float)
    return np.array([n_distinct(rng.random((n_draw, len(freq))) < freq) for _ in range(n_rep)])


def verdict_of(resample_mean, target=TARGET):
    """判定規則（看結果前固定）：回傳 (verdict, revised_target)。"""
    if resample_mean >= REASONABLE_RATIO * target:
        return "reasonable", target
    if resample_mean < target:
        return "too_high", int(math.floor(REVISE_RATIO * resample_mean))
    return "undetermined", target


def _stats(a):
    """平均與範圍。"""
    return float(np.mean(a)), float(np.min(a)), float(np.max(a))


def analyze(X, count, split, n_draw=1000, n_rep=200, seed=0):
    """主流程：回傳 ceiling dict（欄位見 data-model.md）。抽樣分佈只用訓練期。"""
    X, count, split = np.asarray(X), np.asarray(count, float), np.asarray(split)
    tr, te = split == "train", split == "test"
    Xtr, wtr = X[tr], count[tr]
    freq = (Xtr * wtr[:, None]).sum(0) / wtr.sum()
    res = resample_distinct(Xtr, wtr, n_draw, n_rep, seed)
    ind = independent_distinct(freq, n_draw, n_rep, seed + 1)
    seen = {r.tobytes() for r in np.ascontiguousarray(Xtr)}
    Xte = np.unique(X[te], axis=0) if te.any() else X[te]
    novel = float(np.mean([r.tobytes() not in seen for r in np.ascontiguousarray(Xte)])) if len(Xte) else float("nan")
    rm, rlo, rhi = _stats(res)
    im, ilo, ihi = _stats(ind)
    verdict, revised = verdict_of(rm)
    return dict(n_draw=n_draw, n_rep=n_rep, seed=seed,
                train_unique=n_distinct(Xtr), train_effective_n=effective_n(wtr),
                resample_mean=rm, resample_min=rlo, resample_max=rhi,
                indep_mean=im, indep_min=ilo, indep_max=ihi,
                test_unique=n_distinct(X[te]), test_novel_frac=novel,
                verdict=verdict, target=TARGET, revised_target=revised,
                best_so_far={k: dict(value=v, ratio=v / rm) for k, v in BEST_KNOWN.items()})


def write_report(r, out_dir, encoding="drop_const", scale=1):
    """寫 ceiling.json 與 ceiling_report.md。"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "ceiling.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    word = {"reasonable": "目標合理", "too_high": "目標偏高", "undetermined": "無法判定（維持原目標）"}[r["verdict"]]
    lines = [f"# 資料多樣性上限（{encoding}，規模 {scale}）", "",
             f"口徑：編碼後 L 維 0/1 向量，相異向量＝相異突變集；每次抽 {r['n_draw']} 條、重複 {r['n_rep']} 次（種子 {r['seed']}）。",
             "抽樣分佈只用訓練期；測試期僅作對照。", "",
             "| 項目 | 數值 |", "|---|---|",
             f"| A 訓練期相異向量數 | {r['train_unique']}（有效樣本數 {r['train_effective_n']:.0f}） |",
             f"| B 真實分佈抽樣的相異數（上限） | {r['resample_mean']:.1f}［{r['resample_min']:.0f}–{r['resample_max']:.0f}］ |",
             f"| C 位點獨立抽樣的相異數 | {r['indep_mean']:.1f}［{r['indep_min']:.0f}–{r['indep_max']:.0f}］ |",
             f"| D 測試期相異向量數（未見於訓練期的比例） | {r['test_unique']}（{r['test_novel_frac']:.0%}） |", "",
             "## 判定", "",
             f"- 結論：**{word}**（B 平均 {r['resample_mean']:.1f}；目標 {r['target']}；規則：≥{REASONABLE_RATIO}×目標 合理、<目標 偏高）。",
             f"- 門檻：原 {r['target']} → 採用 **{r['revised_target']}**" +
             ("（修訂＝B 平均的 70% 取整，可達但不輕鬆）。" if r["verdict"] == "too_high" else "（不修訂）。"), "",
             "## 目前最佳結果占上限的比例", "", "| 結果 | 相異突變集 | 占 B 平均 |", "|---|---|---|"]
    lines += [f"| {k} | {v['value']} | {v['ratio']:.0%} |" for k, v in r["best_so_far"].items()]
    (out / "ceiling_report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=1); ap.add_argument("--encoding", default="drop_const")
    ap.add_argument("--n-draw", type=int, default=1000); ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args(argv)
    from Research.StyleGAN.evaluate.ablate import _ensure_dataset
    from Research.StyleGAN.FE.build_dataset import load_dataset
    if a.data_dir is None:
        _ensure_dataset(a.encoding)
    ds = load_dataset(a.scale, a.data_dir, a.encoding)
    r = analyze(ds["X"], ds["count"], ds["split"], a.n_draw, a.n_rep, a.seed)
    write_report(r, a.out, a.encoding, a.scale)
    print(f"完成 → {a.out}  判定={r['verdict']}  B={r['resample_mean']:.1f}  門檻={r['revised_target']}")


if __name__ == "__main__":
    main()
