"""訓練設定消融（003）：US1 判定多樣性下降是否為真實現象、逐因子消融、組合有效者、選最佳設定、在 nonorm 驗證。

與 ablate.py（以編碼為軸）不同，本檔以「訓練設定」為軸；指標與 run 讀取沿用 ablate.row_from_run。
已存在 metrics.json 的 run 不重訓；訓練發散的 run 標示失敗、不納入平均。

用法：python -m Research.StyleGAN.evaluate.ablate_recipe --stage {us1,round1,round2,confirm} --results Results/StyleGAN/run03
      （--part i n 可把 run 清單分成 n 份只跑第 i 份，供兩個程序並行；全部跑完後不帶 --part 再執行一次產生報告）
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from Research.StyleGAN.evaluate import ablate

BASE_RESULTS = "Results/StyleGAN/run02"      # 基準：run02 的 drop_const 3 種子（不重跑）
ENC = "drop_const"
US1_TAG, US1_EMA = "us1", 0.999
EFFECT_RATIO = 1.5                          # FR-006：相異突變集平均 ≥ 基準的 1.5 倍才算有效
PEAK_DROP = 0.5                             # US1：最終值 < 最高值的 50% 視為下降
SKIP_FRAC = 0.1                             # US1：扣除最後區塊前 10% 的檢查點（未訓練區間）

# 因子表（research R3）：每個因子 2 個非基準值；tag 會成為 run 目錄名稱的一部分
FACTORS = {
    "lr_scale": [("lr0.25", ["--lr-scale", "0.25"], 0.25), ("lr2", ["--lr-scale", "2.0"], 2.0)],
    "gamma1": [("r1_3", ["--gamma1", "3"], 3), ("r1_30", ["--gamma1", "30"], 30)],
    "d_every": [("dev2", ["--d-every", "2"], 2), ("dev4", ["--d-every", "4"], 4)],
    "last_block_steps": [("last600", ["--last-block-steps", "600"], 600), ("last2400", ["--last-block-steps", "2400"], 2400)],
    "fade_frac": [("fade0.2", ["--fade-frac", "0.2"], 0.2), ("fade1", ["--fade-frac", "1.0"], 1.0)],
    "ema_decay": [("ema0.99", ["--ema", "0.99"], 0.99), ("ema0.9999", ["--ema", "0.9999"], 0.9999)],
}


def run_dir_for_recipe(results, encoding, tag, seed, stage="A", scale=1, loss="logistic"):
    """與 Config.run_dir 相同的目錄規則（編碼後綴之後再加設定標籤）。"""
    enc = "" if encoding == "legacy" else f"_{encoding}"
    return Path(results) / stage / f"scale{scale}" / f"classical_{loss}{enc}{'_' + tag if tag else ''}_seed{seed}"


def run_one_recipe(encoding, tag, args, seed, results, device="auto", steps=3500, n=1000):
    """訓練 → 生成 → 評估一個（編碼、設定、種子）。已有 metrics.json 就沿用；發散則留下 status.json 並回傳。
    耗時（秒）寫入 recipe_meta.json。回傳 run 目錄。"""
    from Research.StyleGAN.evaluate.evaluate import evaluate
    from Research.StyleGAN.evaluate.generate import generate
    from Research.StyleGAN.train import train_StyleGAN as tc
    rd = run_dir_for_recipe(results, encoding, tag, seed)
    if (rd / "metrics.json").exists():
        return rd
    ablate._ensure_dataset(encoding)
    if not (rd / "checkpoint.pt").exists():
        t0 = time.time()
        try:
            tc.main(["--scale", "1", "--encoding", encoding, "--seed", str(seed), "--steps", str(steps),
                     "--results", str(results), "--wandb", "off", "--device", device, "--tag", tag, *args])
        except SystemExit as e:
            if e.code:
                print(f"⚠ 訓練失敗：{encoding} {tag} seed={seed}（見 {rd / 'status.json'}）")
                return rd
        (rd / "recipe_meta.json").write_text(json.dumps({"seconds": time.time() - t0}))
    generate(rd, n=n)
    evaluate(rd)
    return rd


def row_recipe(rd, stage, encoding, factor, value, seed):
    """一個 run 的消融列：沿用 ablate.row_from_run，另加因子、值、耗時。"""
    r = ablate.row_from_run(rd, stage, encoding, seed)
    meta = Path(rd) / "recipe_meta.json"
    r.update(factor=factor, value=str(value), seconds=json.loads(meta.read_text())["seconds"] if meta.exists() else np.nan)
    return r


def baseline_rows(base_results=BASE_RESULTS, encoding=ENC, seeds=(0, 1, 2)):
    """基準列：run02 的 encoding 3 種子（factor=baseline）。"""
    return [row_recipe(ablate.run_dir_for(base_results, encoding, s), "base", encoding, "baseline", "-", s) for s in seeds]


# ---------------- US1：下降是否為真實現象 ----------------
def _drop_verdict(series_by_seed):
    """每個種子取最後區塊（資料已限定）扣除前 10% 後的最高值與最終值：最終值 < 最高值 50% 為下降。回傳 (判定, 下降種子數)。"""
    n_real = 0
    for s in series_by_seed.values():
        v = np.asarray(s, float)
        v = v[int(np.ceil(SKIP_FRAC * len(v))):]
        if len(v) >= 2 and v[-1] < PEAK_DROP * v.max():
            n_real += 1
    return ("real" if n_real >= 2 else "artifact"), n_real


def us1_verdict(trajs):
    """US1 判定：trajs={種子: DataFrame(step, block, distinct_sets, live_distinct_sets)}。
    即時（live_distinct_sets）與 EMA（distinct_sets）各自判定；3 種子中至少 2 個下降 → real，否則 artifact。"""
    last = {s: d[d.block == d.block.max()] for s, d in trajs.items()}
    out = {}
    for name, col in (("ema", "distinct_sets"), ("live", "live_distinct_sets")):
        if all(col in d and d[col].notna().any() for d in last.values()):
            verdict, n = _drop_verdict({s: d[col].values for s, d in last.items()})
            out[name] = verdict
            out.setdefault("seeds_real", {})[name] = n
    return out


# ---------------- 判定與選擇 ----------------
def judge_recipe(df, base):
    """逐（因子、值）判定有效／無效／變差（FR-006）。
    有效：相異突變集平均 ≥ 基準 × 1.5，且 min_hamming_model 平均不比基準差超過基準 3 種子的標準差。
    變差：相異突變集平均與覆蓋平均都低於基準最小值。其餘無效。回傳 {(factor,value): 判定}。"""
    b_mean, b_std = base.min_hamming_model.mean(), base.min_hamming_model.std(ddof=0)
    out = {}
    for (f, v), d in df.groupby(["factor", "value"], sort=False):
        if d.distinct_mutation_sets.mean() >= EFFECT_RATIO * base.distinct_mutation_sets.mean() \
                and d.min_hamming_model.mean() <= b_mean + b_std:
            out[(f, v)] = "有效"
        elif d.distinct_mutation_sets.mean() < base.distinct_mutation_sets.min() \
                and d.position_coverage.mean() < base.position_coverage.min():
            out[(f, v)] = "變差"
        else:
            out[(f, v)] = "無效"
    return out


def select_best(df, distinct_target=None):
    """依固定規則選最佳設定（FR-006a）：先篩（相異突變集平均 ≥100、無種子 <30、覆蓋平均 ≥16），
    通過者取 min_hamming_model 平均最低；無人通過取相異突變集平均最高並標示未達標。
    distinct_target：相異突變集門檻（預設 100；004 上限偏高時傳入修訂門檻）。回傳 (factor, value, 是否達標)；df 空時回傳 None。"""
    thr = ablate.SC["distinct"] if distinct_target is None else distinct_target
    rows = []
    for (f, v), d in df.groupby(["factor", "value"], sort=False):
        ok = (d.distinct_mutation_sets.mean() >= thr and d.distinct_mutation_sets.min() >= ablate.SC["seed_min"]
              and d.position_coverage.mean() >= ablate.SC["coverage"])
        rows.append((f, v, ok, d.min_hamming_model.mean(), d.distinct_mutation_sets.mean()))
    if not rows:
        return None
    passed = [r for r in rows if r[2]]
    best = min(passed, key=lambda r: r[3]) if passed else max(rows, key=lambda r: r[4])
    return best[0], best[1], bool(passed)


def combo_plan(df, verdicts):
    """第 2 輪的組合：每個「有效」因子取相異突變集平均最高的有效值；有效因子少於 2 個時回傳 None。"""
    pick = {}
    for (f, v), j in verdicts.items():
        if j != "有效":
            continue
        d = df[(df.factor == f) & (df.value == v)].distinct_mutation_sets.mean()
        if f not in pick or d > pick[f][1]:
            pick[f] = (v, d)
    return {f: v for f, (v, _) in pick.items()} if len(pick) >= 2 else None


def args_for(factor, value):
    """因子與值（字串）對應的 (tag, 命令列參數)。"""
    for tag, args, v in FACTORS[factor]:
        if str(v) == str(value):
            return tag, args
    raise KeyError((factor, value))


def write_report_recipe(df, base, out_dir, stage, best=None, base_enc=ENC, verdicts=None, title="訓練設定消融報告"):
    """寫 ablation_summary.csv 與 ablation_report.md：逐因子逐值的指標、判定、SC 對照、失敗次數、最佳設定與是否達標。"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    allrows = pd.concat([base, df], ignore_index=True)
    allrows.to_csv(out / "ablation_summary.csv", index=False)
    if verdicts is None:      # 004 可傳入含「以失真換多樣性」覆寫的判定
        verdicts = judge_recipe(df, base) if len(df) else {}
    f1 = lambda d, c, p=1: f"{d[c].mean():.{p}f}［{d[c].min():.{p}f}–{d[c].max():.{p}f}］"
    L = [f"# {title}（{stage}）", "",
         f"基準＝run02 `{base_enc}` 3 種子；「有效」＝相異突變集平均 ≥ {EFFECT_RATIO:g}× 基準，且最小 Hamming（模型自決突變數）平均不比基準差超過 3 種子標準差。", "",
         "| 因子 | 值 | 種子數（失敗） | 相異突變集 | 位點覆蓋 | 頻率相關 | 平均突變數 | 最小 Hamming（模型 / 依頻率） | 耗時（秒） | 判定 |", "|---|---|---|---|---|---|---|---|---|---|"]
    def line(f, v, d, j, fail):
        """報告表格的一列。"""
        return (f"| {f} | {v} | {len(d)}（{fail}） | {f1(d, 'distinct_mutation_sets')} | {f1(d, 'position_coverage')} | "
                f"{f1(d, 'freq_correlation', 2)} | {f1(d, 'mean_mutation_count', 2)} | "
                f"{d.min_hamming_model.mean():.2f} / {d.min_hamming_freq_baseline.mean():.2f} | {'—' if d.seconds.isna().all() else format(d.seconds.mean(), '.0f')} | {j} |")
    L.append(line("baseline", "-", base, "基準", 0))
    for (f, v), d in df.groupby(["factor", "value"], sort=False):
        L.append(line(f, v, d, verdicts[(f, v)], max(0, 3 - len(d))))
    sc = ablate.success_table(allrows.assign(encoding=allrows.factor + "=" + allrows.value)) if len(df) else pd.DataFrame()
    if len(sc):
        L += ["", "## 成功標準對照（SC-002、003、003a、004）", "",
              "| 設定 | SC-002 | SC-003 | SC-003a | SC-004 | Hamming 差距（依頻率−模型） |", "|---|---|---|---|---|---|"]
        for _, r in sc.iterrows():
            if r.encoding == "baseline=-":
                continue
            m = lambda b: "✅" if b else "❌"
            f, v = r.encoding.split("=", 1)
            collapsed = allrows[(allrows.factor == f) & (allrows.value == v)].distinct_mutation_sets.mean() < 10
            sc004 = m(r.SC004) + ("（塌縮，不採計）" if collapsed and r.SC004 else "")   # 輸出幾乎是常數時，Hamming 看起來不差並不代表預測好
            L.append(f"| {r.encoding} | {m(r.SC002)} | {m(r.SC003)} | {m(r.SC003a)} | {sc004} | {r.gap_hamming:.2f} |")
    if best:
        L += ["", f"## 最佳設定（FR-006a）", "", f"- {best[0]} = {best[1]}；{'已通過篩選（SC-002、SC-003 達標）' if best[2] else '**未達標**：沒有設定通過篩選，取相異突變集最高者'}。"]
        if not best[2]:
            L.append("- **訓練設定並非唯一主因（SC-007）**。下一步可檢驗：模型結構、損失函數、判別器架構。")
    elif len(df):
        L += ["", "- 尚無可選設定。"]
    (out / "ablation_report.md").write_text("\n".join(L) + "\n")
    return verdicts, sc


# ---------------- 各階段 ----------------
def _plan(stage, results, seeds, factors, df_prev=None):
    """該階段要跑的 run 清單：[(encoding, tag, args, factor, value, seed)]。"""
    if stage == "us1":
        return [(ENC, US1_TAG, ["--ema", str(US1_EMA)], "us1", "ema0.999", s) for s in seeds]
    if stage == "round1":
        return [(ENC, tag, args, f, v, s) for f in (factors or FACTORS) for tag, args, v in FACTORS[f] for s in seeds]
    if stage == "round2":
        return [(ENC, "combo", sum((args_for(f, v)[1] for f, v in plan.items()), []),
                 "combo", "+".join(f"{f}={v}" for f, v in plan.items()), s) for plan in [df_prev] for s in seeds]
    raise ValueError(stage)


def main(argv=None):
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["us1", "round1", "round2", "confirm"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--factors", nargs="+", default=None, choices=list(FACTORS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--part", type=int, nargs=2, default=None, metavar=("I", "N"), help="只跑 run 清單的第 I 份（共 N 份，從 0 起）")
    a = ap.parse_args(argv)
    root = Path(a.results)
    if root.resolve() == Path(BASE_RESULTS).resolve():
        raise SystemExit("結果目錄不得是 run02（基準不可覆寫）")
    out = root / "A" / "ablation_recipe" / a.stage
    base = pd.DataFrame(baseline_rows())

    def execute(plan, enc=ENC):
        """依 --part 分份執行 run 清單，回傳已完成的 run。"""
        mine = [p for i, p in enumerate(plan) if a.part is None or i % a.part[1] == a.part[0]]
        for e, tag, args, f, v, s in mine:
            run_one_recipe(e, tag, args, s, root, a.device)
        return plan

    if a.stage == "us1":
        plan = execute(_plan("us1", root, a.seeds, None))
        trajs = {}
        for e, tag, args, f, v, s in plan:
            p = run_dir_for_recipe(root, e, tag, s) / "losses.csv"
            if p.exists():
                trajs[s] = pd.read_csv(p)
        if len(trajs) == len(a.seeds) and a.part is None:
            out.mkdir(parents=True, exist_ok=True)
            verdict = us1_verdict(trajs)
            (out / "us1_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
            from Research.StyleGAN.utils import plotting
            sch = json.loads((run_dir_for_recipe(root, ENC, US1_TAG, a.seeds[0]) / "config.json").read_text())["block_schedule"]
            plotting.trajectory_figure(trajs, sch[-1][0], out / "diversity_trajectory.png")
            print(verdict)
        return
    if a.stage == "confirm":
        bp = root / "A" / "ablation_recipe" / "best_recipe.json"
        if not bp.exists():
            raise SystemExit(f"找不到 {bp}：請先完成 round1（與 round2）以選出最佳設定")
        best = json.loads(bp.read_text())
        plan = [("nonorm", best["tag"], best["args"], best["factor"], best["value"], s) for s in a.seeds]
        execute(plan)
        if a.part is None:
            rows = [row_recipe(run_dir_for_recipe(root, "nonorm", best["tag"], s), "confirm", "nonorm", best["factor"], best["value"], s)
                    for s in a.seeds if (run_dir_for_recipe(root, "nonorm", best["tag"], s) / "metrics.json").exists()]
            nb = pd.DataFrame([row_recipe(ablate.run_dir_for(BASE_RESULTS, "nonorm", s), "base", "nonorm", "baseline", "-", s) for s in a.seeds])
            write_report_recipe(pd.DataFrame(rows), nb, out, "confirm（nonorm）", None, base_enc="nonorm")
        return
    df_prev = None
    if a.stage == "round2":
        r1 = pd.read_csv(root / "A" / "ablation_recipe" / "round1" / "ablation_summary.csv", dtype={"value": str})
        r1 = r1[r1.factor != "baseline"]
        df_prev = combo_plan(r1, judge_recipe(r1, base))
        if df_prev is None:
            print("第 1 輪有效的因子少於 2 個，沒有可組合的對象；第 2 輪略過（FR-007）。")
            return
    plan = execute(_plan(a.stage, root, a.seeds, a.factors, df_prev))
    if a.part is not None:
        return
    rows = []
    for e, tag, args, f, v, s in plan:
        rd = run_dir_for_recipe(root, e, tag, s)
        if (rd / "metrics.json").exists():
            rows.append(row_recipe(rd, a.stage, e, f, v, s))
    df = pd.DataFrame(rows)
    prev = df
    if a.stage == "round2":     # 選最佳設定時第 1 輪與第 2 輪一起比
        prev = pd.concat([pd.read_csv(root / "A" / "ablation_recipe" / "round1" / "ablation_summary.csv", dtype={"value": str}).query("factor != 'baseline'"), df])
    best = select_best(prev)
    verdicts, _ = write_report_recipe(df, base, out, a.stage, best)
    if best:
        f, v, _ok = best
        if f == "combo":
            args = sum((args_for(ff, vv)[1] for ff, vv in (x.split("=") for x in v.split("+"))), [])
            tag = "combo"
        else:
            tag, args = args_for(f, v)
        (root / "A" / "ablation_recipe" / "best_recipe.json").write_text(json.dumps(dict(factor=f, value=v, tag=tag, args=args, ok=_ok), ensure_ascii=False))
    print({f"{k[0]}={k[1]}": j for k, j in verdicts.items()}, "→", out)


if __name__ == "__main__":
    main()
