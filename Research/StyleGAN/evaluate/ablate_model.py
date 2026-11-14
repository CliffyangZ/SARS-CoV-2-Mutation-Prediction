"""模型結構與損失函數消融（004）：逐因子檢驗、組合有效者、在 nonorm 確認。

與 ablate_recipe（訓練設定）共用 run 執行、指標列與判定函式；本檔新增因子表、縮減／完整模式、
「以失真換多樣性」檢查與新舊兩套門檻欄位。模式預設依 ceiling.json 的判定決定（偏高 → 縮減）。

用法：python -m Research.StyleGAN.evaluate.ablate_model --stage {round1,round2,confirm} --results Results/StyleGAN/run04 [--mode shrunk|full]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from Research.StyleGAN.evaluate import ablate, ablate_recipe as ar

ENC = ar.ENC
COUNT_TOL = ablate.SC["count_tol"]          # 平均突變數與訓練期相差超過此值 → 失真
OLD_TARGET = ablate.SC["distinct"]

# 因子表（research R2）：每個因子的值依序為「縮減模式必跑」「完整模式另加」；方向於看結果前固定（spec Clarifications）
# 每項 (tag, 命令列參數, 值)
FACTORS = {
    "mapping_layers": [("map2", ["--mapping-layers", "2"], 2), ("map4", ["--mapping-layers", "4"], 4)],
    "noise": [("nonoise", ["--no-noise"], "off")],
    "g_fmap_max": [("g256", ["--g-fmap-max", "256"], 256), ("g128", ["--g-fmap-max", "128"], 128)],
    "latent_dim": [("lat32", ["--latent-dim", "32"], 32), ("lat16", ["--latent-dim", "16"], 16)],
    "mbstd_group": [("mbstd16", ["--mbstd-group", "16"], 16), ("mbstd8", ["--mbstd-group", "8"], 8)],
    "div_reg": [("div1", ["--div-reg", "1.0"], 1.0), ("div0.3", ["--div-reg", "0.3"], 0.3)],
    "d_fmap_max": [("d256", ["--d-fmap-max", "256"], 256), ("d128", ["--d-fmap-max", "128"], 128)],
}
STRUCTURE = ["mapping_layers", "noise", "g_fmap_max", "latent_dim"]
LOSS = ["mbstd_group", "div_reg", "d_fmap_max"]


def mode_from_ceiling(root):
    """依 root/ceiling/ceiling.json 的判定決定模式與修訂門檻：too_high → ('shrunk', 修訂門檻)，否則 ('full', 100)。
    找不到檔案時丟出明確錯誤（要求先跑上限分析）。"""
    p = Path(root) / "ceiling" / "ceiling.json"
    if not p.exists():
        raise SystemExit(f"找不到 {p}：請先執行 python -m Research.StyleGAN.evaluate.ceiling --out {p.parent}")
    c = json.loads(p.read_text())
    return ("shrunk" if c["verdict"] == "too_high" else "full"), int(c["revised_target"])


def factor_values(factor, mode):
    """該因子在模式下要跑的值清單：shrunk 只取第一個，full 取全部。"""
    return FACTORS[factor][:1] if mode == "shrunk" else FACTORS[factor]


def args_for(factor, value):
    """因子與值（字串）對應的 (tag, 命令列參數)。"""
    for tag, args, v in FACTORS[factor]:
        if str(v) == str(value):
            return tag, args
    raise KeyError((factor, value))


def plan_round1(mode, seeds, factors=None):
    """第 1 輪的 run 清單：[(encoding, tag, args, factor, value, seed)]。"""
    return [(ENC, tag, args, f, v, s) for f in (factors or FACTORS) for tag, args, v in factor_values(f, mode) for s in seeds]


def is_distorted(d):
    """平均突變數與訓練期平均相差超過容許值 → 以失真換多樣性（US3 情境 2）。"""
    return bool(abs(d.mean_mutation_count.mean() - d.train_mean_count.mean()) > COUNT_TOL)


def judge_model(df, base):
    """沿用 003 的判定，另把「相異突變集變多但平均突變數失真」的有效者降為「無效（以失真換多樣性）」。"""
    out = ar.judge_recipe(df, base)
    for (f, v), j in list(out.items()):
        d = df[(df.factor == f) & (df.value == v)]
        if j == "有效" and is_distorted(d):
            out[(f, v)] = "無效（以失真換多樣性）"
    return out


def pass_columns(df, new_target):
    """逐（因子、值）的達標欄位：pass_old（門檻 100）、pass_new（修訂門檻）、distorted。"""
    rows = []
    for (f, v), d in df.groupby(["factor", "value"], sort=False):
        ok = lambda t: bool(d.distinct_mutation_sets.mean() >= t and d.distinct_mutation_sets.min() >= ablate.SC["seed_min"])
        rows.append(dict(factor=f, value=v, pass_old=ok(OLD_TARGET), pass_new=ok(new_target), distorted=is_distorted(d)))
    return pd.DataFrame(rows)


def write_report_model(df, base, out_dir, stage, new_target, best=None, base_enc=ENC):
    """寫報告：沿用 003 的版面（標題改為模型與損失），再附上失真檢查與新舊門檻達標表。"""
    verdicts = judge_model(df, base) if len(df) else {}
    ar.write_report_recipe(df, base, out_dir, stage, best, base_enc, verdicts=verdicts, title="模型與損失消融報告")
    out = Path(out_dir)
    pc = pass_columns(df, new_target) if len(df) else pd.DataFrame()
    if len(pc):
        pc.to_csv(out / "pass_columns.csv", index=False)
        L = ["", "## 失真檢查與新舊門檻", "",
             f"舊門檻 {OLD_TARGET}；採用門檻 {new_target}（相異突變集平均達門檻且無種子 <{ablate.SC['seed_min']}）。"
             f"失真＝平均突變數與訓練期相差 >{COUNT_TOL:g}。", "",
             "| 因子 | 值 | 達舊門檻 | 達採用門檻 | 失真 |", "|---|---|---|---|---|"]
        m = lambda b: "✅" if b else "❌"
        L += [f"| {r.factor} | {r.value} | {m(r.pass_old)} | {m(r.pass_new)} | {'是' if r.distorted else '否'} |" for r in pc.itertuples()]
        with open(out / "ablation_report.md", "a", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")
    return verdicts


def main(argv=None):
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["round1", "round2", "confirm"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--mode", choices=["shrunk", "full"], default=None, help="預設依 ceiling.json 判定")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--factors", nargs="+", default=None, choices=list(FACTORS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--part", type=int, nargs=2, default=None, metavar=("I", "N"))
    a = ap.parse_args(argv)
    root = Path(a.results)
    for old in ("run02", "run03"):
        if root.resolve() == Path("Results/StyleGAN") .joinpath(old).resolve():
            raise SystemExit(f"結果目錄不得是 {old}（舊結果不可覆寫）")
    mode, new_target = mode_from_ceiling(root)
    mode = a.mode or mode
    out = root / "A" / "ablation_model" / a.stage
    best_path = root / "A" / "ablation_model" / "best_model_recipe.json"
    base = pd.DataFrame(ar.baseline_rows())

    def execute(plan):
        """依 --part 分份執行 run 清單。"""
        for i, (e, tag, args, f, v, s) in enumerate(plan):
            if a.part is None or i % a.part[1] == a.part[0]:
                ar.run_one_recipe(e, tag, args, s, root, a.device)

    def collect(plan, stage):
        """讀回已完成 run 的消融列。"""
        rows = []
        for e, tag, args, f, v, s in plan:
            rd = ar.run_dir_for_recipe(root, e, tag, s)
            if (rd / "metrics.json").exists():
                rows.append(ar.row_recipe(rd, stage, e, f, v, s))
        return pd.DataFrame(rows)

    if a.stage == "confirm":
        if not best_path.exists():
            raise SystemExit(f"找不到 {best_path}：請先完成 round1（與 round2）以選出最佳配方")
        best = json.loads(best_path.read_text())
        plan = [("nonorm", best["tag"], best["args"], best["factor"], best["value"], s) for s in a.seeds]
        execute(plan)
        if a.part is None:
            nb = pd.DataFrame([ar.row_recipe(ablate.run_dir_for(ar.BASE_RESULTS, "nonorm", s), "base", "nonorm", "baseline", "-", s) for s in a.seeds])
            write_report_model(collect(plan, "confirm"), nb, out, "confirm（nonorm）", new_target, None, base_enc="nonorm")
        return
    combo = None
    if a.stage == "round2":
        r1 = pd.read_csv(root / "A" / "ablation_model" / "round1" / "ablation_summary.csv", dtype={"value": str})
        r1 = r1[r1.factor != "baseline"]
        combo = ar.combo_plan(r1, judge_model(r1, base))
        if combo is None:
            print("第 1 輪有效的因子少於 2 個，沒有可組合的對象；第 2 輪略過（FR-009）。")
            return
        plan = [(ENC, "combo", sum((args_for(f, v)[1] for f, v in combo.items()), []), "combo",
                 "+".join(f"{f}={v}" for f, v in combo.items()), s) for s in a.seeds]
    else:
        plan = plan_round1(mode, a.seeds, a.factors)
    execute(plan)
    if a.part is not None:
        return
    df = collect(plan, a.stage)
    pool = df
    if a.stage == "round2":
        pool = pd.concat([r1, df])
    best = ar.select_best(pool, new_target) if len(pool) else None
    verdicts = write_report_model(df, base, out, a.stage, new_target, best)
    if best:
        f, v, ok = best
        if f == "combo":
            args = sum((args_for(ff, vv)[1] for ff, vv in (x.split("=") for x in v.split("+"))), [])
            tag = "combo"
        else:
            tag, args = args_for(f, v)
        best_path.write_text(json.dumps(dict(factor=f, value=v, tag=tag, args=args, ok=ok), ensure_ascii=False))
    print(f"模式={mode} 門檻={new_target}", {f"{k[0]}={k[1]}": j for k, j in verdicts.items()}, "→", out)


if __name__ == "__main__":
    main()
