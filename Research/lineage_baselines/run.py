"""命令列入口：選擇 baseline 與種子，訓練、評估並寫結果檔（contracts/cli.md）。

用法範例：
    python -m Research.lineage_baselines.run --group classical --representation ae_embedding --seeds 0 1 2 3 4
    python -m Research.lineage_baselines.run --model resnet34 --representation rp --seeds 0 1 2
"""
import argparse
import sys
import time
from pathlib import Path

from . import data as D
from . import results_io as R

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "Results" / "lineage_baselines"

CLASSICAL = ["svm", "nb", "mlp", "knn", "rf", "lr", "dt"]
IMAGE = ["cnn3", "cnn4", "resnet34"]
SIGNAL = ["fcn", "lstm3bi", "mwdn"]
GROUPS = {"classical": CLASSICAL, "image": IMAGE, "signal": SIGNAL}
DEFAULT_REP = {"classical": "ae_embedding", "image": "rp", "signal": "signal"}
VALID_REP = {
    "classical": {"onehot", "ae_embedding"},
    "image": {"rp", "gaf", "mtf"},
    "signal": {"signal"},
}
DETERMINISTIC_ON_ONEHOT = {"knn", "nb", "svm"}  # 這些模型本身沒有隨機性；但輸入若是自編碼器嵌入，嵌入隨種子而變


def is_deterministic(model, rep):
    """結果是否與種子無關：模型本身決定性，且輸入為 one-hot（自編碼器嵌入會隨種子改變）。"""
    return model in DETERMINISTIC_ON_ONEHOT and rep == "onehot"


def group_of(model):
    """回傳模型所屬組別。"""
    for g, ms in GROUPS.items():
        if model in ms:
            return g
    raise ValueError(f"未知模型：{model}")


def build_parser():
    """建立命令列參數（對應 contracts/cli.md）。"""
    p = argparse.ArgumentParser(description="重現論文 baseline")
    p.add_argument("--data", default=str(D.DEFAULT_DATA))
    p.add_argument("--group", choices=sorted(GROUPS))
    p.add_argument("--model", choices=sorted(sum(GROUPS.values(), [])))
    p.add_argument("--representation")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit", type=int, default=None, help="只取 N 條序列（冒煙與測試用）")
    p.add_argument("--ae-epochs", type=int, default=None, help="覆寫自編碼器輪數（預設 100；測試用）")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="ResNet34 載入 ImageNet 預訓練權重（預設開；使用者已同意下載）")
    p.add_argument("--signal-mode", choices=["raw", "cumsum"], default="raw",
                   help="深度模型的訊號：raw（預設）或 cumsum（累積後重新縮放）；請搭配不同的 --out，避免與 raw 結果混在一起")
    p.add_argument("--stop-after", type=int, default=None, help="訓練到第 N 輪就停（量測單輪耗時或測試用）")
    return p


def plan_runs(args):
    """展開成 (model, representation, seed) 清單；決定性模型只保留第一個種子。"""
    if bool(args.group) == bool(args.model):
        raise SystemExit("請擇一指定 --group 或 --model")
    models = GROUPS[args.group] if args.group else [args.model]
    runs = []
    for m in models:
        g = group_of(m)
        rep = args.representation or DEFAULT_REP[g]
        if rep not in VALID_REP[g]:
            raise SystemExit(f"{m} 不支援表示 {rep}，可用：{sorted(VALID_REP[g])}")
        seeds = args.seeds if args.seeds is not None else (list(range(5)) if g == "classical" else list(range(3)))
        if is_deterministic(m, rep):
            seeds = seeds[:1]
        runs += [(m, rep, s) for s in seeds]
    return runs


def _runner_for(group):
    """延遲匯入各組別的執行函式，避免不需要的相依被載入。"""
    if group == "classical":
        from . import classical

        return classical.run_one
    from . import deep_runner

    return deep_runner.run_one


def main(argv=None):
    """主流程：載入資料、逐一執行並寫結果。"""
    args = build_parser().parse_args(argv)
    runs = plan_runs(args)
    try:
        prep = D.load_prepared(args.data, limit=args.limit)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    out = Path(args.out)
    failures = []
    ctx = {}  # 同一次執行內共用的快取（例如每個種子的自編碼器嵌入）
    for model, rep, seed in runs:
        path = R.result_path(out, model, rep, seed)
        deep = group_of(model) != "classical"
        if deep and path.exists():
            prev = R.read_signal_mode(path)
            if prev != args.signal_mode:
                print(f"{path.name} 已存在且訊號模式為 {prev}，與 --signal-mode {args.signal_mode} 不同；請改用另一個 --out", file=sys.stderr)
                return 2
        if R.is_complete(path, min_version=2 if deep else None):
            print(f"跳過（已完成）：{path.name}")
            continue
        t0 = time.time()
        try:
            res = _runner_for(group_of(model))(model, rep, seed, prep, args, ctx)
        except Exception as e:  # 單一 run 失敗不拖垮整批：記錄後繼續，最後以非零碼結束
            failures.append((model, rep, seed, repr(e)))
            print(f"失敗：{model}/{rep}/seed{seed}：{e!r}", file=sys.stderr, flush=True)
            continue
        res.update(
            model=model,
            representation=rep,
            seed=seed,
            deterministic=is_deterministic(model, rep),
            preprocessing=prep.preprocessing_info(),
            environment=R.environment_info(),
        )
        res.setdefault("status", "complete")
        res.setdefault("train_seconds", time.time() - t0)
        R.write_result(path, res)
        m = res["metrics"]
        print(f"{model}/{rep}/seed{seed}: acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} 耗時={res['train_seconds']:.1f}s")
    if failures:
        print(f"共 {len(failures)} 個 run 失敗：", *[f"{m}/{r}/seed{s}" for m, r, s, _ in failures], sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
