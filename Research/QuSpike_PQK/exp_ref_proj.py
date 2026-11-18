"""探索性實驗：(1) 換參考分類器（線性／GBM／blend）、(2) 換群內投影（LDA／LDA+PLS／PLS／PCA）。

**只用 validation**：資料經 ValidationOnlyView，測試陣列從未載入此流程，因此不消耗任何登錄版本的
「測試只評估一次」額度。結果只能用來決定「值不值得另開登錄版本（如 v2.1）」，不是論文數字。
設計參數取自已鎖定的 Doc/registry/pqk_v2/v2.0.json（唯讀）。

用法：python -m Research.QuSpike_PQK.exp_ref_proj --exp ref|proj --seeds 0 1 2
"""
import argparse
import json
import time
from pathlib import Path

from . import features as F, pipeline, registry, run

OUT = F.D.ROOT / "Results" / "quspike_pqk_refproj"
KINDS = ("pqk_v2", "rff_v2")  # 量子專家與其古典替身
REFS = ("linear", "gbm", "blend")
PROJS = (("lda", "zero"), ("lda", "pls"), ("pls", "zero"), ("pca", "zero"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["ref", "proj"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args(argv)
    design = json.loads((registry.DEFAULT_ROOT / "v2.0.json").read_text(encoding="utf-8"))["design"]
    fs = F.load_features(source="mutation")
    sub = registry.ValidationOnlyView(fs).as_featureset()
    del fs
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{a.exp}.jsonl"
    done = {(r["variant"], r["kind"], r["seed"]) for r in map(json.loads, log.read_text().splitlines())} \
        if log.exists() else set()
    ns = run.build_parser().parse_args([])
    variants = [(r, "lda", "pls") for r in REFS] if a.exp == "ref" else \
        [("linear", m, s) for m, s in PROJS]
    for seed in a.seeds:
        for ref, proj, sup in variants:
            for kind in KINDS:
                name = f"ref={ref}" if a.exp == "ref" else f"proj={proj}/{sup}"
                if (name, kind, seed) in done:
                    continue
                cfg = run._v2_cfg(ns, design)
                cfg.update(n_qubits=design["n_qubits"], trotter_steps=design["trotter_steps"],
                           evolution_time=design["evolution_time"], reference=ref, projection=proj)
                t = time.time()
                sc, _, lg = pipeline.tune_hyperparams(kind, True, True, sub, cfg, seed, OUT / "cache",
                                                      sup, cfg["routing"], cfg["soft_threshold"])
                f0 = next(x["val_f1_macro"] for x in lg if x["scale"] == 0.0)
                best = max(x["val_f1_macro"] for x in lg)
                rec = dict(variant=name, kind=kind, seed=seed, f1_reference_only=f0, f1_best=best,
                           gain=best - f0, scale=sc, seconds=round(time.time() - t, 1))
                with log.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)


if __name__ == "__main__":
    main()
