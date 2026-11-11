"""評估：保真度（式 1、2）、預測命中指標、多樣性、k 敏感度、編碼解讀敏感度（FR-018~021、R9、R10）。

階段 A 寫成後原封不動重用於量子實驗（FR-027）。
用法：python -m Research.train.evaluate --run <run_dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from Research.FE.build_dataset import load_dataset
from Research.train.generate import prepare_context, topk_index
from Research.utils import plotting
from Research.utils.baselines import all_baselines
from Research.utils.config import load_config
from Research.utils.fidelity import fidelity_matrix, paper_check, summarize
from Research.utils.prediction_metrics import frequent_recall, min_hamming, novel_hit_rate
from Research.utils.sequence_mapping import map_to_sequences

SOURCES = ["model", "reference", "random", "freq", "trend"]
K_SENS = [3, 6, 9]          # k 敏感度；k=9 取自測試期平均突變數，只作診斷（R3）


def khot(idx, L):
    """(n,k) 索引 → (n,L) 0/1 向量。"""
    v = np.zeros((len(idx), L)); np.put_along_axis(v, idx, 1.0, axis=1) if idx.shape[1] else None
    return v


def onehot_flat(X_int, n_cls=21):
    """整數編碼序列 → (n, 1273*21) one-hot（敏感度分析：編碼完整序列，R9）。"""
    out = np.zeros((len(X_int), X_int.shape[1] * n_cls), dtype=np.float32)
    np.put_along_axis(out.reshape(len(X_int), X_int.shape[1], n_cls), X_int[:, :, None].astype(int), 1.0, axis=2)
    return out


def map_source(name, idx, ds, ctx, seed):
    """來源 → 完整序列。reference 基準直接用參考序列（零突變）；其餘以訓練期背景＋替代殘基映射。"""
    if name == "reference":
        return np.tile(ds["ref"], (len(idx), 1))
    return map_to_sequences(ds["pmap"].spike_pos.values[idx], ctx["X_tr"], ds["ref"], ctx["alt"],
                            np.random.default_rng(seed))[0]


def reference_sets(ds, n_pick=10, seed=0):
    """三種資料集序列集合 → 各抽 n_pick 條（固定種子）：train / test / test_delta_like。回傳 {名: 索引}。"""
    rng = np.random.default_rng(seed)
    sp, lin = ds["split"], ds["lineage"]
    pools = {"train": np.where(sp == "train")[0], "test": np.where(sp == "test")[0],
             "test_delta_like": np.where((sp == "test") & (np.char.startswith(lin.astype(str), "Delta")))[0]}
    return {k: rng.choice(v, min(n_pick, len(v)), replace=False) for k, v in pools.items() if len(v)}


def source_vectors(gen, ds, k, seed, ctx):
    """各來源在 k 下的 (a) 0/1 或連續向量、(b) 位置索引 (n,k)。model 的 literal 向量為 sqrt(diag ρ)。"""
    L, n = gen["diag"].shape[1], len(gen["diag"])
    pm = ds["pmap"]
    idx = {"model": topk_index(gen["diag"], k)}
    idx.update(all_baselines(n, k, L, seed, pm.train_freq.values, ds["trend"].pred_freq_test_start.values[pm.spike_pos.values - 1]))
    return idx


def evaluate(run_dir, data_dir=None, n_pick=10):
    """計算全部指標；寫 metrics.json 與圖；回傳 metrics。"""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    ds = load_dataset(cfg.scale, data_dir)
    ctx = prepare_context(ds)
    g = np.load(run_dir / "generated.npz")
    gen = dict(diag=g["diag"])
    n, L, k = len(gen["diag"]), gen["diag"].shape[1], int(g["k"])
    pm, X_int, split = ds["pmap"], ds["X_int"], ds["split"]
    spike_idx = pm.spike_pos.values - 1
    figs = run_dir / "figs"
    metrics = {"fidelity": [], "prediction": [], "diversity": {}}

    # ---------- 保真度 ----------
    idx = source_vectors(gen, ds, k, cfg.seed, ctx)
    sets = reference_sets(ds, n_pick, seed=0)
    mapped = {s: map_source(s, ix, ds, ctx, cfg.seed) for s, ix in idx.items()}
    for rs, rows in sets.items():
        real_vec, real_int = ds["X"][rows].astype(float), X_int[rows]
        for s in SOURCES:
            vecs = {"topk_binary": khot(idx[s][:n_pick], L)}
            vecs["literal"] = gen["diag"][:n_pick] if s == "model" else vecs["topk_binary"]
            for interp, A in vecs.items():
                for f in ("eq1", "eq2"):
                    M = fidelity_matrix(A, real_vec, f); st = summarize(M)
                    thr, ok = paper_check(rs, f, s, interp, st["mean"])
                    metrics["fidelity"].append(dict(reference_set=rs, formula=f, source=s, interpretation=interp,
                                                    paper_threshold=thr, passed=ok, **st))
                    if interp == "literal" and f == "eq1" and s == "model":
                        plotting.fidelity_heatmap(M, f"{rs} eq1 model", figs / f"fidelity_{rs}_eq1.png")
                    if interp == "literal" and f == "eq2" and s == "model":
                        plotting.fidelity_heatmap(M, f"{rs} eq2 model", figs / f"fidelity_{rs}_eq2.png")
            # 編碼完整序列（one-hot）的解讀：映射後序列 vs 真實序列
            A = onehot_flat(mapped[s][:n_pick]); B = onehot_flat(real_int)
            for f in ("eq1", "eq2"):
                metrics["fidelity"].append(dict(reference_set=rs, formula=f, source=s, interpretation="fullseq_onehot",
                                                paper_threshold=None, passed=None, **summarize(fidelity_matrix(A, B, f))))

    # ---------- 預測命中指標（k 敏感度） ----------
    te = split == "test"
    w_te = ds["count"][te].astype(float)
    mut_te = (X_int[te] != ds["ref"]) & (X_int[te] != 20)
    test_freq = (mut_te * w_te[:, None]).sum(0) / w_te.sum()
    test_mutated = mut_te.any(0)
    for kk in sorted({k, *K_SENS}):
        idk = source_vectors(gen, ds, kk, cfg.seed, ctx)
        for s in SOURCES:
            sp_pos = pm.spike_pos.values[idk[s]]
            seqs = map_source(s, idk[s], ds, ctx, cfg.seed)
            dmin = min_hamming(seqs, X_int[te])
            metrics["prediction"].append(dict(
                source=s, k=kk, diagnostic=bool(kk == 9), reference_set="test",
                min_hamming_mean=float(dmin.mean()), min_hamming_median=float(np.median(dmin)),
                novel_hit_rate=novel_hit_rate(sp_pos, ctx["train_freq"], test_mutated) if sp_pos.size else None,
                frequent_recall=frequent_recall(sp_pos, test_freq, pm.spike_pos.values) if sp_pos.size else None))

    # ---------- 多樣性 ----------
    ms = g["mapped_seq"]
    metrics["diversity"] = dict(distinct_sequences=int(len(set(ms.tolist()))),
                                distinct_mutation_sets=int(len({tuple(sorted(r)) for r in g["mut_index"].tolist()})),
                                position_coverage=int(len(np.unique(g["mut_index"]))), n=n, k=k)

    # ---------- 圖 ----------
    ex = np.where((split == "train") & (ds["X"].sum(1) >= cfg.k))[0][0]   # 取一條至少有 k 個突變的訓練序列
    plotting.intermediate_figure(ds["X"][ex], gen["diag"][0], g["mut_index"][0], figs / "intermediate.png")
    plotting.loss_curves([run_dir / "losses.csv"], [run_dir.name], str(figs / "loss"))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    return metrics


def main():
    """命令列入口。"""
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    a = ap.parse_args()
    m = evaluate(a.run)
    print("完成：", a.run, m["diversity"])


if __name__ == "__main__":
    main()
