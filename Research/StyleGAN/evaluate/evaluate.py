"""評估：保真度（式 1、2）、預測命中指標、多樣性、k 敏感度、編碼解讀敏感度（FR-018~021、R9、R10）。

階段 A 寫成後原封不動重用於量子實驗（FR-027）。
用法：python -m Research.StyleGAN.evaluate.evaluate --run <run_dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from Research.StyleGAN.FE.build_dataset import load_dataset
from Research.StyleGAN.FE.load_spike import accession_labels, accession_lookup
from Research.StyleGAN.evaluate.generate import prepare_context, to_spike_pos, topk_index
from Research.StyleGAN.utils import diversity as dv
from Research.StyleGAN.utils import plotting
from Research.StyleGAN.utils.baselines import all_baselines
from Research.StyleGAN.utils.config import load_config
from Research.StyleGAN.utils.fidelity import fidelity_matrix, paper_check, summarize
from Research.StyleGAN.utils.prediction_metrics import frequent_recall, min_hamming, novel_hit_rate
from Research.StyleGAN.utils.sequence_mapping import map_to_sequences

SOURCES = ["model", "reference", "random", "freq", "trend"]
K_SENS = [3, 6, 9]          # k 敏感度；k=9 取自測試期平均突變數，只作診斷（R3）


def khot(idx, L):
    """(n,k) 索引（-1 為填充）→ (n,L) 0/1 向量。"""
    v = np.zeros((len(idx), L))
    r, c = np.nonzero(np.asarray(idx) >= 0)
    v[r, np.asarray(idx)[r, c]] = 1.0
    return v


def truncate_to_k(idx, ks):
    """把 (n,kmax) 索引依各樣本的突變數 ks 截斷（多出的位置設為 -1），讓基準與模型有相同的突變數。"""
    idx = np.array(idx, copy=True)
    idx[np.arange(idx.shape[1])[None, :] >= np.asarray(ks)[:, None]] = -1
    return idx


def onehot_flat(X_int, n_cls=21):
    """整數編碼序列 → (n, 1273*21) one-hot（敏感度分析：編碼完整序列，R9）。"""
    out = np.zeros((len(X_int), X_int.shape[1] * n_cls), dtype=np.float32)
    np.put_along_axis(out.reshape(len(X_int), X_int.shape[1], n_cls), X_int[:, :, None].astype(int), 1.0, axis=2)
    return out


def map_source(name, idx, ds, ctx, seed):
    """來源 → 完整序列。reference 基準直接用參考序列（零突變）；其餘以訓練期背景＋替代殘基映射。"""
    if name == "reference":
        return np.tile(ds["ref"], (len(idx), 1))
    return map_to_sequences(to_spike_pos(idx, ds["pmap"].spike_pos.values), ctx["X_tr"], ds["ref"], ctx["alt"],
                            np.random.default_rng(seed))[0]


def reference_sets(ds, n_pick=10, seed=0):
    """三種資料集序列集合 → 各抽 n_pick 條（固定種子）：train / test / test_delta_like。回傳 {名: 索引}。"""
    rng = np.random.default_rng(seed)
    sp, lin = ds["split"], ds["lineage"]
    pools = {"train": np.where(sp == "train")[0], "test": np.where(sp == "test")[0],
             "test_delta_like": np.where((sp == "test") & (np.char.startswith(lin.astype(str), "Delta")))[0]}
    return {k: rng.choice(v, min(n_pick, len(v)), replace=False) for k, v in pools.items() if len(v)}


def labels_for_rows(ds, rows, lookup=None):
    """熱圖 Y 軸標籤：優先 unique_meta.acc，否則由 FASTA 對回代表登錄號。"""
    meta = ds.get("meta")
    if meta is not None and "acc" in meta.columns:
        return [str(a) for a in meta.acc.iloc[list(rows)].tolist()]
    return accession_labels(ds["X_int"][rows], lookup=lookup)


def source_vectors(gen, ds, k, seed, ctx, model_idx=None, ks=None):
    """各來源的位置索引 (n,k_max)（-1 為填充）。

    model_idx 給定時（模型自決突變數）用它當模型的突變集，並把基準截斷成相同的各樣本突變數 ks；
    否則模型取 diag 的前 k 大（k 敏感度與診斷用）。"""
    L, n = gen["diag"].shape[1], len(gen["diag"])
    pm = ds["pmap"]
    kmax = k if model_idx is None else model_idx.shape[1]
    idx = {"model": topk_index(gen["diag"], k) if model_idx is None else model_idx}
    base = all_baselines(n, kmax, L, seed, pm.train_freq.values, ds["trend"].pred_freq_test_start.values[pm.spike_pos.values - 1])
    idx.update({s: (truncate_to_k(v, ks) if ks is not None and s != "reference" else v) for s, v in base.items()})
    return idx


def outside_encoding(X_int_test, w, ref, spike_pos):
    """測試期突變中，落在編碼位點之外的比例（FR-001b）。回傳 dict：
    weighted＝以突變次數（依序列數加權）；distinct_positions＝以相異位點數。"""
    mut = (X_int_test != ref) & (X_int_test != 20)
    inside = np.zeros(mut.shape[1], bool); inside[np.asarray(spike_pos) - 1] = True
    events, out_events = (mut * w[:, None]).sum(), (mut[:, ~inside] * w[:, None]).sum()
    pos = mut.any(0)
    return dict(weighted=float(out_events / events) if events else 0.0,
                distinct_positions=float((pos & ~inside).sum() / pos.sum()) if pos.sum() else 0.0)


def evaluate(run_dir, data_dir=None, n_pick=10):
    """計算全部指標；寫 metrics.json 與圖；回傳 metrics。"""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    ds = load_dataset(cfg.scale, data_dir, cfg.enc.name)
    ctx = prepare_context(ds)
    g = np.load(run_dir / "generated.npz")
    gen = dict(diag=g["diag"])
    n, L, k = len(gen["diag"]), gen["diag"].shape[1], int(g["k"])
    # 各樣本突變數；舊檔（如 run01）沒有 k_per_sample 時視為固定 k，並在 metrics 標記
    fallback = "k_per_sample" not in g.files
    ks = np.full(n, k, dtype=int) if fallback else g["k_per_sample"]
    model_idx = g["mut_index"]
    pm, X_int, split = ds["pmap"], ds["X_int"], ds["split"]
    spike_idx = pm.spike_pos.values - 1
    figs = run_dir / "figs"
    metrics = {"fidelity": [], "prediction": [], "diversity": {}, "k_source_fallback": bool(fallback)}

    # ---------- 保真度（模型用生成的突變集；基準取相同的各樣本突變數） ----------
    idx = source_vectors(gen, ds, k, cfg.seed, ctx, model_idx, ks)
    sets = reference_sets(ds, n_pick, seed=0)
    mapped = {s: map_source(s, ix, ds, ctx, cfg.seed) for s, ix in idx.items()}
    # 論文圖 10-12：Y 軸為資料集登錄號；舊資料集無 acc 欄時以 FASTA 對回
    need_fasta = ds.get("meta") is None or "acc" not in getattr(ds.get("meta"), "columns", [])
    acc_lut = accession_lookup() if need_fasta else None
    for rs, rows in sets.items():
        real_vec, real_int = ds["X"][rows].astype(float), X_int[rows]
        y_labels = labels_for_rows(ds, rows, lookup=acc_lut)
        for s in SOURCES:
            vecs = {"topk_binary": khot(idx[s][:n_pick], L)}
            vecs["literal"] = gen["diag"][:n_pick] if s == "model" else vecs["topk_binary"]
            for interp, A in vecs.items():
                for f in ("eq1", "eq2"):
                    M = fidelity_matrix(A, real_vec, f); st = summarize(M)
                    thr, ok = paper_check(rs, f, s, interp, st["mean"])
                    metrics["fidelity"].append(dict(reference_set=rs, formula=f, source=s, interpretation=interp,
                                                    paper_threshold=thr, passed=ok, **st))
                    # 突變向量保真度熱圖（字面解讀）；色階依資料範圍，對齊論文輸出形式
                    if interp == "literal" and s == "model":
                        plotting.fidelity_heatmap(
                            M, f"{rs} {f} model (mutation)", figs / f"fidelity_{rs}_{f}.png",
                            y_labels=y_labels)
            # 編碼完整序列（one-hot）的解讀：映射後序列 vs 真實序列（數值區間接近論文圖 10-12）
            A = onehot_flat(mapped[s][:n_pick]); B = onehot_flat(real_int)
            for f in ("eq1", "eq2"):
                M = fidelity_matrix(A, B, f)
                metrics["fidelity"].append(dict(reference_set=rs, formula=f, source=s, interpretation="fullseq_onehot",
                                                paper_threshold=None, passed=None, **summarize(M)))
                if s == "model":
                    plotting.fidelity_heatmap(
                        M, f"{rs} {f} model (fullseq one-hot)", figs / f"fidelity_{rs}_{f}_fullseq.png",
                        y_labels=y_labels)

    # ---------- 預測命中指標 ----------
    # 三種 k 來源：model＝模型自行決定的突變數（正式結果）；test_mean_diagnostic＝測試期平均突變數（診斷，
    # 用到測試期資訊，不列入達標判定）；topk_sweep＝固定 k 的敏感度。（FR-003）
    te = split == "test"
    w_te = ds["count"][te].astype(float)
    mut_te = (X_int[te] != ds["ref"]) & (X_int[te] != 20)
    test_freq = (mut_te * w_te[:, None]).sum(0) / w_te.sum()
    test_mutated = mut_te.any(0)
    k_test = max(1, int(round(float((ds["X"][te] > 0).sum(1).mean()))))   # 編碼位點上的測試期平均突變數

    def add_pred(kind, idk, kval, diagnostic):
        """計算某一組各來源的預測指標並加入 metrics。"""
        for s in SOURCES:
            sp_pos = to_spike_pos(idk[s], pm.spike_pos.values)
            dmin = min_hamming(map_source(s, idk[s], ds, ctx, cfg.seed), X_int[te])
            has = bool((sp_pos > 0).any())
            metrics["prediction"].append(dict(
                source=s, k=kval, k_source=kind, diagnostic=diagnostic, reference_set="test",
                min_hamming_mean=float(dmin.mean()), min_hamming_median=float(np.median(dmin)),
                novel_hit_rate=novel_hit_rate(sp_pos, ctx["train_freq"], test_mutated) if has else None,
                frequent_recall=frequent_recall(sp_pos, test_freq, pm.spike_pos.values) if has else None))

    add_pred("model", idx, float(ks.mean()), False)
    for kk in K_SENS:
        add_pred("topk_sweep", source_vectors(gen, ds, kk, cfg.seed, ctx), kk, kk == 9)
    add_pred("test_mean_diagnostic", source_vectors(gen, ds, k_test, cfg.seed, ctx), k_test, True)

    # ---------- 多樣性（FR-001、FR-001a、FR-001b、FR-002） ----------
    ms = g["mapped_seq"]
    tr_freq_L = pm.train_freq.values
    rng = np.random.default_rng(0)
    pick = lambda m: np.where(split == m)[0][rng.permutation((split == m).sum())[:n]]
    div = dv.report(model_idx, L, tr_freq_L)
    div.update(distinct_sequences=int(len(set(ms.tolist()))), n=n, k=k)
    ref_sets = {"train": ds["X"][pick("train")], "test": ds["X"][pick("test")]}
    div["reference"] = {name: dv.report(dv.vectors_to_idx(x), L, tr_freq_L) for name, x in ref_sets.items()}
    for name in ("random", "freq"):
        div["reference"][name] = dv.report(idx[name], L, tr_freq_L)
    div["mutation_count_gap_vs_test"] = div["mean_mutation_count"] - div["reference"]["test"]["mean_mutation_count"]
    div["test_outside_encoding_frac"] = outside_encoding(X_int[te], w_te, ds["ref"], pm.spike_pos.values)
    metrics["diversity"] = div
    (run_dir / "diversity.json").write_text(json.dumps(div, indent=1, ensure_ascii=False))

    # ---------- 圖 ----------
    cand = np.where((split == "train") & (ds["X"].sum(1) >= max(cfg.k, 1)))[0]
    ex = cand[0] if len(cand) else np.where(split == "train")[0][0]   # 取一條至少有 k 個突變的訓練序列
    plotting.intermediate_figure(ds["X"][ex], gen["diag"][0], g["mut_index"][0][g["mut_index"][0] >= 0], figs / "intermediate.png")
    plotting.loss_curves([run_dir / "losses.csv"], [run_dir.name], str(figs / "loss"))
    plotting.diversity_curve(run_dir / "losses.csv", [a for a, _ in cfg.block_schedule[1:]], figs / "diversity_curve.png")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    return metrics


def main():
    """命令列入口。塌縮時 stdout 第一行先印警示（結束碼仍為 0）。"""
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    a = ap.parse_args()
    m = evaluate(a.run)
    d = m["diversity"]
    if d["collapse_warning"]:
        print(f"⚠ 疑似模式塌縮：{d['n']} 條生成只有 {d['distinct_mutation_sets']} 種突變集、"
              f"覆蓋 {d['position_coverage']} 個位點（門檻 {d['warning_threshold']}）")
    print("完成：", a.run, {k: d[k] for k in ("distinct_mutation_sets", "position_coverage", "freq_correlation")})


if __name__ == "__main__":
    main()
