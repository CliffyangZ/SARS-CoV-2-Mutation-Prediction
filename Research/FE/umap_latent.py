"""資料集 + 生成序列的 UMAP（Jaccard 距離，FR-030a，research R15）。

用法：python -m Research.FE.umap_latent --run <run_dir>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

from Research.FE.build_dataset import load_dataset
from Research.utils.config import load_config


def embed(X_data, X_gen, seed=0, n_neighbors=15):
    """合併資料與生成向量做 UMAP（metric=jaccard）；回傳 (n_data+n_gen, 2)。"""
    X = np.vstack([X_data, X_gen]).astype(bool)
    return umap.UMAP(metric="jaccard", n_neighbors=n_neighbors, random_state=seed).fit_transform(X)


def plot_umap(Z, n_data, labels, period, path):
    """兩張圖：以 lineage 代理標籤（生成標為「生成」）、以時間（訓練／測試／生成）著色。"""
    lab = np.concatenate([labels, ["generated"] * (len(Z) - n_data)])
    per = np.concatenate([period, ["generated"] * (len(Z) - n_data)])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, title in ((axs[0], lab, "lineage proxy"), (axs[1], per, "period")):
        for u in np.unique(key):
            m = key == u
            ax.scatter(Z[m, 0], Z[m, 1], s=14 if u == "generated" else 4, alpha=.8 if u == "generated" else .4,
                       label=u, marker="x" if u == "generated" else "o", lw=.5)
        ax.legend(fontsize=6, markerscale=1.5); ax.set_title(f"UMAP (Jaccard) — {title}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_umap(run_dir, out_path, max_data=4000, max_gen=500, seed=0, data_dir=None):
    """對一個 run 產 UMAP；資料取訓練期 + 測試期（子抽樣到 max_data 以控制時間）。"""
    cfg = load_config(Path(run_dir) / "config.json")
    ds = load_dataset(cfg.scale, data_dir)
    rng = np.random.default_rng(seed)
    keep = np.where(np.isin(ds["split"], ["train", "test"]))[0]
    keep = np.sort(rng.choice(keep, min(max_data, len(keep)), replace=False))
    g = np.load(Path(run_dir) / "generated.npz")
    gen_vec = np.zeros((min(max_gen, len(g["mut_index"])), ds["X"].shape[1]))
    np.put_along_axis(gen_vec, g["mut_index"][:len(gen_vec)], 1.0, axis=1)
    Z = embed(ds["X"][keep], gen_vec, seed)
    plot_umap(Z, len(keep), ds["lineage"][keep], ds["split"][keep], out_path)
    return Z


def main():
    """命令列入口。"""
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); run_umap(a.run, a.out); print("完成：", a.out)


if __name__ == "__main__":
    main()
