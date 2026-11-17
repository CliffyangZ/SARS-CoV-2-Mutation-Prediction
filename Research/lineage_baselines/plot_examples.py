"""畫出同一條序列的 RP、GAF、MTF（raw 與 cumsum 兩種訊號），用來檢查影像轉換。

用法：
    python -m Research.lineage_baselines.plot_examples --lineages AY.4 B.1.526 --out Results/lineage_baselines_v2/figures/rp_gaf_mtf.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import transforms as T


def plot(prep, lineages, out, index=0):
    """每個譜系取第 index 條訓練序列，畫 訊號＋RP＋GAF＋MTF；raw 與 cumsum 各一列。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(l, m) for l in lineages for m in T.SIGNAL_MODES]
    fig, ax = plt.subplots(len(rows), 4, figsize=(16, 3.8 * len(rows)))
    for r, (lin, mode) in enumerate(rows):
        k = prep.labels.index(lin)
        i = int(np.where(np.asarray(prep.y_train) == k)[0][index])
        sig = T.apply_signal_mode(T.signal_from_ids(torch.from_numpy(prep.x_train[i : i + 1])), mode)
        ax[r, 0].plot(sig[0].numpy(), lw=0.6)
        ax[r, 0].set_title(f"{lin} | signal ({mode}), L={sig.shape[1]}")
        for c, (name, fn) in enumerate([("RP", T.rp_matrix), ("GAF", T.gaf_matrix), ("MTF", T.mtf_matrix)], 1):
            ax[r, c].imshow(fn(sig)[0], cmap="PuBuGn_r")
            ax[r, c].set_title(f"{name} ({mode})")
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=80)
    plt.close(fig)
    return out


def paper_cmap():
    """仿論文 Figure 2 的色階：粉白 → 淡藍 → 青綠 → 深橄欖。"""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("fig2", ["#ffdcff", "#b4b4ff", "#2e8b7a", "#3a3a00"])


def plot_paper(prep, lineages, out, index=0, mode="cumsum"):
    """論文 Figure 2 版面：每列一個譜系，三欄 RP / GAF / MTF，座標為原始序列位置。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fns = [("RP", T.rp_matrix), ("GAF", T.gaf_matrix), ("MTF", T.mtf_matrix)]
    cmap = paper_cmap()
    fig, ax = plt.subplots(len(lineages), 3, figsize=(12, 4.6 * len(lineages)), facecolor="black")
    ax = np.atleast_2d(ax)
    letters = "abcdefghijkl"
    for r, lin in enumerate(lineages):
        k = prep.labels.index(lin)
        i = int(np.where(np.asarray(prep.y_train) == k)[0][index])
        sig = T.apply_signal_mode(T.signal_from_ids(torch.from_numpy(prep.x_train[i : i + 1])), mode)
        for c, (name, fn) in enumerate(fns):
            a = ax[r, c]
            a.set_facecolor("black")
            a.imshow(fn(sig)[0].numpy(), cmap=cmap, interpolation="nearest")
            a.tick_params(colors="white", labelsize=6)
            a.set_xlabel(f"({letters[r * 3 + c]}) {name}", color="white", fontsize=13, labelpad=14)
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100, facecolor="black")
    plt.close(fig)
    return out


def main(argv=None):
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(D.DEFAULT_DATA))
    ap.add_argument("--lineages", nargs="+", default=["AY.4", "B.1.526"])
    ap.add_argument("--index", type=int, default=0, help="該譜系的第幾條訓練序列")
    ap.add_argument("--out", default="Results/lineage_baselines_v2/figures/rp_gaf_mtf.png")
    ap.add_argument("--paper", action="store_true", help="輸出論文 Figure 2 版面（2 譜系 × RP/GAF/MTF）")
    ap.add_argument("--mode", default="cumsum", choices=T.SIGNAL_MODES, help="--paper 時使用的訊號模式")
    a = ap.parse_args(argv)
    if a.paper:
        out = a.out if a.out != ap.get_default("out") else "Results/lineage_baselines_v2/figures/fig2_paper_style.png"
        print(plot_paper(D.load_prepared(a.data), a.lineages, out, a.index, a.mode))
        return 0
    print(plot(D.load_prepared(a.data), a.lineages, a.out, a.index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
