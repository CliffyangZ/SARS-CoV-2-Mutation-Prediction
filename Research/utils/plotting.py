"""繪圖：保真度熱圖、損失曲線、中間結果圖（第一部分，T047）；圓形頻率圖（第二部分，T053）。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path):
    """存圖並關閉。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fidelity_heatmap(M, title, path):
    """10×10 保真度熱圖（列＝生成，欄＝資料集序列）。"""
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("dataset sequence"); ax.set_ylabel("generated sequence")
    ax.set_title(f"{title}\nmin {M.min():.3f} / mean {M.mean():.3f} / max {M.max():.3f}", fontsize=8)
    fig.colorbar(im, ax=ax); _save(fig, path)


def loss_curves(csv_paths, labels, path_prefix):
    """損失曲線（對應 Fig.2c–d / S2）：原始與對數（|loss|）兩版，G、D 各一欄。"""
    for tag, log in (("raw", False), ("log", True)):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
        for p, lab in zip(csv_paths, labels):
            df = pd.read_csv(p)
            for ax, col in zip(axs, ("loss_g", "loss_d")):
                ax.plot(df.step, df[col].abs() if log else df[col], label=lab, lw=1)
                ax.set_title(col + (" (log)" if log else "")); ax.set_xlabel("step")
                if log:
                    ax.set_yscale("log")
        axs[0].legend(fontsize=6); _save(fig, f"{path_prefix}_{tag}.png")


def intermediate_figure(x_real, diag, topk, path):
    """中間結果圖（FR-030）：真實突變向量、生成 sqrt(diag ρ)（機率向量）與 top-k 位置。"""
    fig, axs = plt.subplots(2, 1, figsize=(8, 3.6), sharex=True)
    axs[0].stem(np.where(x_real)[0], np.ones(int(x_real.sum())), basefmt=" "); axs[0].set_title("real mutation vector (example)")
    axs[1].bar(range(len(diag)), diag, color="#4c78a8")
    axs[1].scatter(topk, diag[topk], color="r", zorder=3, label="top-k")
    axs[1].set_title("generated sqrt(diag ρ) and top-k"); axs[1].legend(fontsize=7); axs[1].set_xlabel("position index")
    _save(fig, path)


# ---------------- 第二部分（T053）：圓形頻率圖 ----------------
def segment_freqs(freq, segments):
    """各位置頻率 (1273,) → 每段的平均突變頻率。"""
    return np.array([freq[a - 1:b].mean() for a, b in segments])


def peak_table(freqs, top=10):
    """freqs: {名稱: (1273,) 頻率} → 各組頻率最高的前 top 個位置（DataFrame）。"""
    rows = []
    for name, f in freqs.items():
        for pos in np.argsort(-f)[:top]:
            rows.append(dict(source=name, spike_pos=int(pos + 1), freq=float(f[pos])))
    return pd.DataFrame(rows)


def circular_frequency_plot(freqs, segments, regions, path):
    """圓形頻率圖（對應 Fig.4）：四段，三組（訓練期／測試期／生成）頻率疊圖，標示生成最高頻區段與功能區。
    freqs: {名稱: (1273,) 位置頻率}；以各位置頻率開根號（壓縮動態範圍）畫成極座標長條。"""
    fig = plt.figure(figsize=(6.5, 6.5)); ax = fig.add_subplot(projection="polar")
    n = max(b for _, b in segments)
    theta = (np.arange(n) + 0.5) / n * 2 * np.pi
    colors = {"train": "#4c78a8", "test": "#e45756", "generated": "#54a24b"}
    for name, f in freqs.items():
        ax.bar(theta, np.sqrt(np.asarray(f)), width=2 * np.pi / n, alpha=.55, color=colors.get(name, None), label=name)
    for a, b in segments:                                   # 四段邊界
        ax.plot([2 * np.pi * (a - 1) / n] * 2, [0, 1], color="k", lw=.6, ls="--")
    gen = segment_freqs(freqs["generated"], segments) if "generated" in freqs else None
    for nm, a, b in regions:                                # 功能區標示
        ax.text(2 * np.pi * ((a + b) / 2) / n, 1.05, nm, fontsize=6, ha="center")
    if gen is not None:
        top = int(gen.argmax()); ax.set_title(f"highest generated segment: {segments[top][0]}–{segments[top][1]}", fontsize=9)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.legend(fontsize=7, loc="lower left")
    _save(fig, path)
