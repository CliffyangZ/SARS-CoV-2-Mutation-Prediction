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


def fidelity_heatmap(M, title, path, y_labels=None, x_labels=None, cmap="jet", scale="data"):
    """保真度熱圖（對齊論文圖 10-12）：列＝資料集序列（登錄號），欄＝生成樣本 1…n。

    參數
    ----
    M : (n_gen, n_data)
        ``fidelity_matrix(generated, real)`` 的結果；繪圖時轉置。
    y_labels : 長度 n_data 的登錄號（可選）。
    scale : ``data``＝依矩陣 min/max 著色（凸顯高保真度窄區間）；``unit``＝固定 0–1。
    """
    H = np.asarray(M, dtype=float).T                      # (n_data, n_gen)
    n_data, n_gen = H.shape
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    if scale == "unit":
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(H)), float(np.nanmax(H))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = (0.0, 1.0) if not np.isfinite(vmin) else (vmin, vmin + 1e-6)
    im = ax.imshow(H, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n_gen))
    ax.set_xticklabels(list(x_labels) if x_labels is not None else [str(i + 1) for i in range(n_gen)])
    ax.set_yticks(range(n_data))
    ax.set_yticklabels(list(y_labels) if y_labels is not None else [str(i + 1) for i in range(n_data)],
                       fontsize=7)
    ax.set_xlabel("generated sample")
    ax.set_ylabel("dataset sequence")
    ax.set_title(f"{title}\nmin {H.min():.4f} / mean {H.mean():.4f} / max {H.max():.4f}", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)
    # 同步輸出數值矩陣（列＝資料集登錄號，欄＝生成樣本），方便對照論文圖
    csv_path = Path(path).with_suffix(".csv")
    cols = list(x_labels) if x_labels is not None else [str(i + 1) for i in range(n_gen)]
    idx = list(y_labels) if y_labels is not None else [str(i + 1) for i in range(n_data)]
    pd.DataFrame(H, index=idx, columns=cols).to_csv(csv_path, float_format="%.6f")


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


def diversity_curve(csv_path, block_steps, path):
    """多樣性曲線（002、003）：相異突變集、位點覆蓋、頻率相關隨步數變化；虛線為漸進式區塊切換的步數。
    losses.csv 有 live_* 欄位（訓練時啟用 EMA）時，EMA 權重畫實線、即時權重畫虛線，
    並以灰色陰影標示最後區塊的前 10%（尚未訓練的高解析度區塊，數值僅供參考）。"""
    df = pd.read_csv(csv_path)
    cols = [c for c in ("distinct_sets", "pos_coverage", "freq_corr") if c in df and df[c].notna().any()]
    if not cols:
        return False
    fig, axs = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3), squeeze=False)
    for ax, c in zip(axs[0], cols):
        ax.plot(df.step, df[c], lw=1, label="EMA" if "live_" + c in df and df["live_" + c].notna().any() else None)
        if "live_" + c in df and df["live_" + c].notna().any():
            ax.plot(df.step, df["live_" + c], lw=1, ls="--", label="live")
            ax.legend(fontsize=6)
        for b in block_steps:
            ax.axvline(b, color="gray", ls=":", lw=0.8)
        if len(block_steps):
            last = block_steps[-1]
            ax.axvspan(last, last + 0.1 * max(df.step.max() - last, 0), color="gray", alpha=0.15)
        ax.set_title(c, fontsize=9); ax.set_xlabel("step")
    _save(fig, path)
    return True


def trajectory_figure(trajs, block_start, path):
    """US1 的多樣性軌跡圖：每個種子一格，EMA（實線）與即時權重（虛線）的相異突變集隨步數變化。
    trajs: {種子: DataFrame(step, distinct_sets, live_distinct_sets)}；block_start：最後區塊的起點步數。"""
    fig, axs = plt.subplots(1, len(trajs), figsize=(3.6 * len(trajs), 3), squeeze=False, sharey=True)
    for ax, (seed, df) in zip(axs[0], trajs.items()):
        ax.plot(df.step, df.distinct_sets, lw=1, label="EMA")
        if "live_distinct_sets" in df and df.live_distinct_sets.notna().any():
            ax.plot(df.step, df.live_distinct_sets, lw=1, ls="--", label="live")
        ax.axvline(block_start, color="gray", ls=":", lw=0.8)
        ax.axvspan(block_start, block_start + 0.1 * (df.step.max() - block_start), color="gray", alpha=0.15)
        ax.set_title(f"seed {seed}", fontsize=9); ax.set_xlabel("step")
    axs[0][0].set_ylabel("distinct sets"); axs[0][0].legend(fontsize=6)
    _save(fig, path)


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
