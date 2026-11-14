"""跨實驗圖表：圓形頻率圖、UMAP、峰值位置表（T053–T055，FR-020、FR-030a）。

用法：python -m Research.StyleGAN.evaluate.figures --stage A [--scale 1]
輸出到 Results/<階段>/figures/，並寫 peaks.md。
"""
import argparse
from pathlib import Path

import numpy as np

from Research.StyleGAN.FE.build_dataset import load_dataset
from Research.StyleGAN.FE.umap_latent import run_umap
from Research.StyleGAN.utils import plotting
from Research.StyleGAN.utils.config import load_config
from Research.StyleGAN.utils.spike_regions import REGIONS, SEGMENTS, region_of

KNOWN = {614: "D614G", 452: "L452R", 681: "P681H", 478: "T478K", 484: "E484K", 501: "N501Y", 570: "A570D"}


def position_freqs(ds, gen_pos, n):
    """回傳 {train, test, generated}：各 spike 位置（1273）的突變頻率。"""
    out = {}
    for name in ("train", "test"):
        m = ds["split"] == name
        mut = (ds["X_int"][m] != ds["ref"]) & (ds["X_int"][m] != 20)
        w = ds["count"][m].astype(float)
        out[name] = (mut * w[:, None]).sum(0) / w.sum()
    g = np.zeros(1273)
    np.add.at(g, gen_pos.ravel() - 1, 1.0)
    out["generated"] = g / n
    return out


def make_figures(stage, scale=1, results="Results", umap=True):
    """對該階段所有 run 產圖並寫 peaks.md；回傳 peaks.md 路徑。"""
    runs = sorted(Path(results, stage, f"scale{scale}").glob("*_seed*"))
    runs = [r for r in runs if (r / "generated.npz").exists()]
    fig_dir = Path(results, stage, "figures"); fig_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(scale)
    lines = [f"# 階段 {stage} 突變頻率峰值（規模 {scale}）", "",
             "訓練期快速上升突變（D614G、L452R、P681H、T478K、E484K 等）與測試期頻繁位點對照。",
             "圖：`circular_<run>.png`（四段 1–318／319–636／637–954／955–1273，疊上訓練期、測試期、生成序列頻率）。", ""]
    test_top = np.argsort(-position_freqs(ds, np.array([[1]]), 1)["test"])[:10] + 1
    lines += [f"測試期頻率最高的 10 個位置：{', '.join(f'{p}({region_of(int(p))})' for p in test_top)}", ""]
    for r in runs:
        g = np.load(r / "generated.npz")
        fr = position_freqs(ds, g["mut_spike_pos"], len(g["mut_index"]))
        plotting.circular_frequency_plot(fr, SEGMENTS, REGIONS, fig_dir / f"circular_{r.name}.png")
        seg = plotting.segment_freqs(fr["generated"], SEGMENTS).argmax()
        top = plotting.peak_table({"generated": fr["generated"]}, 5)
        hits = [f"{p}={KNOWN[p]}" for p in top.spike_pos if p in KNOWN]
        lines += [f"## {r.name}", f"- 生成序列最高頻區段：{SEGMENTS[seg][0]}–{SEGMENTS[seg][1]}",
                  f"- 生成頻率最高的 5 個位置：{', '.join(f'{p}({region_of(int(p))})' for p in top.spike_pos)}",
                  f"- 其中對應已知快速上升突變：{', '.join(hits) or '無'}",
                  f"- 這些位置的訓練期頻率：{', '.join('%.3f' % fr['train'][p - 1] for p in top.spike_pos)}", ""]
        if umap:
            run_umap(r, fig_dir / f"umap_{r.name}.png")
    p = fig_dir / "peaks.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def main():
    """命令列入口。"""
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="A"); ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--no-umap", action="store_true")
    a = ap.parse_args(); print("完成：", make_figures(a.stage, a.scale, umap=not a.no_umap))


if __name__ == "__main__":
    main()
