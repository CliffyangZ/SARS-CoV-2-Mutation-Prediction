"""由檢查點生成突變結構並加上位點層級解釋（FR-016、FR-017、FR-029）。

用法：python -m Research.StyleGAN.evaluate.generate --run <run_dir> --n 1000 [--k K]
"""
import argparse

import numpy as np
import pandas as pd
import torch

from Research.StyleGAN.FE.build_dataset import load_dataset
from Research.StyleGAN.FE.encoding import generator_diag, readout
from Research.StyleGAN.FE.load_spike import to_int  # noqa: F401  （保持 FE 為資料處理入口）
from models.StyleGAN.generator import Generator
from Research.StyleGAN.utils.config import get_device, load_config, set_seed
from Research.StyleGAN.utils.sequence_mapping import map_to_sequences, train_alt_residues
from Research.StyleGAN.utils.spike_regions import region_of

AA = "ACDEFGHIKLMNPQRSTVWYX"


def decode(arr):
    """整數編碼 (n,1273) → 字串陣列。"""
    return np.array(["".join(AA[i] for i in row) for row in arr])


def sample_diag(G, cfg, n, seed, bs=50):
    """用最終解析度生成 n 個樣本，只保留各位點讀回量（l2 編碼為 sqrt(diag ρ)，sigmoid 編碼為對角線機率）。"""
    g = torch.Generator().manual_seed(seed)
    dev = next(G.parameters()).device
    out, level = [], len(cfg.block_sizes) - 1
    with torch.no_grad():
        for i in range(0, n, bs):
            z = torch.randn(min(bs, n - i), cfg.latent_dim, generator=g).to(dev)
            out.append(generator_diag(G(z, level, 1.0), cfg.enc).cpu())
    return torch.cat(out).numpy()


def topk_index(diag, k):
    """機率向量前 k 大的位置（L 座標，由大到小）。"""
    return np.argsort(-diag, axis=1)[:, :k]


def to_spike_pos(idx, spike_pos):
    """L 座標索引（-1 為填充）→ spike 位置（1 起算；填充為 0）。"""
    idx = np.asarray(idx)
    return np.where(idx >= 0, np.asarray(spike_pos)[np.clip(idx, 0, None)], 0)


def load_generator(cfg, ck):
    """依 config 的編碼組裝生成器並載入權重（有 EMA 權重就用 EMA）。"""
    G = Generator(cfg.block_sizes[-1], latent_size=cfg.latent_dim, dlatent_size=cfg.latent_dim,
                  mapping_layers=cfg.mapping_layers, fmap_base=cfg.fmap_base, fmap_max=cfg.g_fmap_max or cfg.fmap_max,
                  use_noise=cfg.noise, output_norm=cfg.enc.output_norm)
    G.load_state_dict(ck["G_ema"] if cfg.ema_decay > 0 and "G_ema" in ck else ck["G"]); G.eval()
    return G


def prepare_context(ds):
    """訓練期統計：1273 位點訓練期頻率、替代殘基、背景序列。"""
    tr = ds["split"] == "train"
    X_tr, w = ds["X_int"][tr], ds["count"][tr].astype(float)
    mut = (X_tr != ds["ref"]) & (X_tr != 20)
    return dict(X_tr=X_tr, train_freq=(mut * w[:, None]).sum(0) / w.sum(),
                alt=train_alt_residues(X_tr, ds["ref"], w))


def generate(run_dir, n=1000, k=None, seed=None, data_dir=None):
    """主流程；寫 generated.npz 與 explanations.csv，回傳 dict。"""
    from pathlib import Path
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    k = cfg.k if k is None else k
    seed = cfg.seed if seed is None else seed
    set_seed(seed)
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    G = load_generator(cfg, ck)
    G = G.to(get_device())
    ds = load_dataset(cfg.scale, data_dir, cfg.enc.name)
    ctx = prepare_context(ds)
    diag = sample_diag(G, cfg, n, seed)
    # 讀回突變集：legacy 固定取前 k 個（只為重現 run01）；其餘編碼由輸出閾值決定突變數（可為 0）
    idx, k_per_sample = readout(diag, cfg.enc, k)
    spike_pos = to_spike_pos(idx, ds["pmap"].spike_pos.values)
    mapped, existing = map_to_sequences(spike_pos, ctx["X_tr"], ds["ref"], ctx["alt"], np.random.default_rng(seed))
    np.savez_compressed(run_dir / "generated.npz", diag=diag, mut_index=idx, mut_spike_pos=spike_pos,
                        mapped_seq=decode(mapped), k=k, seed=seed, k_per_sample=k_per_sample)
    slope = ds["trend"].slope_per_month.values
    rows = [dict(gen_id=g, index=int(idx[g, j]), spike_pos=int(spike_pos[g, j]), region=region_of(int(spike_pos[g, j])),
                 train_freq=float(ctx["train_freq"][spike_pos[g, j] - 1]), trend_slope=float(slope[spike_pos[g, j] - 1]),
                 is_novel=bool(ctx["train_freq"][spike_pos[g, j] - 1] == 0),
                 count_matches_k=bool(len(set(idx[g][idx[g] >= 0].tolist())) == k_per_sample[g]))
            for g in range(n) for j in range(idx.shape[1]) if idx[g, j] >= 0]
    pd.DataFrame(rows, columns=["gen_id", "index", "spike_pos", "region", "train_freq", "trend_slope", "is_novel",
                                "count_matches_k"]).to_csv(run_dir / "explanations.csv", index=False)
    return dict(diag=diag, idx=idx, spike_pos=spike_pos, mapped=mapped, existing=existing, k=k,
                k_per_sample=k_per_sample)


def main():
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=None); ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    generate(a.run, a.n, a.k, a.seed)
    print("完成：", a.run)


if __name__ == "__main__":
    main()
