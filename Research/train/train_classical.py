"""階段 A：古典 StyleGAN（生成器 + 古典判別器）。本檔不得匯入 pennylane。

用法：python -m Research.train.train_classical --scale 1 --loss logistic --seed 0
"""
import argparse
import sys

import numpy as np

from Research.FE.build_dataset import PROCESSED
from Research.models.classical_discriminator import ClassicalDiscriminator
from Research.models.generator import Generator
from Research.train.loop import train
from Research.utils.config import make_config, set_seed


def build_models(cfg):
    """依設定組裝生成器與古典判別器。"""
    set_seed(cfg.seed)
    G = Generator(cfg.block_sizes, cfg.channels, cfg.latent_dim, cfg.mapping_layers, noise=cfg.noise)
    return G, ClassicalDiscriminator(cfg.channels)


def load_train_vectors(scale, data_dir=None):
    """讀資料集中的訓練期 0/1 突變向量。"""
    d = np.load((data_dir or PROCESSED / f"aa_scale{scale}") / "vectors.npz")
    return d["X"][d["split"] == "train"]


def main(argv=None):
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, choices=[1, 2], default=1)
    ap.add_argument("--loss", choices=["logistic", "rahinge"], default="logistic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--wandb", choices=["online", "offline", "off"], default="online")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--results", default="Results")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--ema", type=float, default=0.0, help="生成器 EMA 衰減率，0 為關閉，建議 0.999")
    ap.add_argument("--noise", action="store_true", help="啟用 noise injection")
    ap.add_argument("--r1-rahinge", action="store_true", help="rahinge 的 D 也加 R1 懲罰")
    a = ap.parse_args(argv)
    from pathlib import Path
    X = load_train_vectors(a.scale, Path(a.data_dir) if a.data_dir else None)
    cfg = make_config(a.scale, stage=a.stage, model="classical", loss=a.loss, seed=a.seed, steps=a.steps,
                      ema_decay=a.ema, noise=a.noise, r1_rahinge=a.r1_rahinge)
    G, D = build_models(cfg)
    st = train(cfg, G, D, X, cfg.run_dir(a.results), a.wandb, a.progress)
    print(f"{cfg.run_dir(a.results)} → {st['state']} {st['reason']} ({st['seconds']:.0f}s)")
    if st["state"] != "finished":
        print("訓練失敗：", st["reason"], file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
