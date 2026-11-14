"""StyleGAN（古典判別器）訓練腳本：模型組裝、訓練迴圈、命令列入口，皆在本檔。

每步 D、G 交替更新；依 progressive.level_alpha 切換解析度；每 log_every 步記錄損失與梯度範數
（梯度範數用來偵測 barren plateau）；偵測 NaN 即中止並寫 status.json。本檔不得匯入 pennylane。
一個模型一個訓練腳本；量子版另開 train_<Model>.py。

用法：python -m Research.StyleGAN.train.train_StyleGAN --scale 1 --loss logistic --seed 0 [--encoding legacy|drop_const|nonorm|freq_order|組合]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from Research.StyleGAN.FE.build_dataset import dataset_dir
from Research.StyleGAN.FE.encoding import encode_real, fake_image, generator_diag, readout, real_scale
from models.StyleGAN import losses as L, update_average
from models.StyleGAN.classical_discriminator import Discriminator
from models.StyleGAN.generator import Generator
from models.StyleGAN.progressive import level_alpha, real_at_level
from Research.StyleGAN.utils.config import get_device, make_config, save_config, set_seed
from Research.StyleGAN.utils import diversity as dv
from Research.StyleGAN.utils.tracking import Tracker


def grad_norm(module):
    """模組所有參數梯度的 L2 範數。"""
    g = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
    return float(torch.cat(g).norm()) if g else 0.0


def write_status(run_dir, state, reason=""):
    """寫 status.json：created / training / finished / failed。"""
    (Path(run_dir) / "status.json").write_text(json.dumps({"state": state, "reason": reason}, ensure_ascii=False))


N_DIV = 200   # 每個記錄點用多少個固定潛碼量測多樣性


def _diversity_of(m, cfg, z, train_freq):
    """用固定潛碼在最終解析度讓生成器 m 生成，讀回突變集並算多樣性三指標。"""
    was_training = m.training
    m.eval()
    devs = [z.device] if z.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devs):   # 量測不能改變訓練用的亂數序列（noise injection 會用到）
        diag = np.concatenate([generator_diag(m(zc, len(cfg.block_sizes) - 1, 1.0), cfg.enc).cpu().numpy()
                               for zc in z.split(25)])   # 分批生成：一次 200 個在最高解析度會用掉數 GB 顯存
    m.train(was_training)
    idx, _ = readout(diag, cfg.enc, cfg.k)
    corr = dv.freq_correlation(idx, train_freq)
    return dict(distinct_sets=dv.distinct_sets(idx), pos_coverage=dv.position_coverage(idx),
                freq_corr=None if np.isnan(corr) else corr)


def diversity_now(G, G_ema, cfg, z, train_freq):
    """多樣性三指標（FR-009）。有 EMA 時主欄位為 EMA 權重，另以 live_* 附上即時權重（003 FR-001）；
    沒有 EMA 時只有即時權重（主欄位）。進入最後一個區塊前，最高解析度區塊尚未訓練，數值僅供參考。"""
    if G_ema is None:
        return _diversity_of(G, cfg, z, train_freq)
    out = _diversity_of(G_ema, cfg, z, train_freq)
    out.update({"live_" + k: v for k, v in _diversity_of(G, cfg, z, train_freq).items()})
    return out


def train(cfg, G, D, X_train, run_dir, wandb_mode="offline", progress=False, train_freq=None):
    """訓練 G、D。X_train: (n,L) 0/1 訓練期突變向量。回傳 status dict。失敗時不丟例外，只寫 status。
    train_freq：各位點訓練期頻率（算頻率相關用）；省略時取 X_train 的平均。"""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = next(G.parameters()).device   # 裝置跟著模型走（train_StyleGAN 已把 G、D 搬到 GPU）
    enc = cfg.enc                          # 本次實驗的編碼（寫入、讀回、輸出正規化皆由它決定）
    save_config(cfg, run_dir / "config.json")
    write_status(run_dir, "training")
    tracker = Tracker(run_dir, {**cfg.__dict__, "block_schedule": cfg.block_schedule}, mode=wandb_mode,
                      name=run_dir.name)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g * cfg.lr_scale, betas=cfg.betas)   # mapping 的 0.01 學習率乘數由 GMapping 的 lrmul 負責
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d * cfg.lr_scale, betas=cfg.betas)
    # 生成器 EMA：產生樣本與評估都用平滑後的權重（StyleGAN 慣例，降低震盪）
    G_ema = None
    if cfg.ema_decay > 0:
        G_ema = copy.deepcopy(G).requires_grad_(False)
    X = torch.as_tensor(X_train, dtype=torch.float32)   # 留在 CPU，每步只把 batch 搬上裝置
    rng = torch.Generator().manual_seed(cfg.seed)
    state, reason, t0 = "finished", "", time.time()
    tf = np.asarray(X_train, float).mean(0) if train_freq is None else np.asarray(train_freq, float)
    z_div = torch.randn(N_DIV, cfg.latent_dim, generator=torch.Generator().manual_seed(cfg.seed + 12345)).to(device)

    for step in range(cfg.total_steps):
        level, alpha = level_alpha(cfg.block_schedule, step, cfg.fade_frac)
        side = cfg.block_sizes[level]   # ρ 跡為 1、元素約 1/邊長；真假 ρ 同乘邊長使量級 O(1)（本專案假設，R1 懲罰也在此尺度上算）
        xb = X[torch.randint(len(X), (cfg.batch_size,), generator=rng)].to(device)
        real = real_at_level(encode_real(xb, enc), side, alpha, mode=enc.pool_mode) * real_scale(enc, side)
        z = torch.randn(cfg.batch_size, cfg.latent_dim, generator=rng).to(device)   # CPU generator 產生，結果與 CPU 訓練一致

        # ---- 更新判別器 ----
        with torch.no_grad():
            fake = fake_image(G(z, level, alpha), enc, side)
        real.requires_grad_(cfg.loss == "logistic" or cfg.r1_rahinge)
        d_real, d_fake = D(real, level, alpha), D(fake, level, alpha)
        if cfg.loss == "logistic":                      # 式 (4) ＋ 式 (5) R1 型梯度懲罰
            loss_d = L.d_loss_logistic(d_real, d_fake) + L.r1_penalty(d_real, real, cfg.gamma1)
        else:                                           # 式 (6)–(8)，可選加 R1 懲罰（式 5）
            loss_d = L.d_loss_rahinge(d_real, d_fake)
            if cfg.r1_rahinge:
                loss_d = loss_d + L.r1_penalty(d_real, real, cfg.gamma1)
        opt_d.zero_grad(); loss_d.backward(); gd = grad_norm(D)
        if step % cfg.d_every == 0:                     # 003：d_every>1 時降低判別器更新頻率（損失仍每步計算以便記錄）
            opt_d.step()

        # ---- 更新生成器 ----
        fake = fake_image(G(z, level, alpha), enc, side)
        d_fake = D(fake, level, alpha)
        if cfg.loss == "logistic":                      # 式 (3)
            loss_g = L.g_loss_logistic(d_fake)
        else:                                           # 式 (9)–(10)
            with torch.no_grad():
                d_real = D(real.detach(), level, alpha)
            loss_g = L.g_loss_rahinge(d_real, d_fake)
        if cfg.div_reg > 0:                             # 004：多樣性正則——另取一組 z 生成第二批（預設關閉，不多做前向）
            z2 = torch.randn(cfg.batch_size, cfg.latent_dim, generator=rng).to(device)
            fake2 = fake_image(G(z2, level, alpha), enc, side)
            loss_g = loss_g + L.diversity_reg(fake, fake2, z, z2, cfg.div_reg)
        opt_g.zero_grad(); loss_g.backward(); gg = grad_norm(G); opt_g.step()
        if G_ema is not None:                           # EMA 衰減率前期線性升溫，避免早期平滑過頭
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            update_average(G_ema, G, decay)

        lg, ld = float(loss_g), float(loss_d)
        if not np.isfinite([lg, ld]).all():             # NaN / 發散偵測
            state, reason = "failed", f"NaN 或發散於第 {step} 步（loss_g={lg}, loss_d={ld}）"
            break
        if step % cfg.log_every == 0:
            div = diversity_now(G, G_ema, cfg, z_div, tf)
            tracker.log(step, level, alpha, lg, ld, grad_norm_g=gg, grad_norm_d=gd, **div)
            warn, thr = dv.collapse_warning(div["distinct_sets"], N_DIV)
            if progress:
                print(f"step {step} level {level} a={alpha:.2f} Lg={lg:.3f} Ld={ld:.3f} gG={gg:.2e} gD={gd:.2e} "
                      f"sets={div['distinct_sets']} cov={div['pos_coverage']} ({time.time()-t0:.0f}s)"
                      + ("  ⚠ 疑似塌縮" if warn and level == len(cfg.block_sizes) - 1 else ""), flush=True)

    ckpt = {"G": {k: v.cpu() for k, v in G.state_dict().items()}, "D": {k: v.cpu() for k, v in D.state_dict().items()}}
    if G_ema is not None:
        ckpt["G_ema"] = {k: v.cpu() for k, v in G_ema.state_dict().items()}
    torch.save(ckpt, run_dir / "checkpoint.pt")
    write_status(run_dir, state, reason)
    tracker.close()
    return {"state": state, "reason": reason, "seconds": time.time() - t0, "wandb_url": tracker.url}


def build_models(cfg):
    """依設定組裝生成器與古典判別器。"""
    set_seed(cfg.seed)
    res = cfg.block_sizes[-1]
    G = Generator(res, latent_size=cfg.latent_dim, dlatent_size=cfg.latent_dim, mapping_layers=cfg.mapping_layers,
                  fmap_base=cfg.fmap_base, fmap_max=cfg.g_fmap_max or cfg.fmap_max, use_noise=cfg.noise,
                  output_norm=cfg.enc.output_norm)
    return G, Discriminator(res, fmap_base=cfg.fmap_base, fmap_max=cfg.d_fmap_max or cfg.fmap_max,
                            mbstd_group_size=cfg.mbstd_group)


def load_train_vectors(scale, data_dir=None, encoding="legacy"):
    """讀資料集中的訓練期 0/1 突變向量（依編碼找資料夾）。"""
    d = np.load((data_dir or dataset_dir(scale, encoding)) / "vectors.npz")
    return d["X"][d["split"] == "train"]


def main(argv=None):
    """命令列入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, choices=[1, 2], default=1)
    ap.add_argument("--loss", choices=["logistic", "rahinge"], default="logistic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--wandb", choices=["online", "offline", "off"], default="online")
    ap.add_argument("--device", default="auto", help="auto（有 GPU 就用）| cuda | cuda:N | cpu")
    ap.add_argument("--encoding", default="legacy", help="legacy | drop_const | nonorm | freq_order | 以 + 組合")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--results", default="Results")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--ema", type=float, default=0.0, help="生成器 EMA 衰減率，0 為關閉，建議 0.999")
    ap.add_argument("--no-noise", action="store_true", help="關閉 noise injection（預設開啟，同 reference）")
    ap.add_argument("--r1-rahinge", action="store_true", help="rahinge 的 D 也加 R1 懲罰")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="G、D 學習率同乘此值")
    ap.add_argument("--gamma1", type=float, default=10.0, help="R1 權重")
    ap.add_argument("--d-every", type=int, default=1, help="每 n 步才更新一次判別器")
    ap.add_argument("--last-block-steps", type=int, default=None, help="最後區塊的訓練步數（其餘區塊不動）")
    ap.add_argument("--fade-frac", type=float, default=0.5, help="淡入長度占區塊長度的比例")
    ap.add_argument("--tag", default="", help="run 目錄後綴（設定標籤）")
    ap.add_argument("--mapping-layers", type=int, default=8, help="mapping 網路層數")
    ap.add_argument("--latent-dim", type=int, default=64, help="潛在向量維度")
    ap.add_argument("--g-fmap-max", type=int, default=None, help="生成器特徵圖寬度上限（預設同 512）")
    ap.add_argument("--d-fmap-max", type=int, default=None, help="判別器特徵圖寬度上限（預設同 512）")
    ap.add_argument("--mbstd-group", type=int, default=4, help="判別器批次統計（minibatch 標準差）群組大小")
    ap.add_argument("--div-reg", type=float, default=0.0, help="多樣性正則權重（0＝關閉）")
    a = ap.parse_args(argv)
    X = load_train_vectors(a.scale, Path(a.data_dir) if a.data_dir else None, a.encoding)
    extra = {} if a.encoding == "legacy" else {"k": int(round(float(X.sum(1).mean())))}   # k＝該編碼訓練期平均突變數（只用訓練期）
    cfg = make_config(a.scale, encoding=a.encoding, stage=a.stage, model="classical", loss=a.loss, seed=a.seed,
                      steps=a.steps, ema_decay=a.ema, noise=not a.no_noise, r1_rahinge=a.r1_rahinge,
                      lr_scale=a.lr_scale, gamma1=a.gamma1, d_every=a.d_every, last_block_steps=a.last_block_steps,
                      fade_frac=a.fade_frac, tag=a.tag, mapping_layers=a.mapping_layers, latent_dim=a.latent_dim,
                      g_fmap_max=a.g_fmap_max, d_fmap_max=a.d_fmap_max, mbstd_group=a.mbstd_group,
                      div_reg=a.div_reg, **extra)
    device = get_device(a.device)
    G, D = build_models(cfg)
    G, D = G.to(device), D.to(device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
    import pandas as pd
    pm = dataset_dir(a.scale, a.encoding) / "position_map.csv" if not a.data_dir else Path(a.data_dir) / "position_map.csv"
    tf = pd.read_csv(pm).train_freq.values if pm.exists() else None   # 加權訓練期頻率，與評估一致
    st = train(cfg, G, D, X, cfg.run_dir(a.results), a.wandb, a.progress, train_freq=tf)
    print(f"{cfg.run_dir(a.results)} → {st['state']} {st['reason']} ({st['seconds']:.0f}s)")
    if st["state"] != "finished":
        print("訓練失敗：", st["reason"], file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
