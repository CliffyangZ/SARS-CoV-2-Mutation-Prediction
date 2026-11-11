"""共用訓練迴圈（古典 / 量子判別器皆用，FR-015、FR-032）。

每步 D、G 交替更新；依 progressive.level_alpha 切換解析度；每 log_every 步記錄損失與梯度範數
（梯度範數用來偵測 barren plateau）；偵測 NaN 即中止並寫 status.json。
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from Research.FE.encoding import vector_to_rho
from Research.models import losses as L
from Research.models.progressive import level_alpha, real_at_level
from Research.utils.config import save_config, set_seed
from Research.utils.tracking import Tracker


def grad_norm(module):
    """模組所有參數梯度的 L2 範數。"""
    g = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
    return float(torch.cat(g).norm()) if g else 0.0


def write_status(run_dir, state, reason=""):
    """寫 status.json：created / training / finished / failed。"""
    (Path(run_dir) / "status.json").write_text(json.dumps({"state": state, "reason": reason}, ensure_ascii=False))


def train(cfg, G, D, X_train, run_dir, wandb_mode="offline", progress=False):
    """訓練 G、D。X_train: (n,L) 0/1 訓練期突變向量。回傳 status dict。失敗時不丟例外，只寫 status。"""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    save_config(cfg, run_dir / "config.json")
    write_status(run_dir, "training")
    tracker = Tracker(run_dir, {**cfg.__dict__, "block_schedule": cfg.block_schedule}, mode=wandb_mode,
                      name=run_dir.name)
    opt_g = torch.optim.Adam(G.param_groups(cfg.lr_g), betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)
    # 生成器 EMA：產生樣本與評估都用平滑後的權重（StyleGAN 慣例，降低震盪）
    G_ema = None
    if cfg.ema_decay > 0:
        G_ema = copy.deepcopy(G).requires_grad_(False)
    X = torch.as_tensor(X_train, dtype=torch.float32)
    rng = torch.Generator().manual_seed(cfg.seed)
    state, reason, t0 = "finished", "", time.time()

    for step in range(cfg.steps):
        level, alpha = level_alpha(cfg.block_schedule, step)
        side = cfg.block_sizes[level]   # ρ 跡為 1、元素約 1/邊長；真假 ρ 同乘邊長使量級 O(1)（本專案假設，R1 懲罰也在此尺度上算）
        real = real_at_level(vector_to_rho(X[torch.randint(len(X), (cfg.batch_size,), generator=rng)]), side, alpha) * side
        z = torch.randn(cfg.batch_size, cfg.latent_dim, generator=rng)

        # ---- 更新判別器 ----
        with torch.no_grad():
            fake = G(z, level, alpha)[:, None] * side
        real.requires_grad_(cfg.loss == "logistic" or cfg.r1_rahinge)
        d_real, d_fake = D(real, level, alpha), D(fake, level, alpha)
        if cfg.loss == "logistic":                      # 式 (4) ＋ 式 (5) R1 型梯度懲罰
            loss_d = L.d_loss_logistic(d_real, d_fake) + L.r1_penalty(d_real, real, cfg.gamma1)
        else:                                           # 式 (6)–(8)，可選加 R1 懲罰（式 5）
            loss_d = L.d_loss_rahinge(d_real, d_fake)
            if cfg.r1_rahinge:
                loss_d = loss_d + L.r1_penalty(d_real, real, cfg.gamma1)
        opt_d.zero_grad(); loss_d.backward(); gd = grad_norm(D); opt_d.step()

        # ---- 更新生成器 ----
        fake = G(z, level, alpha)[:, None] * side
        d_fake = D(fake, level, alpha)
        if cfg.loss == "logistic":                      # 式 (3)
            loss_g = L.g_loss_logistic(d_fake)
        else:                                           # 式 (9)–(10)
            with torch.no_grad():
                d_real = D(real.detach(), level, alpha)
            loss_g = L.g_loss_rahinge(d_real, d_fake)
        opt_g.zero_grad(); loss_g.backward(); gg = grad_norm(G); opt_g.step()
        if G_ema is not None:                           # EMA 衰減率前期線性升溫，避免早期平滑過頭
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, p in zip(G_ema.parameters(), G.parameters()):
                    pe.lerp_(p, 1 - decay)

        lg, ld = float(loss_g), float(loss_d)
        if not np.isfinite([lg, ld]).all():             # NaN / 發散偵測
            state, reason = "failed", f"NaN 或發散於第 {step} 步（loss_g={lg}, loss_d={ld}）"
            break
        if step % cfg.log_every == 0:
            tracker.log(step, level, alpha, lg, ld, grad_norm_g=gg, grad_norm_d=gd)
            if progress:
                print(f"step {step} level {level} a={alpha:.2f} Lg={lg:.3f} Ld={ld:.3f} gG={gg:.2e} gD={gd:.2e} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    ckpt = {"G": G.state_dict(), "D": D.state_dict()}
    if G_ema is not None:
        ckpt["G_ema"] = G_ema.state_dict()
    torch.save(ckpt, run_dir / "checkpoint.pt")
    write_status(run_dir, state, reason)
    tracker.close()
    return {"state": state, "reason": reason, "seconds": time.time() - t0, "wandb_url": tracker.url}
