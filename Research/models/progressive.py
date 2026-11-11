"""漸進式訓練：區塊排程、淡入係數 α、降尺度（FR-013）。

古典降尺度為平均池化（論文）；量子降尺度（partial trace）於階段 B 加入（T062）。
"""
import torch
import torch.nn.functional as F


def level_alpha(block_schedule, step):
    """回傳 (當前區塊 level, 淡入係數 α)。
    每個區塊前半段 α 從 0 線性升到 1（淡入），後半段維持 1；第 0 個區塊 α=1。"""
    level = next((i for i, (a, b) in enumerate(block_schedule) if a <= step < b), len(block_schedule) - 1)
    start, end = block_schedule[level]
    if level == 0:
        return 0, 1.0
    half = max((end - start) / 2, 1)
    return level, float(min(1.0, (step - start) / half))


def downscale_classical(rho, factor):
    """古典降尺度：平均池化（rho: (B,1,S,S)）。"""
    return F.avg_pool2d(rho, factor) if factor > 1 else rho


def real_at_level(rho_full, side, alpha, pool=4):
    """把真實 ρ（B,L,L）降到當前解析度 side；淡入期與「較低解析度再放大」混合（同 ProGAN）。"""
    r = downscale_classical(rho_full[:, None], rho_full.shape[-1] // side)
    if alpha < 1.0 and side > 4:
        low = downscale_classical(r, pool)
        r = alpha * r + (1 - alpha) * F.interpolate(low, scale_factor=pool, mode="nearest")
    return r
