"""漸進式訓練：區塊排程、淡入係數 α、降尺度（FR-013）。

古典降尺度為平均池化（論文）；量子降尺度（partial trace）於階段 B 加入（T062）。
"""
import torch
import torch.nn.functional as F


def level_alpha(block_schedule, step, fade_frac=0.5):
    """回傳 (當前區塊 level, 淡入係數 α)。
    每個區塊的前 fade_frac（預設一半）α 從 0 線性升到 1（淡入），其餘維持 1；第 0 個區塊 α=1。"""
    level = next((i for i, (a, b) in enumerate(block_schedule) if a <= step < b), len(block_schedule) - 1)
    start, end = block_schedule[level]
    if level == 0:
        return 0, 1.0
    half = max((end - start) * fade_frac, 1)
    return level, float(min(1.0, (step - start) / half))


def downscale_classical(rho, factor, mode="avg"):
    """古典降尺度（rho: (B,1,S,S)）：avg＝平均池化（論文，用於 ρ）；max＝最大池化（0/1 外積，保持 0/1 範圍）。"""
    if factor <= 1:
        return rho
    return F.max_pool2d(rho, factor) if mode == "max" else F.avg_pool2d(rho, factor)


def real_at_level(rho_full, side, alpha, pool=4, mode="avg"):
    """把真實 ρ（B,L,L）降到當前解析度 side；淡入期與「較低解析度再放大」混合（同 ProGAN）。"""
    r = downscale_classical(rho_full[:, None], rho_full.shape[-1] // side, mode)
    if alpha < 1.0 and side > 4:
        low = downscale_classical(r, pool, mode)
        r = alpha * r + (1 - alpha) * F.interpolate(low, scale_factor=pool, mode="nearest")
    return r
