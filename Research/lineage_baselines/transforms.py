"""序列轉影像：Spike2Signal 訊號 → Recurrence Plot、GAF、MTF（研究文件 D4、D5）。

全部以 PyTorch 在 GPU 上對一批序列即時產生，不預存影像（17k 張 360×360 會吃掉大量記憶體）。
輸出為 3 通道、360×360、值在 [0, 1] 的影像；論文稱 RP 為 RGBA，alpha 為常數，不使用。
"""
import functools

import numpy as np
import torch
import torch.nn.functional as F

from .data import ALPHABET

IMAGE_SIZE = 360
MTF_BINS = 8  # 論文未說明分箱數，自行選用（chosen）


def signal_from_ids(ids):
    """Spike2Signal 的 torch 版：符號索引 → [-1, 1] 的訊號（與 data.to_signal 相同）。"""
    return ids.float() / (len(ALPHABET) - 1) * 2.0 - 1.0


SIGNAL_MODES = ("raw", "cumsum")


def apply_signal_mode(sig, mode="raw"):
    """訊號後處理。raw：不變（預設，與既有結果一致）；cumsum：先減去每條序列的平均、沿序列累積（零均值隨機漫步），再逐條重新縮放到 [-1, 1]。

    cumsum 是自行加入的選項（論文未說明 Spike2Signal 是否累積，provenance 為 chosen）。
    累積和可逆，資訊量不變，但訊號變平滑，影像出現大尺度結構。
    """
    if mode == "raw":
        return sig
    if mode != "cumsum":
        raise ValueError(f"未知訊號模式：{mode}，可用：{SIGNAL_MODES}")
    c = torch.cumsum(sig - sig.mean(dim=1, keepdim=True), dim=1)  # 去均值：避免單調漂移，圖形才有論文的交叉結構
    c = c - c.min(dim=1, keepdim=True).values
    return 2.0 * c / c.max(dim=1, keepdim=True).values.clamp_min(1e-12) - 1.0


def rp_matrix(x):
    """Recurrence Plot 的距離矩陣（m=1、τ=1）：|x_i - x_j|，形狀 (B, L, L)。"""
    return (x[:, :, None] - x[:, None, :]).abs()


def gaf_matrix(x):
    """Gramian Angular Summation Field：cos(φ_i + φ_j)，φ = arccos(x)，形狀 (B, L, L)。"""
    x = x.clamp(-1, 1)
    s = torch.sqrt((1 - x * x).clamp_min(0))
    return x[:, :, None] * x[:, None, :] - s[:, :, None] * s[:, None, :]


def quantize(x, q=MTF_BINS):
    """把 [-1, 1] 的訊號均分成 q 個 bin，回傳 (B, L) 的 bin 索引。"""
    return ((x + 1) / 2 * q).long().clamp(0, q - 1)


def mtf_transition(bins, q=MTF_BINS):
    """由 bin 序列計算列正規化的馬可夫轉移矩陣 (B, q, q)。"""
    b = bins.shape[0]
    idx = bins[:, :-1] * q + bins[:, 1:]
    counts = torch.zeros(b, q * q, device=bins.device)
    counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))
    counts = counts.view(b, q, q)
    return counts / counts.sum(-1, keepdim=True).clamp_min(1)


def mtf_matrix(x, q=MTF_BINS):
    """Markov Transition Field：MTF_ij = W[bin_i, bin_j]，形狀 (B, L, L)。"""
    bins = quantize(x, q)
    return _mtf_lookup(mtf_transition(bins, q), bins)


def _mtf_lookup(w, bins):
    """以 bin 索引查表：out[b, i, j] = w[b, bins[b, i], bins[b, j]]。"""
    rows = torch.gather(w, 1, bins[:, :, None].expand(-1, -1, w.shape[2]))  # (B, L, q)
    return torch.gather(rows, 2, bins[:, None, :].expand(-1, bins.shape[1], -1))  # (B, L, L)


@functools.lru_cache(maxsize=4)
def _lut(device):
    """固定的 256 級 viridis 色彩對照表 (256, 3)。"""
    from matplotlib import colormaps

    return torch.tensor(colormaps["viridis"](np.linspace(0, 1, 256))[:, :3], dtype=torch.float32, device=device)


def to_image(kind, x, size=IMAGE_SIZE):
    """把 (B, L) 的訊號轉成 (B, 3, size, size) 影像；kind 為 rp、gaf 或 mtf。"""
    if kind == "rp":
        m = rp_matrix(x) / 2.0  # 訊號範圍為 [-1, 1]，最大距離 2
    elif kind == "gaf":
        m = (gaf_matrix(x) + 1) / 2
    elif kind == "mtf":
        m = mtf_matrix(x)
    else:
        raise ValueError(f"未知影像類型：{kind}")
    m = F.interpolate(m[:, None], size=(size, size), mode="area")[:, 0].clamp(0, 1)
    idx = (m * 255).round().long()
    return _lut(str(x.device))[idx].permute(0, 3, 1, 2).contiguous()
