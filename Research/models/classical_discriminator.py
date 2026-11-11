"""古典漸進式判別器（FR-012）：卷積 → 固定 depthwise blur → 卷積 → 平均池化，最後 dense。

輸入與量子判別器相同：單通道 2ⁿ×2ⁿ 密度矩陣，使比較公平。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def minibatch_std(x):
    """minibatch 標準差特徵（StyleGAN 判別器的標準做法，用來偵測並抑制 mode collapse）：
    跨樣本的標準差取平均後，作為一個額外通道附加在特徵圖上。"""
    s = (x.var(dim=0, unbiased=False) + 1e-8).sqrt().mean().expand(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.cat([x, s], dim=1)


class Blur(nn.Module):
    """固定（不可訓練）的 depthwise [1,2,1] 低通濾波。"""

    def __init__(self, ch):
        super().__init__()
        k = torch.tensor([1., 2., 1.]); k = (k[:, None] * k[None, :]); k = k / k.sum()
        self.register_buffer("k", k.expand(ch, 1, 3, 3).contiguous())
        self.ch = ch

    def forward(self, x):
        """對每個通道套用固定低通濾波。"""
        return F.conv2d(x, self.k, padding=1, groups=self.ch)


class DBlock(nn.Module):
    """一個判別器區塊：conv → blur → conv → 平均池化（×4 降尺度，對應生成器的 ×4 放大）。"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.c1, self.blur, self.c2 = nn.Conv2d(c_in, c_in, 3, padding=1), Blur(c_in), nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x):
        """卷積 → blur → 卷積 → 平均池化。"""
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.c2(self.blur(x)), 0.2)
        return F.avg_pool2d(x, 4)


class ClassicalDiscriminator(nn.Module):
    """forward(ρ (B,1,S,S), level, alpha) → 分數 (B,)。"""

    def __init__(self, channels):
        super().__init__()
        self.from_map = nn.ModuleList(nn.Conv2d(1, c, 1) for c in channels)
        self.blocks = nn.ModuleList(DBlock(channels[i], channels[i - 1]) for i in range(1, len(channels)))
        c0 = channels[0]
        self.final_conv, self.dense = nn.Conv2d(c0 + 1, c0, 3, padding=1), nn.Linear(c0 * 16, 1)

    def forward(self, x, level, alpha=1.0):
        """ρ → 逐區塊降尺度（淡入期與較低解析度輸入混合）→ dense → 分數。"""
        h = F.leaky_relu(self.from_map[level](x), 0.2)
        for i in range(level, 0, -1):
            h = self.blocks[i - 1](h)
            if i == level and alpha < 1.0:        # 淡入：與「較低解析度的輸入」混合
                low = F.leaky_relu(self.from_map[level - 1](F.avg_pool2d(x, 4)), 0.2)
                h = alpha * h + (1 - alpha) * low
        h = F.leaky_relu(self.final_conv(minibatch_std(h)), 0.2)
        return self.dense(h.flatten(1)).squeeze(1)
