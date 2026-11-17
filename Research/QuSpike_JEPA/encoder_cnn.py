"""1D CNN 編碼器（U-Net 的 encoder 半邊，沒有 skip connection）：突變 token → 1273 位稠密張量 → 卷積 → z̃。

與 DeepSets 的介面相同（forward 回傳 (z̃, z)），供 `--encoder cnn1d` 消融使用。
遮蔽符號有獨立通道，未列出的位置全為 0（即「確定等於參考」），兩者可區分。
"""
import torch
from torch import nn

from .dataset import N_SYM
from .encoder import Encoder
from .mutations import N_POS


class CNN1DEncoder(nn.Module):
    """卷積加步幅下採樣，最後平均與最大池化合併後線性到 k 維。"""

    def __init__(self, k=6, dim=64, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.k = k
        c = [N_SYM, dim // 2, dim, dim]
        layers = []
        for i in range(3):
            layers += [nn.Conv1d(c[i], c[i + 1], kernel_size=9, stride=2, padding=4), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(2 * dim, k)

    def dense(self, pos, sym, valid):
        """token → (B, N_SYM, N_POS) 的 0/1 張量；位置為 1 起算。"""
        x = torch.zeros(pos.shape[0], N_SYM, N_POS + 1, device=pos.device)
        b = torch.arange(pos.shape[0], device=pos.device).unsqueeze(1).expand_as(pos)
        x[b[valid], sym[valid], pos[valid]] = 1.0
        return x[:, :, 1:]

    def raw(self, pos, sym, valid):
        """回傳未縮放的潛向量 z（B, k）。"""
        h = self.conv(self.dense(pos, sym, valid))
        return self.out(torch.cat([h.mean(-1), h.amax(-1)], 1))

    scale = staticmethod(Encoder.scale)

    def forward(self, pos, sym, valid):
        """回傳 (z̃, z)。"""
        z = self.raw(pos, sym, valid)
        return self.scale(z), z
