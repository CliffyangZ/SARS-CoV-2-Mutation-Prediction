"""古典編碼器（DeepSets）：突變 token 集合 → 潛向量 z̃ ∈ (−π/2, π/2)^k。

縮放用 (π/2)·tanh(z)：不用 π·tanh，因為 R_y(π) 與 R_y(−π) 只差全域相位，飽和時兩個極端會被映成同一點。
"""
import math

import torch
from torch import nn

from .dataset import N_SYM
from .mutations import N_POS


class Encoder(nn.Module):
    """位置嵌入＋胺基酸嵌入 → MLP → 遮罩加總池化 → 線性到 k 維。"""

    def __init__(self, k=6, dim=64, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))
        self.k = k
        self.pos_emb = nn.Embedding(N_POS + 1, dim)
        self.sym_emb = nn.Embedding(N_SYM, dim)
        self.phi = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, k))

    def latent(self, pos, sym, valid):
        """回傳寬潛向量 u（B, dim）：池化後經 rho 前兩層；最後一層 rho[2] 即角度映射 A。"""
        h = self.phi(self.pos_emb(pos) + self.sym_emb(sym))
        h = (h * valid.unsqueeze(-1)).sum(1) / 16.0
        return self.rho[1](self.rho[0](h))

    def raw(self, pos, sym, valid):
        """回傳未縮放的 z（B, k）。空集合的池化為 0。"""
        return self.rho[2](self.latent(pos, sym, valid))

    def encode(self, pos, sym, valid):
        """回傳 (z̃, z_raw, u)。"""
        u = self.latent(pos, sym, valid)
        z = self.rho[2](u)
        return self.scale(z), z, u

    @staticmethod
    def scale(z):
        """潛向量縮放 (π/2)·tanh(z)，輸出落在 (−π/2, π/2)。"""
        return (math.pi / 2) * torch.tanh(z)

    def forward(self, pos, sym, valid):
        """回傳 (z̃, z_raw)。"""
        z = self.raw(pos, sym, valid)
        return self.scale(z), z
