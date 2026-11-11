"""向量 / 特徵圖 → 密度矩陣 ρ（research R6）。

真實資料：ρ = v vᵀ，v 為 L2 正規化的 0/1 突變向量（實對稱、半正定、跡為 1）。
生成資料：ρ = G Gᵀ / Tr(G Gᵀ)（論文未說明如何得到合法 ρ，此為本專案假設）。
讀回：v̂ = sqrt(diag ρ)，與論文「取對角線、開根號」一致。
"""
import torch


def vector_to_rho(x, eps=1e-12):
    """x: (B, L) 0/1 → ρ: (B, L, L)；全零向量以 eps 避免除以 0。"""
    x = torch.as_tensor(x, dtype=torch.float32)
    v = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v.unsqueeze(-1) * v.unsqueeze(-2)


def featuremap_to_rho(g, eps=1e-12):
    """g: (B, 1, L, L) 或 (B, L, L) 特徵圖 → ρ = G Gᵀ / Tr(G Gᵀ)。"""
    if g.dim() == 4:
        g = g[:, 0]
    rho = g @ g.transpose(-1, -2)
    tr = rho.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
    return rho / tr[:, None, None]


def rho_to_diag(rho):
    """ρ → sqrt(diag ρ)，長度 L（負值因數值誤差先截為 0）。"""
    return rho.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
