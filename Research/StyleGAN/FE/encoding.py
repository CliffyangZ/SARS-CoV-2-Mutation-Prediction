"""向量 / 特徵圖 → 密度矩陣 ρ（research R6）。

真實資料：ρ = v vᵀ，v 為 L2 正規化的 0/1 突變向量（實對稱、半正定、跡為 1）。
生成資料：ρ = G Gᵀ / Tr(G Gᵀ)（論文未說明如何得到合法 ρ，此為本專案假設）。
讀回：v̂ = sqrt(diag ρ)，與論文「取對角線、開根號」一致。
"""
from dataclasses import asdict, dataclass

import numpy as np
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


# =====================================================================
# 002：可切換的突變編碼（research R1–R5）
# =====================================================================
@dataclass(frozen=True)
class EncodingSpec:
    """一種突變編碼的全部設定；隨 config.json 保存。`legacy` 即 run01 的行為（供重現）。"""
    name: str = "legacy"
    drop_const_threshold: float = None   # 訓練期頻率 ≥ 此值的位點排除；None 表示不排除
    normalize: str = "l2"                # 真實樣本向量：l2（v/‖v‖）或 none（0/1）
    order: str = "spike"                 # 位點排列：spike 位置，或 freq（訓練期頻率由高到低）
    readout: str = "topk"                # 讀回突變集：topk（固定 k）或 threshold（由輸出決定突變數）
    readout_threshold: float = 0.5       # 閾值法的門檻（l2 編碼為相對最大 diag 的比例）
    output_norm: str = "batch"           # 生成器輸出：batch（舊）/ sample（逐樣本）/ sigmoid

    def to_dict(self):
        """轉成可寫入 json 的 dict。"""
        return asdict(self)

    @property
    def pool_mode(self):
        """真實樣本降尺度方式：0/1 外積用 max（保持 0/1 範圍），ρ 用平均池化（論文）。"""
        return "max" if self.normalize == "none" else "avg"


# 單一編碼成分；組合以 "+" 連接（如 drop_const+nonorm），各成分的欄位互不衝突
_PARTS = {
    "drop_const": dict(drop_const_threshold=0.9),
    "nonorm": dict(normalize="none"),
    "freq_order": dict(order="freq"),
}


def get_encoding(name="legacy"):
    """由名稱取得 EncodingSpec；非 legacy 一律用閾值讀回與逐樣本（或 sigmoid）輸出。"""
    if name in (None, "", "legacy"):
        return EncodingSpec()
    kw = {}
    for part in name.split("+"):
        if part not in _PARTS:
            raise ValueError(f"未知的編碼成分 {part!r}；可用：legacy、{', '.join(_PARTS)}（以 + 組合）")
        kw.update(_PARTS[part])
    kw["readout"] = "threshold"
    kw["output_norm"] = "sigmoid" if kw.get("normalize") == "none" else "sample"
    return EncodingSpec(name=name, **kw)


def spec_from_dict(d):
    """從 config.json 的 dict 還原 EncodingSpec；缺欄位（舊 run）視為 legacy。"""
    return EncodingSpec(**d) if d else EncodingSpec()


def encode_real(x, spec):
    """真實 0/1 向量 (B,L) → (B,L,L) 影像矩陣。l2：ρ=vvᵀ（現行）；none：0/1 外積 xxᵀ。"""
    x = torch.as_tensor(x, dtype=torch.float32)
    return vector_to_rho(x) if spec.normalize == "l2" else x.unsqueeze(-1) * x.unsqueeze(-2)


def fake_image(g, spec, side):
    """生成器輸出 (B,1,S,S) → 餵給判別器的影像。sigmoid 編碼直接用輸出；否則 ρ=GGᵀ/Tr 再乘邊長（R2 假設）。"""
    if spec.output_norm == "sigmoid":
        return g
    return featuremap_to_rho(g)[:, None] * side


def real_scale(spec, side):
    """真實影像降尺度後的倍率：與 fake_image 一致（l2 乘邊長；0/1 不縮放）。"""
    return 1.0 if spec.normalize == "none" else float(side)


def matrix_diag(m, spec):
    """影像矩陣 → 各位點的讀回量：l2 取 sqrt(diag ρ)（論文），none 直接取對角線（機率）。"""
    m = m[:, 0] if m.dim() == 4 else m
    if spec.output_norm == "sigmoid" or spec.normalize == "none":
        return m.diagonal(dim1=-2, dim2=-1).clamp_min(0)
    return rho_to_diag(m)


def generator_diag(g, spec):
    """生成器輸出 (B,1,S,S) → 讀回量 (B,L)。"""
    if spec.output_norm == "sigmoid":
        return matrix_diag(g, spec)
    return rho_to_diag(featuremap_to_rho(g))


def readout(diag, spec, k=None):
    """讀回量 (n,L) → (突變索引 (n,k_max) 以 -1 填充, 各樣本突變數 (n,))。

    topk：每個樣本固定取前 k 大（legacy，只為重現 run01）。
    threshold：機率編碼取 p ≥ 門檻；l2 編碼取 ≥ 門檻 × 該樣本最大值（突變數由輸出決定，可為 0）。
    """
    diag = np.asarray(diag, dtype=float)
    n, L = diag.shape
    if spec.readout == "topk":
        k = 0 if k is None else k
        idx = np.argsort(-diag, axis=1, kind="stable")[:, :k]
        return idx, np.full(n, k, dtype=int)
    if spec.output_norm == "sigmoid" or spec.normalize == "none":
        keep = diag >= spec.readout_threshold
    else:
        keep = diag >= spec.readout_threshold * diag.max(1, keepdims=True).clip(1e-12)
    ks = keep.sum(1)
    idx = np.full((n, max(int(ks.max()), 0) if n else 0), -1, dtype=int)
    for r in range(n):
        pos = np.where(keep[r])[0]
        pos = pos[np.argsort(-diag[r, pos], kind="stable")]
        idx[r, :len(pos)] = pos
    return idx, ks.astype(int)


def to_rho(v, eps=1e-12):
    """0/1（或機率）突變向量 → 合法密度矩陣 ρ=vvᵀ/‖v‖²（對稱、半正定、跡為 1）。
    全零向量沒有合法純態，回傳全零矩陣（跡為 0），由呼叫端單獨計入「空突變集」。"""
    v = torch.as_tensor(v, dtype=torch.float32)
    n2 = (v * v).sum(-1, keepdim=True)
    u = v / n2.clamp_min(eps).sqrt()
    return u.unsqueeze(-1) * u.unsqueeze(-2) * (n2 > eps).float().unsqueeze(-1)
