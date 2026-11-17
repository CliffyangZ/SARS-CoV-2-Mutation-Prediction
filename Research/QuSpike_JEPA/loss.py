"""JEPA 損失：不變性項、二階白化加抗塌陷懲罰（輸出有界，不要求高斯）與有效秩。"""
import torch
import torch.nn.functional as F


def effective_rank(x):
    """嵌入的有效秩：中心化後奇異值正規化的熵取 exp；塌陷時接近 1。回傳 float。"""
    x = x.detach().float()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    if float(s.sum()) <= 1e-12:
        return 1.0
    p = s / s.sum()
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


COV_WEIGHT = 25.0  # 非對角共變異數權重；1 時 d 小的情況下會落入秩 1 的解


def _cov(z):
    z = z - z.mean(0, keepdim=True)
    return z.T @ z / max(z.shape[0] - 1, 1)


def whitening_penalty(z, kind="vicreg"):
    """kind='vicreg'：變異數下限＋非對角共變異數；'identity'：‖Cov − I‖²；'none'：不加。"""
    d = z.shape[1]
    if kind == "none":
        return z.new_zeros(())
    c = _cov(z)
    if kind == "identity":
        return ((c - torch.eye(d, device=z.device)) ** 2).sum() / d
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(1.0 - std).mean()
    off = c - torch.diag(torch.diag(c))
    return 25.0 * var + COV_WEIGHT * (off ** 2).sum() / d


def vicreg(a, b, sim=25.0, whiten="vicreg"):
    """回傳 (總損失, 分項字典)。不變性為兩 view 嵌入的均方差；白化懲罰分別加在兩個 view 上。"""
    inv = F.mse_loss(a, b)
    reg = whitening_penalty(a, whiten) + whitening_penalty(b, whiten)
    total = sim * inv + reg
    return total, {"inv": float(inv.detach()), "reg": float(reg.detach())}


def clip_gradients(params, max_norm=1.0):
    """梯度裁剪，回傳裁剪前的範數。"""
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def min_dim_std(x):
    """嵌入各維度標準差的最小值（診斷用，不單獨作為塌陷判準）。

    有效秩用正規化奇異值、與尺度無關，抓不到「變異趨近 0 但仍有殘餘雜訊」的塌陷，故另看絕對尺度。
    但在寬讀出（m=128）下，單一維度失效不等於整體塌陷，故判準改用 dead_dim_frac。
    """
    x = torch.as_tensor(x).detach().float()
    return float(x.std(0).min()) if x.shape[0] > 1 else 0.0


def dead_dim_frac(x, floor=1e-3):
    """標準差低於 floor 的維度比例（§5 修訂的方差塌陷判準）。

    用比例而非最小值：H 個並聯頭的讀出中偶有一個觀測量變成常數是正常的，
    只有大部分維度都失去變異才算塌陷。
    """
    x = torch.as_tensor(x).detach().float()
    if x.shape[0] <= 1:
        return 1.0
    return float((x.std(0) < floor).float().mean())


def majority_macro_f1(y, n_classes):
    """全部猜最大類時的 macro F1（§5 修訂：probe 未明顯高於此值即視為失敗）。"""
    import numpy as np

    cnt = np.bincount(np.asarray(y), minlength=n_classes)
    if cnt.sum() == 0:
        return 0.0
    p = cnt.max() / cnt.sum()
    return float(2 * p / (p + 1) / n_classes)
