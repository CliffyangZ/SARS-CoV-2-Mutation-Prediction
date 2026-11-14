"""損失函數（論文式 3–10）。

式 (6)–(10) 目前實作的是標準 relativistic-average hinge（Jolicoeur-Martineau）。
注意：尚未對照 QuStyleGAN 論文原文的式 (8)(9)（論文全文不在專案內），若兩者符號不同需修正。
"""
import torch
import torch.nn.functional as F


def g_loss_logistic(d_fake):
    """式 (3)：生成器 logistic 損失 E[softplus(-D(G(z)))]。"""
    return F.softplus(-d_fake).mean()


def d_loss_logistic(d_real, d_fake):
    """式 (4)：判別器 logistic 損失 E[softplus(D(G(z)))] + E[softplus(-D(x))]。"""
    return F.softplus(d_fake).mean() + F.softplus(-d_real).mean()


def r1_penalty(d_real_out, x_real, gamma1=10.0):
    """式 (5)：R1 型梯度懲罰 (γ1/2)·E‖∇_x D(x)‖²（γ2=0，不懲罰生成樣本）。
    x_real 需 requires_grad=True，d_real_out 為 D(x_real)。"""
    grad, = torch.autograd.grad(d_real_out.sum(), x_real, create_graph=True)
    return 0.5 * gamma1 * grad.pow(2).flatten(1).sum(1).mean()


def d_loss_rahinge(d_real, d_fake):
    """式 (6)–(8)：relativistic-hinge 判別器損失。
    C(x)=D(x)-E[D(G)]；C(G)=D(G)-E[D(x)]；L_D = E[relu(1-C(x))] + E[relu(1+C(G))]。"""
    c_real = d_real - d_fake.mean()
    c_fake = d_fake - d_real.mean()
    return F.relu(1 - c_real).mean() + F.relu(1 + c_fake).mean()


def g_loss_rahinge(d_real, d_fake):
    """式 (9)–(10)：relativistic-hinge 生成器損失（與 D 對稱，真假互換）。"""
    c_real = d_real - d_fake.mean()
    c_fake = d_fake - d_real.mean()
    return F.relu(1 + c_real).mean() + F.relu(1 - c_fake).mean()


def diversity_reg(fake1, fake2, z1, z2, weight, eps=1e-5):
    """批次內兩兩差異式多樣性正則（模式尋求式，004 R3）：
    lz = mean|G(z1)-G(z2)| / mean|z1-z2|，回傳 weight / (lz + eps)；輸出越相似懲罰越大。
    weight == 0 時直接回傳 0（不計算，確保預設行為不變）。"""
    if weight == 0:
        return fake1.new_zeros(())
    lz = (fake1 - fake2).abs().flatten(1).mean(1) / ((z1 - z2).abs().mean(1) + eps)
    return weight / (lz.mean() + eps)
