"""式 3–10 的手算驗證。"""
import math
import torch
from models.StyleGAN import losses as L


def sp(x):
    return math.log1p(math.exp(x))


def test_logistic():
    f, r = torch.tensor([1.0, -1.0]), torch.tensor([2.0, 0.0])
    assert abs(L.g_loss_logistic(f).item() - (sp(-1) + sp(1)) / 2) < 1e-6
    exp = (sp(1) + sp(-1)) / 2 + (sp(-2) + sp(0)) / 2
    assert abs(L.d_loss_logistic(r, f).item() - exp) < 1e-6


def test_r1_penalty_quadratic():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    out = (x ** 2).sum(1)                       # ∇ = 2x = [2, 4]，‖∇‖² = 20
    assert abs(L.r1_penalty(out, x, gamma1=10).item() - 0.5 * 10 * 20) < 1e-5


def test_rahinge_hand():
    r, f = torch.tensor([2.0]), torch.tensor([0.0])   # C(x)=2, C(G)=-2
    assert L.d_loss_rahinge(r, f).item() == 0.0       # relu(1-2)+relu(1-2)
    assert L.g_loss_rahinge(r, f).item() == 3.0 + 3.0  # relu(1+2)+relu(1+2)


def test_gradients_flow():
    f = torch.randn(4, requires_grad=True)
    L.g_loss_rahinge(torch.randn(4), f).backward()
    assert f.grad is not None


def test_diversity_reg():
    """004：權重 0 回 0；輸出相同時懲罰大於輸出分散時；梯度可回傳。"""
    z1, z2 = torch.randn(4, 8), torch.randn(4, 8)
    same = torch.rand(4, 1, 4, 4)
    assert L.diversity_reg(same, same, z1, z2, 0.0).item() == 0.0
    spread = torch.rand(4, 1, 4, 4)
    near = L.diversity_reg(same, same + 1e-3, z1, z2, 1.0)
    far = L.diversity_reg(same, spread, z1, z2, 1.0)
    assert near > far > 0
    a = torch.rand(4, 1, 4, 4, requires_grad=True)
    L.diversity_reg(a, spread, z1, z2, 1.0).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
