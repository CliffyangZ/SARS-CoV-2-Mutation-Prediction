"""古典判別器：各邊長輸入、梯度、blur。"""
import torch
from Research.models.classical_discriminator import Blur, ClassicalDiscriminator


def test_scores_and_grad():
    D = ClassicalDiscriminator([8, 8, 4])
    for lv, s in enumerate([4, 16, 64]):
        x = torch.rand(3, 1, s, s, requires_grad=True)
        out = D(x, lv, 0.5)
        assert out.shape == (3,)
        out.sum().backward()
        assert x.grad is not None
    assert any(isinstance(m, Blur) for m in D.modules())


def test_blur_is_fixed():
    assert sum(p.numel() for p in Blur(4).parameters()) == 0


def test_minibatch_std_gradient_finite_on_identical_batch():
    """批次內樣本完全相同時（標準差為 0），梯度也不可為 NaN。"""
    D = ClassicalDiscriminator([4, 4])
    x = torch.rand(1, 1, 4, 4).repeat(5, 1, 1, 1).requires_grad_(True)
    D(x, 0, 1.0).sum().backward()
    assert torch.isfinite(x.grad).all()
