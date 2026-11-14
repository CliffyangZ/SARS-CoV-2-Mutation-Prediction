"""古典判別器（依 reference_gan.py）：各解析度輸入、梯度、Blur / minibatch stddev。"""
import pytest
import torch
from models.StyleGAN import update_average
from models.StyleGAN.classical_discriminator import Discriminator
from models.StyleGAN.custom_layers import BlurLayer, StddevLayer


def test_scores_and_grad():
    D = Discriminator(64, fmap_max=8)
    for depth, s in enumerate([4, 16, 64]):
        x = torch.rand(4, 1, s, s, requires_grad=True)
        out = D(x, depth, 0.5)
        assert out.shape == (4, 1)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_depth_out_of_range():
    with pytest.raises(AssertionError):
        Discriminator(64, fmap_max=8)(torch.rand(4, 1, 64, 64), 3)


def test_blur_is_fixed():
    assert sum(p.numel() for p in BlurLayer().parameters()) == 0


def test_stddev_layer_adds_channel_and_finite_grad_on_identical_batch():
    x = torch.rand(1, 3, 4, 4).repeat(4, 1, 1, 1).requires_grad_(True)
    y = StddevLayer(4, 1)(x)
    assert y.shape == (4, 4, 4, 4)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_update_average():
    a, b = Discriminator(16, fmap_max=4), Discriminator(16, fmap_max=4)
    update_average(a, b, 0.0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_smaller_width_has_fewer_params_and_mbstd_group_forward():
    """004：fmap_max 較小參數較少；mbstd_group=16 在 batch 16 可前向；預設群組為 4。"""
    big, small = Discriminator(64, fmap_max=16), Discriminator(64, fmap_max=8)
    assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in big.parameters())
    assert Discriminator(64, fmap_max=8).mbstd_group_size == 4
    D = Discriminator(64, fmap_max=8, mbstd_group_size=16)
    assert D(torch.rand(16, 1, 64, 64), 2).shape == (16, 1)
