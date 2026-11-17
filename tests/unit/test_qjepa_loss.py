"""損失與塌陷偵測。"""
import torch

from Research.QuSpike_JEPA import loss as L


def test_correlated_dims_penalised_more_than_independent():
    g = torch.Generator().manual_seed(0)
    ind = torch.randn(512, 6, generator=g)
    cor = torch.randn(512, 1, generator=g).repeat(1, 6)
    assert float(L.whitening_penalty(cor)) > float(L.whitening_penalty(ind))


def test_whitening_kinds():
    z = torch.randn(256, 4)
    assert float(L.whitening_penalty(z, "none")) == 0.0
    assert float(L.whitening_penalty(z, "identity")) < 0.2


def test_clip_gradients_bounds_norm():
    p = torch.nn.Parameter(torch.ones(4))
    (p * 100).sum().backward()
    L.clip_gradients([p], 1.0)
    assert float(p.grad.norm()) <= 1.0 + 1e-5
