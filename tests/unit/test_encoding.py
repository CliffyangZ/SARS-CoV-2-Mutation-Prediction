"""ρ 編碼的性質測試。"""
import torch
from Research.FE.encoding import featuremap_to_rho, rho_to_diag, vector_to_rho


def test_vector_rho_properties():
    x = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32)
    rho = vector_to_rho(x)
    assert torch.allclose(rho, rho.transpose(-1, -2))
    assert torch.allclose(rho.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(2), atol=1e-6)
    assert torch.linalg.eigvalsh(rho).min() > -1e-6


def test_featuremap_rho_properties():
    rho = featuremap_to_rho(torch.randn(3, 1, 8, 8))
    assert torch.allclose(rho, rho.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(rho.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(3), atol=1e-5)
    assert torch.linalg.eigvalsh(rho).min() > -1e-5


def test_diag_sqrt_of_real_is_norm_over_norm():
    x = torch.tensor([[1., 0, 1, 1]])
    v = x / x.norm()
    assert torch.allclose(rho_to_diag(vector_to_rho(x)), v.abs(), atol=1e-6)
    # sqrt(diag(v vᵀ)) = |v|
