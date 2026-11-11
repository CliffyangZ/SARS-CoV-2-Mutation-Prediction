"""生成器輸出邊長、ρ 合法性、梯度。"""
import pytest
import torch
from Research.models.generator import Generator
from Research.utils.config import SCALES


@pytest.mark.parametrize("scale", [1, 2])
def test_sizes_and_rho(scale):
    c = SCALES[scale]
    G = Generator(c["block_sizes"], c["channels"], 16, 8)
    top = 2 if scale == 1 else 3          # 規模二只測到 256，避免 1024² 拖慢測試
    for lv in range(top + 1):
        rho = G(torch.randn(2, 16), lv, 0.5)
        s = c["block_sizes"][lv]
        assert rho.shape == (2, s, s)
        assert torch.allclose(rho.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(2), atol=1e-4)
        assert torch.linalg.eigvalsh(rho.double()).min() > -1e-6


def test_grad_reaches_mapping():
    G = Generator([4, 16], [8, 8], 16, 8)
    G(torch.randn(2, 16), 1, 0.3).sum().backward()
    assert G.mapping[0].weight.grad is not None and G.mapping[0].weight.grad.abs().sum() > 0


def test_noise_injection_adds_parameters_and_randomness():
    torch.manual_seed(0)
    G = Generator([4, 16], [8, 8], 16, 8, noise=True)
    assert hasattr(G.blocks[1], "nw1")
    z = torch.randn(2, 16)
    with torch.no_grad():
        for b in G.blocks:
            b.nw1.fill_(0.5); b.nw2.fill_(0.5)
    a, b = G.feature_map(z, 1, 1.0), G.feature_map(z, 1, 1.0)
    assert not torch.allclose(a, b)


def test_output_depends_on_latent_at_init():
    """初始化時，不同 z 必須得到明顯不同的對角線（避免 mapping network 訊號衰減造成的「天生塌縮」）。"""
    torch.manual_seed(0)
    G = Generator([4, 16, 64], [16, 16, 8], 64, 8)
    with torch.no_grad():
        d = G(torch.randn(64, 64), 2, 1.0).diagonal(dim1=-2, dim2=-1)
    assert d.std(0).mean() > 1e-3
