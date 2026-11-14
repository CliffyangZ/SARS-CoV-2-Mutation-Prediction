"""生成器（依 reference_gan.py）：輸出形狀與值域、各子模組、梯度、截斷 / 風格混合 / noise。"""
import pytest
import torch
from models.StyleGAN.generator import GMapping, Generator, GSynthesis
from Research.StyleGAN.FE.encoding import featuremap_to_rho

KW = dict(latent_size=16, dlatent_size=16, mapping_fmaps=16, fmap_max=8)


@pytest.mark.parametrize("resolution,top", [(64, 2), (256, 3)])
def test_sizes_range_and_rho(resolution, top):
    G = Generator(resolution, **KW)
    for depth in range(top + 1):
        img = G(torch.randn(2, 16), depth, 0.5)
        s = 4 ** (depth + 1)
        assert img.shape == (2, 1, s, s)
        assert img.min() >= 0 and img.max() <= 1
        rho = featuremap_to_rho(img)
        assert torch.allclose(rho.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(2), atol=1e-4)
        assert torch.linalg.eigvalsh(rho.double()).min() > -1e-6


def test_mapping_broadcast_and_depth():
    m = GMapping(16, 16, dlatent_broadcast=6, mapping_fmaps=16)
    assert m(torch.randn(3, 16)).shape == (3, 6, 16)
    assert GSynthesis(16, resolution=1024, fmap_max=8).depth == 5


def test_depth_out_of_range():
    G = Generator(64, **KW)
    with pytest.raises(AssertionError):
        G(torch.randn(2, 16), 3, 1.0)


def test_grad_reaches_mapping():
    G = Generator(64, **KW)
    G(torch.randn(2, 16), 1, 0.3).sum().backward()
    g = G.g_mapping.map.dense0.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_truncation_updates_in_training_and_applies_in_eval():
    torch.manual_seed(0)
    G = Generator(64, **KW)
    assert G.truncation.avg_latent.abs().sum() == 0
    G(torch.randn(4, 16), 1, 1.0)                       # training 模式 → 更新 avg_latent
    assert G.truncation.avg_latent.abs().sum() > 0
    G.eval(); z = torch.randn(2, 16)
    with torch.no_grad():
        a = G(z, 1, 1.0)
        G.truncation = None
        b = G(z, 1, 1.0)
    assert not torch.allclose(a, b)                     # eval 時仍套用截斷


def test_noise_layer_makes_output_random():
    torch.manual_seed(0)
    G = Generator(64, use_noise=True, **KW).eval()
    z = torch.randn(2, 16)
    with torch.no_grad():
        for n in G.modules():
            if hasattr(n, "noise") and n.noise is not None:
                n.noise.weight.fill_(0.5)
        assert not torch.allclose(G(z, 1, 1.0), G(z, 1, 1.0))


def test_output_depends_on_latent_at_init():
    torch.manual_seed(0)
    G = Generator(64, truncation_psi=0, **KW).eval()
    with torch.no_grad():
        d = featuremap_to_rho(G(torch.randn(64, 16), 2, 1.0)).diagonal(dim1=-2, dim2=-1)
    assert d.std(0).mean() > 1e-4


def test_sample_norm_independent_of_batch():
    """output_norm=sample：同一潛碼放進不同 batch 組成，輸出完全相同（FR-008）。"""
    import torch
    from models.StyleGAN.generator import Generator
    torch.manual_seed(0)
    G = Generator(16, latent_size=8, dlatent_size=8, mapping_layers=2, fmap_base=64, fmap_max=8,
                  output_norm="sample").eval()
    z = torch.randn(4, 8)
    with torch.no_grad():
        a = G(z, 1, 1.0)[0]
        b = G(torch.cat([z[:1], torch.randn(5, 8) * 5]), 1, 1.0)[0]
    assert torch.allclose(a, b, atol=1e-6)
    assert a.min() >= 0 and a.max() <= 1 + 1e-6


def test_batch_norm_is_batch_dependent_and_sigmoid_symmetric():
    """舊的 batch 級正規化確實依賴 batch（所以需要新選項）；sigmoid 輸出對稱且在 [0,1]。"""
    import torch
    from models.StyleGAN.generator import Generator
    torch.manual_seed(0)
    kw = dict(latent_size=8, dlatent_size=8, mapping_layers=2, fmap_base=64, fmap_max=8)
    z = torch.randn(4, 8)
    Gb = Generator(16, output_norm="batch", **kw).eval()
    with torch.no_grad():
        assert not torch.allclose(Gb(z, 1, 1.0)[0], Gb(torch.cat([z[:1], torch.randn(5, 8) * 5]), 1, 1.0)[0], atol=1e-6)
    Gs = Generator(16, output_norm="sigmoid", **kw).eval()
    with torch.no_grad():
        o = Gs(z, 1, 1.0)
    assert torch.allclose(o, o.transpose(-1, -2), atol=1e-6) and o.min() >= 0 and o.max() <= 1


def test_mapping_layers_and_width_change_param_count():
    """004：mapping 層數與寬度上限越小，參數越少（因子確實生效）。"""
    def n(**kw):
        return sum(p.numel() for p in Generator(64, **{**KW, **kw}).parameters())
    assert n(mapping_layers=2) < n(mapping_layers=8)
    assert n(fmap_max=4) < n(fmap_max=8)


def test_load_generator_respects_g_fmap_max():
    """004 回歸：評估階段重建生成器時必須用 g_fmap_max，否則載入 g_fmap_max 較小的檢查點會尺寸不符。"""
    from Research.StyleGAN.train.train_StyleGAN import build_models
    from Research.StyleGAN.evaluate.generate import load_generator
    from Research.StyleGAN.utils.config import make_config
    cfg = make_config(1, fmap_max=8, g_fmap_max=4, mapping_layers=2, latent_dim=16)
    G, _ = build_models(cfg)
    ck = {"G": G.state_dict()}
    assert load_generator(cfg, ck) is not None
