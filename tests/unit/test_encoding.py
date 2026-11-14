"""ρ 編碼的性質測試。"""
import torch
from Research.StyleGAN.FE.encoding import featuremap_to_rho, rho_to_diag, vector_to_rho


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


# ---------------- 002：可切換編碼（T017） ----------------
import numpy as np
import pytest
from Research.StyleGAN.FE.encoding import (encode_real, fake_image, get_encoding, matrix_diag, readout, spec_from_dict,
                                  to_rho)

NAMES = ["legacy", "drop_const", "nonorm", "freq_order", "drop_const+nonorm", "drop_const+freq_order"]


@pytest.mark.parametrize("name", NAMES)
def test_write_read_roundtrip(name):
    """寫入（encode_real）→ 讀回（matrix_diag + readout）可還原 0/1 突變集；突變數由輸出決定（非 legacy）。"""
    spec = get_encoding(name)
    x = torch.tensor([[1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1]], dtype=torch.float32)
    idx, ks = readout(matrix_diag(encode_real(x, spec), spec).numpy(), spec, k=2)
    if spec.readout == "topk":                       # legacy：固定 k，只看前 k 個
        assert (ks == 2).all()
        return
    assert ks.tolist() == x.sum(1).int().tolist()
    for r in range(len(x)):
        assert set(idx[r][idx[r] >= 0].tolist()) == set(np.where(x[r].numpy() > 0)[0].tolist())


def test_get_encoding_components_and_errors():
    s = get_encoding("drop_const+nonorm")
    assert (s.drop_const_threshold, s.normalize, s.output_norm, s.readout) == (0.9, "none", "sigmoid", "threshold")
    assert get_encoding("freq_order").order == "freq" and get_encoding("legacy").readout == "topk"
    assert spec_from_dict(None).name == "legacy" and spec_from_dict(s.to_dict()) == s
    with pytest.raises(ValueError):
        get_encoding("nonsense")


@pytest.mark.parametrize("name", NAMES)
def test_to_rho_is_valid_density_matrix(name):
    """to_rho：非零向量得到對稱、半正定、跡為 1 的矩陣（FR-006a）；全零向量不產生 NaN。"""
    v = torch.tensor([[1., 0, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]])
    rho = to_rho(v)
    assert torch.isfinite(rho).all()
    assert torch.allclose(rho, rho.transpose(-1, -2))
    assert torch.allclose(rho[:2].diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(2), atol=1e-6)
    assert torch.linalg.eigvalsh(rho[:2]).min() > -1e-6
    assert rho[2].abs().sum() == 0                  # 空突變集：沒有合法純態，記為零矩陣


def test_nonorm_value_independent_of_mutation_count():
    """nonorm：單一突變的對角線值固定為 1；l2 則隨突變數變小（稀釋）。"""
    x = torch.tensor([[1., 0, 0, 0], [1., 1, 1, 0]])
    d_none = matrix_diag(encode_real(x, get_encoding("nonorm")), get_encoding("nonorm"))
    d_l2 = matrix_diag(encode_real(x, get_encoding("legacy")), get_encoding("legacy"))
    assert d_none[0, 0] == d_none[1, 0] == 1.0
    assert d_l2[0, 0] > d_l2[1, 0]


def test_zero_vector_readout_is_empty_set_no_nan():
    for name in ("legacy", "drop_const", "nonorm"):
        spec = get_encoding(name)
        m = encode_real(torch.zeros(2, 5), spec)
        assert torch.isfinite(m).all()
        idx, ks = readout(matrix_diag(m, spec).numpy(), spec, k=0)
        assert (idx < 0).all() and (ks == 0).all()


def test_fake_image_paths():
    g = torch.rand(2, 1, 4, 4)
    assert torch.equal(fake_image(g, get_encoding("nonorm"), 4), g)
    rho = fake_image(g, get_encoding("legacy"), 4)
    assert torch.allclose(rho[:, 0].diagonal(dim1=-2, dim2=-1).sum(), torch.tensor(2. * 4), atol=1e-4)    # 每個樣本 ρ 的跡為 1，乘邊長 4
