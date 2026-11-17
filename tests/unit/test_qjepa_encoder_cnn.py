"""CNN 編碼器：介面與 DeepSets 一致、遮蔽與參考可區分、梯度可回傳。"""
import torch

from Research.QuSpike_JEPA.dataset import MASK_SYM
from Research.QuSpike_JEPA.encoder_cnn import CNN1DEncoder


def _batch():
    pos = torch.tensor([[10, 20, 0], [614, 0, 0]])
    sym = torch.tensor([[3, 5, 0], [MASK_SYM, 0, 0]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    return pos, sym, valid


def test_output_shape_and_range():
    zt, z = CNN1DEncoder(k=6, seed=0)(*_batch())
    assert zt.shape == (2, 6) and zt.abs().max() <= 1.5708 + 1e-5


def test_mask_differs_from_unlisted_position():
    enc = CNN1DEncoder(k=6, seed=0)
    pos, sym, valid = _batch()
    masked = enc.dense(pos, sym, valid)
    assert masked[1, MASK_SYM, 613] == 1.0 and masked[0].sum() == 2.0
    empty = enc.dense(pos, sym, torch.zeros_like(valid))
    assert empty.sum() == 0.0


def test_gradients_flow():
    enc = CNN1DEncoder(k=6, seed=0)
    enc(*_batch())[0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_model_builds_cnn_encoder_from_config():
    import copy

    from Research.QuSpike_JEPA import pretrain as PT

    cfg = copy.deepcopy(PT.DEFAULTS)
    cfg["encoder"] = "cnn1d"
    m = PT.Model(cfg, 0)
    assert isinstance(m.enc, CNN1DEncoder)
    q, z = m.embed(*_batch())
    assert q.shape == (2, 12)


def test_encoder_ablation_axis_registered():
    from Research.QuSpike_JEPA import run as R

    a = R.build_parser().parse_args(["--ablate", "encoder", "--values", "deepsets", "cnn1d"])
    assert [R.make_cfg(b)["encoder"] for b, _ in R.plan(a)] == ["deepsets", "cnn1d"]
