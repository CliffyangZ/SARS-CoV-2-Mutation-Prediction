"""Spike2Signal 與 RP／GAF／MTF 轉換的單元測試（FR-003、D4、D5）。"""
import numpy as np
import pytest
import torch

from Research.lineage_baselines import data as D
from Research.lineage_baselines import transforms as T


def _sig(b=3, n=50, seed=0):
    """隨機符號序列轉成的訊號 (b, n)。"""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 21, size=(b, n)).astype(np.int8)
    return torch.from_numpy(D.to_signal(ids))


def test_signal_maps_21_symbols_into_range():
    """21 個符號對應到固定、遞增、位於 [-1, 1] 的值。"""
    ids = np.arange(21, dtype=np.int8)[None, :]
    s = D.to_signal(ids)[0]
    assert s[0] == -1 and s[-1] == 1 and np.all(np.diff(s) > 0)
    assert np.allclose(T.signal_from_ids(torch.from_numpy(ids)).numpy()[0], s)


def test_rp_matrix_symmetric_zero_diagonal():
    """RP 距離矩陣對稱、對角線為 0。"""
    m = T.rp_matrix(_sig())
    assert m.shape == (3, 50, 50)
    assert torch.allclose(m, m.transpose(1, 2)) and torch.all(torch.diagonal(m, dim1=1, dim2=2) == 0)


def test_gaf_range_and_symmetry():
    """GAF（summation）值域在 [-1, 1] 且對稱。"""
    g = T.gaf_matrix(_sig())
    assert g.min() >= -1 - 1e-5 and g.max() <= 1 + 1e-5
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-6)


def test_mtf_transition_rows_sum_to_one():
    """MTF 的轉移矩陣中，被走訪過的來源 bin 其列和為 1。"""
    s = _sig(b=2, n=200)
    bins = T.quantize(s, T.MTF_BINS)
    w = T.mtf_transition(bins, T.MTF_BINS)
    visited = torch.zeros(2, T.MTF_BINS, dtype=torch.bool)
    for b in range(2):
        visited[b, bins[b, :-1].unique()] = True
    sums = w.sum(-1)
    assert torch.allclose(sums[visited], torch.ones_like(sums[visited]), atol=1e-5)
    m = T.mtf_matrix(s)
    assert m.shape == (2, 200, 200) and m.min() >= 0 and m.max() <= 1


@pytest.mark.parametrize("kind", ["rp", "gaf", "mtf"])
def test_image_shape_and_determinism(kind):
    """輸出為 3 通道 360×360，數值在 [0, 1]，重複產生結果相同。"""
    s = _sig(b=2, n=80)
    a, b = T.to_image(kind, s), T.to_image(kind, s)
    assert a.shape == (2, 3, 360, 360)
    assert a.min() >= 0 and a.max() <= 1 and torch.equal(a, b)


def test_different_sequences_give_different_images():
    """不同序列產生不同影像（避免轉換退化成常數）。"""
    s = _sig(b=2, n=80)
    img = T.to_image("rp", s)
    assert not torch.equal(img[0], img[1])


def test_optional_pyts_reference():
    """若環境有 pyts，RP 與參考實作的距離矩陣一致（沒有就略過）。"""
    pyts = pytest.importorskip("pyts.image")
    s = _sig(b=1, n=40)
    ref = pyts.RecurrencePlot(dimension=1, time_delay=1, threshold=None).transform(s.numpy())[0]
    assert np.allclose(T.rp_matrix(s)[0].numpy(), ref, atol=1e-5)


def test_signal_mode_cumsum():
    """cumsum 模式：範圍為 [-1, 1]、保持形狀、為去均值累積和的線性重新縮放；raw 不變；未知模式報錯。"""
    import pytest

    ids = torch.randint(0, 21, (3, 50), dtype=torch.int8)
    raw = T.signal_from_ids(ids)
    assert torch.equal(T.apply_signal_mode(raw, "raw"), raw)
    c = T.apply_signal_mode(raw, "cumsum")
    assert c.shape == raw.shape
    assert torch.allclose(c.min(1).values, torch.full((3,), -1.0), atol=1e-5)
    assert torch.allclose(c.max(1).values, torch.full((3,), 1.0), atol=1e-5)
    cs = torch.cumsum(raw - raw.mean(1, keepdim=True), 1)  # 去均值後累積
    expect = 2 * (cs - cs.min(1, keepdim=True).values) / (cs.max(1, keepdim=True).values - cs.min(1, keepdim=True).values) - 1
    assert torch.allclose(c, expect, atol=1e-5)  # 只是累積和的線性重新縮放
    with pytest.raises(ValueError):
        T.apply_signal_mode(raw, "nope")
    flat = T.apply_signal_mode(torch.zeros(1, 5), "cumsum")  # 常數訊號不應產生 NaN
    assert torch.isfinite(flat).all()


def test_pyts_reference_gaf_and_mtf():
    """GAF 與 MTF（均勻分箱）和 pyts 一致（沒有 pyts 就略過）。以 cumsum 訊號比對，避免分箱邊界上的離散值差異。"""
    import pytest

    pyts = pytest.importorskip("pyts.image")
    ids = torch.randint(0, 21, (1, 200), dtype=torch.int8)
    s = T.apply_signal_mode(T.signal_from_ids(ids), "cumsum")
    ref = pyts.GramianAngularField(method="summation").transform(s.numpy())[0]
    assert abs(ref - T.gaf_matrix(s)[0].numpy()).max() < 1e-5
    ref = pyts.MarkovTransitionField(n_bins=T.MTF_BINS, strategy="uniform").transform(s.numpy())[0]
    assert abs(ref - T.mtf_matrix(s)[0].numpy()).max() < 1e-5
