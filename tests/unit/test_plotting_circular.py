"""圓形頻率圖：四段邊界與最高頻區段。"""
import numpy as np
from Research.utils import plotting
from Research.utils.spike_regions import REGIONS, SEGMENTS, segment_of


def test_segments_and_plot(tmp_path):
    assert SEGMENTS == [(1, 318), (319, 636), (637, 954), (955, 1273)]
    assert [segment_of(p) for p in (1, 318, 319, 636, 637, 954, 955, 1273)] == [0, 0, 1, 1, 2, 2, 3, 3]
    f = np.zeros(1273); f[400:500] = 0.5                       # 第二段頻率最高
    freqs = {"train": f * 0.5, "test": f, "generated": f}
    assert plotting.segment_freqs(f, SEGMENTS).argmax() == 1
    plotting.circular_frequency_plot(freqs, SEGMENTS, REGIONS, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0
    assert plotting.peak_table(freqs, 3).shape == (9, 3)
