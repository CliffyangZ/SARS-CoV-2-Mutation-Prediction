"""四種基準。"""
import numpy as np
from Research.StyleGAN.utils.baselines import all_baselines


def test_shapes_unique_and_trend_bias():
    L, k = 64, 3
    pred = np.full(L, 1e-6); pred[5] = 1.0                        # 位點 5 斜率（預測頻率）最大
    b = all_baselines(500, k, L, 0, np.full(L, 1 / L), pred)
    assert b["reference"].shape == (500, 0)
    for name in ("random", "freq", "trend"):
        assert b[name].shape == (500, k) and all(len(set(r)) == k for r in b[name])
    assert (b["trend"] == 5).any(1).mean() > 0.95 > (b["random"] == 5).any(1).mean() + 0.3


def test_freq_baseline_follows_train_freq():
    f = np.full(64, 1e-6); f[7] = 1.0
    b = all_baselines(200, 2, 64, 0, f, np.ones(64))
    assert (b["freq"] == 7).any(1).mean() > 0.95
