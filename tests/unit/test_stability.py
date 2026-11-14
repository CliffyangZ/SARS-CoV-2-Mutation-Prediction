"""穩定性指標在合成曲線上的行為。"""
import numpy as np
from Research.StyleGAN.utils.stability import convergence_step, late_std, spike_count


def test_oscillating_vs_converging():
    t = np.arange(47)
    osc = np.sin(t) * 2
    conv = 2 * np.exp(-t / 5) * np.cos(t) + 0.0
    assert late_std(osc) > late_std(conv)
    assert convergence_step(osc) is None
    assert convergence_step(conv) is not None


def test_spikes():
    flat = np.random.RandomState(0).randn(47) * 0.01
    sp = flat.copy(); sp[10] += 5; sp[30] -= 5
    assert spike_count(sp) >= 2 and spike_count(sp) > spike_count(flat)
