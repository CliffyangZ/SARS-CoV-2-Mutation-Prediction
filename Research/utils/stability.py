"""訓練穩定性指標（research R14）：後段標準差、尖峰數、收斂步數。"""
import numpy as np


def late_std(loss):
    """後 50% 步數的損失標準差。"""
    loss = np.asarray(loss, float)
    return float(loss[len(loss) // 2:].std())


def spike_count(loss, window=5, k=3.0):
    """損失偏離滾動中位數超過 k×MAD 的次數。"""
    loss = np.asarray(loss, float); n = 0
    for i in range(len(loss)):
        w = loss[max(0, i - window): i + window + 1]
        med = np.median(w); mad = np.median(np.abs(w - med))
        if mad > 0 and abs(loss[i] - med) > k * mad:
            n += 1
    return n


def convergence_step(loss, steps=None, window=5, frac=0.1):
    """第一個「其後滾動標準差（窗 5 點）皆低於 初期標準差×10%」的步數；沒有則回傳 None（未收斂）。"""
    loss = np.asarray(loss, float)
    steps = np.arange(len(loss)) if steps is None else np.asarray(steps)
    thr = frac * loss[:window].std()
    if len(loss) < window or thr <= 0:
        return None
    roll = np.array([loss[i:i + window].std() for i in range(len(loss) - window + 1)])
    ok = roll < thr
    for i in range(len(ok)):
        if ok[i:].all():
            return int(steps[i])
    return None


def stability_report(loss, steps=None):
    """整合三個指標。"""
    return dict(late_std=late_std(loss), spikes=spike_count(loss), converge_step=convergence_step(loss, steps))
