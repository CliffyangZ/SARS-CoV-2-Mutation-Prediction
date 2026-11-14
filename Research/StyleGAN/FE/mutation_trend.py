"""每位點的月頻率與訓練期線性斜率（notebook 2.1）。

斜率只用訓練期各月（樣本 ≥30 的月份）；也供基準 4（趨勢外推）使用。
"""
import numpy as np
import pandas as pd


def monthly_mut_freq(X_all, ref, month_idx, n_months):
    """X_all: (n,1273) 原始序列；month_idx: 每條所屬月序號 → (n_months, 1273) 突變頻率與各月樣本數。"""
    is_mut = ((X_all != ref[None, :]) & (X_all != 20)).astype(np.float32)
    n_m = np.bincount(month_idx, minlength=n_months)
    F = np.zeros((n_months, X_all.shape[1]))
    for m in range(n_months):
        if n_m[m]:
            F[m] = is_mut[month_idx == m].mean(0)
    return F, n_m


def train_trend(F, n_m, train_months, min_n=30):
    """訓練期月頻率對月序號做線性回歸，回傳 (斜率, 訓練期最後月頻率, 外推到測試期起點月的頻率)。"""
    use = [m for m in train_months if n_m[m] >= min_n]
    t = np.array(use, float)
    slope = np.array([np.polyfit(t, F[use, i], 1)[0] for i in range(F.shape[1])]) if len(use) >= 2 \
        else np.zeros(F.shape[1])
    last = F[use[-1]] if use else np.zeros(F.shape[1])
    return slope, last


def trend_table(spike_pos, slope, last, pred):
    """整理成 trend.csv 的欄位。"""
    return pd.DataFrame({"spike_pos": spike_pos, "slope_per_month": slope, "freq_end_of_train": last,
                         "pred_freq_test_start": pred})
