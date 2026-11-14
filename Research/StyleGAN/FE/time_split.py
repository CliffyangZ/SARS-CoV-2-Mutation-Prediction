"""時間切分：訓練期 / 空白期 / 測試期（以 first_date 判定，FR-002）。"""
import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2021-02-28")
TEST_START = pd.Timestamp("2021-04-01")


def split_labels(first_date, train_end=TRAIN_END, test_start=TEST_START):
    """回傳字串陣列：train（≤train_end）、test（≥test_start）、其餘 gap。"""
    fd = pd.to_datetime(pd.Series(first_date)).values
    out = np.full(len(fd), "gap", dtype=object)
    out[fd <= np.datetime64(train_end)] = "train"
    out[fd >= np.datetime64(test_start)] = "test"
    return out
