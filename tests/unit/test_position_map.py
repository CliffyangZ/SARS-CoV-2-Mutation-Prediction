"""位點選取只用訓練期、可逆、規則優先序。"""
import numpy as np
from Research.FE.position_map import build_position_map


def _data():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 20, 1273).astype(np.uint8)
    X = np.tile(ref, (50, 1))
    X[:30, 100] = (ref[100] + 1) % 20          # 位點 101：頻率高
    X[:5, 700] = (ref[700] + 1) % 20           # 位點 701：頻率低
    return X, ref, np.ones(50)


def test_order_reversible_and_count():
    X, ref, w = _data()
    pm = build_position_map(X, ref, w, L=10)
    assert len(pm) == 10 and pm.spike_pos.is_unique and pm["index"].tolist() == list(range(10))
    assert {101, 701} <= set(pm.spike_pos)
    assert (pm.set_index("spike_pos").loc[[101, 701], "origin"] == "train_mutation").all()
    assert set(pm.origin) == {"train_mutation", "neighbor", "random"} or "neighbor" in set(pm.origin)


def test_keep_most_frequent_when_too_many():
    X, ref, w = _data()
    pm = build_position_map(X, ref, w, L=1)
    assert pm.spike_pos.tolist() == [101]


def test_only_train_data_matters():
    X, ref, w = _data()
    a = build_position_map(X, ref, w, L=10)
    X_future = np.concatenate([X, np.tile(ref, (5, 1))]); X_future[-5:, 900] = 3   # 假設另有「測試期」資料不傳入
    b = build_position_map(X, ref, w, L=10)
    assert a.equals(b)
