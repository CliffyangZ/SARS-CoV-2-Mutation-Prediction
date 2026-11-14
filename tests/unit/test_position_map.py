"""位點選取只用訓練期、可逆、規則優先序。"""
import numpy as np
from Research.StyleGAN.FE.position_map import build_position_map


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


# ---------------- 002：排除近常數位點與依頻率排列（T018） ----------------
from Research.StyleGAN.FE.position_map import excluded_positions


def _data_const():
    """位點 101（index 100）訓練期頻率 0.95（近常數），位點 201 為 0.4，位點 301 為 0.1。"""
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 20, 1273).astype(np.uint8)
    X = np.tile(ref, (100, 1))
    X[:95, 100] = (ref[100] + 1) % 20
    X[:40, 200] = (ref[200] + 1) % 20
    X[:10, 300] = (ref[300] + 1) % 20
    return X, ref, np.ones(100)


def test_drop_const_excludes_and_refills():
    X, ref, w = _data_const()
    pm = build_position_map(X, ref, w, L=10, drop_const_threshold=0.9)
    ex = excluded_positions(X, ref, w, 0.9)
    assert ex.spike_pos.tolist() == [101] and ex.reason.tolist() == ["excluded_const"]
    assert 101 not in set(pm.spike_pos) and len(pm) == 10 and pm.spike_pos.is_unique
    assert {201, 301} <= set(pm.spike_pos)
    assert set(pm[~pm.spike_pos.isin([201, 301])].origin) <= {"neighbor", "random"}   # 補位來源有記錄（C2）
    assert not set(ex.spike_pos) & set(pm.spike_pos)
    # legacy（不排除）仍含 101，且與舊行為相同
    assert 101 in set(build_position_map(X, ref, w, L=10).spike_pos)


def test_freq_order_sorted_by_train_frequency_and_reversible():
    X, ref, w = _data_const()
    pm = build_position_map(X, ref, w, L=5, order="freq")
    assert pm.spike_pos.tolist()[:3] == [101, 201, 301]                  # 頻率 0.95、0.4、0.1
    assert pm.train_freq.is_monotonic_decreasing and pm["index"].tolist() == list(range(5))
    assert pm.spike_pos.is_unique                                         # 索引→spike 位置一對一，可還原


def test_new_options_use_only_train_data():
    """改動「測試期」資料（不傳入）不影響結果；兩個選項的輸出只由傳入的訓練資料決定。"""
    X, ref, w = _data_const()
    a = build_position_map(X, ref, w, L=8, drop_const_threshold=0.9, order="freq")
    Xf = np.concatenate([X, np.tile(ref, (20, 1))]); Xf[-20:, 900] = 3
    b = build_position_map(X, ref, w, L=8, drop_const_threshold=0.9, order="freq")
    assert a.equals(b)
