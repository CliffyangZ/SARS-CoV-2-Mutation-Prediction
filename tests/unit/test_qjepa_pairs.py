"""配對：長尾退回、鄰近用突變距離與日期窗、跨譜系統計。"""
import numpy as np

from Research.QuSpike_JEPA import dataset as D
from Research.QuSpike_JEPA import pairs as P
from Research.QuSpike_JEPA import views as V


def _ts(pos_sets, y, months):
    T = max(len(p) for p in pos_sets)
    n = len(y)
    pos = np.zeros((n, T), np.int16); valid = np.zeros((n, T), bool)
    for i, p in enumerate(pos_sets):
        pos[i, :len(p)] = p; valid[i, :len(p)] = True
    return D.TokenSet(pos, np.zeros((n, T), np.int8), valid, np.array(y), np.array(months, np.int32), ["a", "b"])


def test_variant_aliases():
    assert P.canonical_variant("same_lineage") == "VA" and P.canonical_variant("neighbor") == "VB"


def test_singleton_lineage_falls_back_to_self_and_is_counted():
    ts = _ts([[10, 20], [10, 30], [50, 60]], [0, 0, 1], [24000] * 3)
    vm = V.ViewMaker(ts, variant="VA", seed=0)
    assert vm.partner(2) == 2 and vm.stats["fallback_self"] == 1


def test_neighbor_uses_distance_and_date_window_not_labels():
    ts = _ts([[10, 20], [10, 20], [10, 21], [10, 20], [300, 400]], [0, 1, 0, 0, 0], [24000, 24000, 24000, 24010, 24000])
    nb = P.neighbor_lists(ts, max_dist=2, max_months=1)
    assert set(nb[0].tolist()) == {1, 2}  # 3 超出日期窗、4 距離太遠；1 為不同譜系仍可成為鄰居


def test_cross_lineage_ratio_is_counted():
    ts = _ts([[10, 20], [10, 20], [10, 20]], [0, 1, 1], [24000] * 3)
    vm = V.ViewMaker(ts, variant="VB", seed=0)
    vm.batch(np.arange(3))
    assert vm.stats["pairs"] == 3 and vm.stats["cross_lineage"] >= 1
