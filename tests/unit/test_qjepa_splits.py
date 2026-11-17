"""validation 切分：每譜系日期最晚 15%、無日期不入 validation、不與其餘重疊。"""
import numpy as np

from Research.QuSpike_JEPA import dataset as D


def _ts(months, y):
    n = len(y)
    return D.TokenSet(np.zeros((n, 1), np.int16), np.zeros((n, 1), np.int8), np.ones((n, 1), bool), np.array(y),
                      np.array(months, np.int32), ["a", "b"])


def test_latest_fraction_per_lineage():
    months = list(range(100, 120)) + list(range(100, 120))
    y = [0] * 20 + [1] * 20
    fit, val = D.split_validation(_ts(months, y))
    assert len(val) == 6 and set(fit).isdisjoint(val)
    for k in (0, 1):
        v = [i for i in val if y[i] == k]
        assert min(months[i] for i in v) > max(months[i] for i in fit if y[i] == k)


def test_undated_never_in_validation():
    months = [-1] * 5 + list(range(10, 25))
    fit, val = D.split_validation(_ts(months, [0] * 20))
    assert all(months[i] >= 0 for i in val)


def test_parse_month():
    assert D.parse_month("2021-03-05") == 2021 * 12 + 2 and D.parse_month("2021") == 2021 * 12
    assert D.parse_month(None) == -1 and D.parse_month("") == -1
