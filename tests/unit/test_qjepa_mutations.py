"""突變集合：往返、X 保留、刪除、首尾缺失不算刪除、位點變異度只用傳入資料。"""
import numpy as np

from Research.QuSpike_JEPA import dataset as D
from Research.QuSpike_JEPA import mutations as M


def test_identical_sequence_has_no_mutations():
    assert M.mutations_of(M.REFERENCE) == []


def test_truncated_ends_are_not_deletions():
    """比參考短的首尾視為未觀測，不產生刪除 token。"""
    seq = M.REFERENCE[2:-3]
    assert M.mutations_of(seq) == []


def test_substitution_after_internal_deletion_is_positioned_on_reference():
    seq = M.REFERENCE[:100] + M.REFERENCE[103:300] + "W" + M.REFERENCE[301:]
    got = dict(M.mutations_of(seq))
    assert got[301 - 0] == M.SYM_INDEX["W"] or got.get(301) == M.SYM_INDEX["W"]
    assert all(got[p] == M.DELETION for p in (101, 102, 103))


def test_site_variability_counts_fraction():
    pos = np.array([[10, 20], [10, 0]], np.int16)
    sym = np.array([[1, 2], [1, 0]], np.int8)
    valid = np.array([[True, True], [True, False]])
    f = M.site_variability(pos, sym, valid)
    assert f[10 * D.N_SYM + 1] == 1.0 and f[20 * D.N_SYM + 2] == 0.5
