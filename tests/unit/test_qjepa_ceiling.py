"""標籤歧義上限與譜系群分解（§5 修訂）的單元測試。"""
import numpy as np

from Research.QuSpike_JEPA import ceiling as C
from Research.QuSpike_JEPA import dataset as DS
from Research.QuSpike_JEPA.mutations import UNKNOWN
from tests.unit.test_quspike_jepa import _toy_tokenset


def test_identical_mutation_sets_share_signature():
    """突變集合相同的序列拿到同一個簽章；不同者不同。"""
    ts = _toy_tokenset(n=6)
    ts.pos[1], ts.sym[1], ts.valid[1] = ts.pos[0], ts.sym[0], ts.valid[0]
    sig = C.signatures(ts)
    assert sig[0] == sig[1]


def test_oracle_is_perfect_when_signatures_are_pure():
    """每個簽章只含一個標籤時，oracle 的 macro F1 為 1。"""
    ts = _toy_tokenset(n=40)
    sig = C.signatures(ts)
    # 讓標籤完全由簽章決定
    ts.y[:] = sig % max(len(ts.labels), 1)
    r = C.oracle(ts)
    assert r["f1_macro"] > 0.999 and r["accuracy"] > 0.999


def test_oracle_is_an_upper_bound_on_ambiguous_data():
    """同一簽章帶兩種標籤時，oracle 無法全對，但仍 ≥ 多數類基準。"""
    ts = _toy_tokenset(n=20)
    ts.pos[:], ts.sym[:], ts.valid[:] = ts.pos[0], ts.sym[0], ts.valid[0]  # 全部同一簽章
    ts.y[:10], ts.y[10:] = 0, 1
    r = C.oracle(ts)
    assert r["accuracy"] == 0.5 and r["n_signatures"] == 1


def test_higher_min_count_lowers_the_ceiling():
    """提高 min_count 會丟掉稀有突變，簽章變少、上限不增。"""
    ts = _toy_tokenset(n=80)
    a, b = C.oracle(ts, min_count=1), C.oracle(ts, min_count=10)
    assert b["n_signatures"] <= a["n_signatures"]
    assert b["f1_macro"] <= a["f1_macro"] + 1e-9


def test_signatures_ignore_unknown_symbol():
    """X 不進簽章：只在某條序列加 X 不應改變它的簽章分組。"""
    ts = _toy_tokenset(n=10)
    base = C.signatures(ts)
    j = int(np.where(~ts.valid[0])[0][0]) if (~ts.valid[0]).any() else None
    if j is not None:
        ts.pos[0, j], ts.sym[0, j], ts.valid[0, j] = 999, UNKNOWN, True
        assert C.signatures(ts)[0] == base[0]


def test_group_macro_f1_averages_named_lineages():
    """分群平均只取該群存在的譜系，其餘歸 others。"""
    per_class = [
        {"lineage": "B.1.617.2", "f1": 0.5, "precision": 0, "recall": 0, "support": 1},
        {"lineage": "AY.44", "f1": 0.3, "precision": 0, "recall": 0, "support": 1},
        {"lineage": "AY.3", "f1": 0.5, "precision": 0, "recall": 0, "support": 1},
        {"lineage": "P.1", "f1": 1.0, "precision": 0, "recall": 0, "support": 1},
    ]
    g = C.group_macro_f1(per_class)
    assert abs(g["delta_parent"][0] - 0.5) < 1e-9 and g["delta_parent"][1] == 1
    assert abs(g["delta_sub"][0] - 0.4) < 1e-9 and g["delta_sub"][1] == 2
    assert abs(g["others"][0] - 1.0) < 1e-9


def test_oracle_vocab_can_come_from_train():
    """以 train 的詞彙表評估 test，不洩漏 test 的稀有突變。"""
    tr, te = _toy_tokenset(n=60), _toy_tokenset(n=30, seed=7)
    r = C.oracle(te, min_count=3, train_ts=tr, labels=tr.labels)
    assert 0.0 <= r["f1_macro"] <= 1.0 and r["min_count"] == 3
