"""天花板報告（Q5，FR-006／SC-004）。以已知衝突比例的合成資料驗證。"""
import numpy as np
import pytest

from Research.QuSpike_PQK import ceiling, features as F


def _fs():
    """6 個類別（0,1 = 群 A；2..5 = 其他）。群 A 測試列：
    簽章 s1 ×4（標籤 0,0,0,1 → 衝突）、簽章 s2 ×2（標籤 1,1 → 不衝突）。"""
    s1, s2, s3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    x_te = np.array([s1, s1, s1, s1, s2, s2, s3, s3], np.float32)
    y_te = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    x_tr = np.array([s1, s1, s2, s2, s3], np.float32)
    y_tr = np.array([0, 1, 1, 1, 2])
    return F.FeatureSet(source="t", x_train=x_tr, x_test=x_te, y_train=y_tr, y_test=y_te,
                        labels=[f"L{i}" for i in range(6)], data_hash="h", split_version="v",
                        selection_reason="t", ood_fraction=0.0)


def _groups(fs):
    tr = np.where(np.isin(fs.y_train, [0, 1]))[0]
    te = np.where(np.isin(fs.y_test, [0, 1]))[0]
    return [F.Group("delta", (0, 1), len(tr), len(te), True, tr, te)]


def test_conflict_fraction_and_upper_bound():
    fs = _fs()
    r = ceiling.ceiling_report(fs, _groups(fs))["delta"]
    # 測試群內 6 列：s1 的 4 列衝突 → 4/6
    assert r["conflict_fraction_test"] == pytest.approx(4 / 6)
    assert r["n_signatures_test"] == 2 and r["n_conflicting_signatures_test"] == 1
    # 訓練群內 4 列：s1 (0,1) 衝突 2 列 → 2/4
    assert r["conflict_fraction_train"] == pytest.approx(0.5)
    # oracle：s1 猜 0（錯 1 筆）、s2 猜 1（全對）→ 最少錯誤 1
    assert r["oracle_errors"] == 1
    assert r["upper_bound_is_optimistic"] is True
    assert 0.0 < r["macro_f1_upper_bound"] < 1.0


def test_unavoidable_fraction_uses_c0_errors():
    fs = _fs()
    pc = [dict(lineage="L0", recall=0.5, support=3, precision=0, f1=0),
          dict(lineage="L1", recall=0.5, support=3, precision=0, f1=0)]
    r = ceiling.ceiling_report(fs, _groups(fs), pc)["delta"]
    # C0 錯 round(3*.5)+round(3*.5)=4 筆；oracle 最少 1 筆 → 25%
    assert r["unavoidable_error"]["fraction"] == pytest.approx(1 / 4)


def test_confusions_report_pair():
    fs = _fs()
    r = ceiling.ceiling_report(fs, _groups(fs))["delta"]
    assert r["confusions"]["L1"]["confused_with"] == "L0"


def test_markdown_mentions_optimistic():
    fs = _fs()
    rep = dict(split_version="v", ood_mutation_fraction=0.0, groups=ceiling.ceiling_report(fs, _groups(fs)))
    assert "樂觀上限" in ceiling.to_markdown(rep)
