"""譜系資料載入、共同長度、截斷與編碼的單元測試（US1 前置，FR-006）。"""
import numpy as np
import pandas as pd
import pytest

from Research.lineage_baselines import data as D


def _make_df(train_lens, test_lens, lineages=("A", "B")):
    """建立合成資料表：序列內容由長度與譜系決定，含一個 X。"""
    rows = []
    for split, lens in (("train", train_lens), ("test", test_lens)):
        for i, n in enumerate(lens):
            lin = lineages[i % len(lineages)]
            seq = ("ACDEFGHIKLMNPQRSTVWY" * 100)[: n - 1] + "X"
            rows.append(dict(lineage=lin, accession=f"{split}{i}", date="2021", count=1, split=split, sequence=seq))
    return pd.DataFrame(rows)


def test_common_length_covers_999_per_mille():
    """L 至少讓 99.9% 的 train 序列不短於它，且是這樣的最大值。"""
    lens = [100] * 2000 + [90] * 1 + [80] * 1  # 2002 條，容許約 2 條更短
    L = D.common_length(lens, coverage=0.999)
    assert sum(1 for n in lens if n < L) <= int(len(lens) * 0.001)
    assert L == 100


def test_short_sequences_excluded_and_counted():
    """比 L 短的序列被排除，train 與 test 各自計數。"""
    df = _make_df([100] * 3000 + [50], [100] * 10 + [60, 60])
    p = D.prepare_from_frame(df)
    assert p.length == 100
    assert p.n_excluded_train == 1
    assert p.n_excluded_test == 2
    assert p.x_train.shape == (3000, 100) and p.x_test.shape == (10, 100)


def test_x_is_21st_symbol_not_dropped():
    """X 保留為第 21 個符號。"""
    assert len(D.ALPHABET) == 21 and "X" in D.ALPHABET
    df = _make_df([30] * 5, [30] * 2)
    p = D.prepare_from_frame(df)
    assert (p.x_train == D.ALPHABET.index("X")).any()
    assert p.x_train.shape[0] == 5


def test_test_content_does_not_change_L():
    """改動 test 序列（長度與內容）不改變 L 與 train 資料。"""
    base = _make_df([100] * 1500, [100] * 5)
    other = _make_df([100] * 1500, [20] * 5)
    p1, p2 = D.prepare_from_frame(base), D.prepare_from_frame(other)
    assert p1.length == p2.length
    assert np.array_equal(p1.x_train, p2.x_train)


def test_label_encoding_sorted_and_stable():
    """標籤依譜系名稱排序編碼，兩次載入一致。"""
    df = _make_df([30] * 6, [30] * 4, lineages=("B.1", "A.1", "C"))
    p = D.prepare_from_frame(df)
    assert p.labels == ["A.1", "B.1", "C"]
    assert set(p.y_train.tolist()) <= {0, 1, 2}
    assert D.prepare_from_frame(df).labels == p.labels


def test_onehot_and_signal_shapes():
    """one-hot 稀疏矩陣與訊號的形狀、值域正確。"""
    df = _make_df([30] * 4, [30] * 2)
    p = D.prepare_from_frame(df)
    oh = D.one_hot_sparse(p.x_train)
    assert oh.shape == (4, 30 * 21)
    assert np.allclose(np.asarray(oh.sum(axis=1)).ravel(), 30)
    s = D.to_signal(p.x_train)
    assert s.shape == (4, 30) and s.min() >= -1 and s.max() <= 1


def test_limit_is_deterministic_subsample():
    """--limit 以固定種子隨機抽樣，且同時保留 train 與 test。"""
    df = _make_df([40] * 400, [40] * 200)
    a = D.prepare_from_frame(df, limit=100)
    b = D.prepare_from_frame(df, limit=100)
    assert a.x_train.shape[0] + a.x_test.shape[0] <= 100
    assert np.array_equal(a.x_train, b.x_train)
    assert a.x_test.shape[0] > 0


def test_is_complete_respects_recipe_version(tmp_path):
    """舊版配方（無 recipe_version 或版本較小）的深度結果視為未完成，需重新評估。"""
    import json

    from Research.lineage_baselines import results_io as R

    p = tmp_path / "a.json"
    p.write_text(json.dumps({"status": "complete"}))
    assert R.is_complete(p) and not R.is_complete(p, min_version=2)
    p.write_text(json.dumps({"status": "complete", "recipe_version": 2}))
    assert R.is_complete(p, min_version=2)
