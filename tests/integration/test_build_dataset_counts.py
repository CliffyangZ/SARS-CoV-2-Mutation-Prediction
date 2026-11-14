"""真實資料數量對照論文 / notebook（資料不存在則跳過）。"""
import numpy as np
import pytest
from Research.StyleGAN.FE import load_spike as ls
from Research.StyleGAN.FE.build_dataset import PROCESSED, build


@pytest.mark.slow
def test_real_counts(tmp_path):
    try:
        ls.default_fasta()
    except FileNotFoundError:
        pytest.skip("無資料")
    r = build(1, out_dir=tmp_path)
    assert r["problems"] == []
    assert r["counts"] == dict(raw=137132, filtered=132301, unique=13638)
    assert (r["split"]["train"], r["split"]["test"]) == (8161, 3066)
    v = np.load(tmp_path / "vectors.npz")
    assert v["X"].shape[1] == 64 and set(np.unique(v["X"])) <= {0, 1}


@pytest.mark.slow
def test_encodings_dataset(tmp_path):
    """legacy 的向量與既有 aa_scale1 逐元素相同；drop_const 排除 614 並在報告記錄排除位點與全零向量（T019）。"""
    try:
        ls.default_fasta()
    except FileNotFoundError:
        pytest.skip("無資料")
    old = PROCESSED / "aa_scale1" / "vectors.npz"
    leg = build(1, out_dir=tmp_path / "legacy", encoding="legacy")
    if old.exists():
        assert np.array_equal(np.load(old)["X"], np.load(tmp_path / "legacy" / "vectors.npz")["X"])
    dc = build(1, out_dir=tmp_path / "dc", encoding="drop_const")
    import pandas as pd
    pm = pd.read_csv(tmp_path / "dc" / "position_map.csv")
    ex = pd.read_csv(tmp_path / "dc" / "excluded_positions.csv")
    assert 614 in ex.spike_pos.tolist() and 614 not in pm.spike_pos.tolist() and len(pm) == 64
    rep = (tmp_path / "dc" / "dataset_report.md").read_text()
    assert "排除位點" in rep and "614" in rep and "全零突變向量" in rep
    assert dc["zero"]["train"] >= leg["zero"]["train"]
    fo = build(1, out_dir=tmp_path / "fo", encoding="freq_order")
    assert pd.read_csv(tmp_path / "fo" / "position_map.csv").train_freq.is_monotonic_decreasing
