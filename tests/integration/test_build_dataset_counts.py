"""真實資料數量對照論文 / notebook（資料不存在則跳過）。"""
import numpy as np
import pytest
from Research.FE import load_spike as ls
from Research.FE.build_dataset import PROCESSED, build


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
