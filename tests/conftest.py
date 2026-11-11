"""測試共用 fixture：固定種子、小型合成 FASTA 與突變矩陣。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 讓 `import Research` 可用


@pytest.fixture(autouse=True)
def _seed():
    """每個測試前固定隨機種子。"""
    np.random.seed(0)
    import torch
    torch.manual_seed(0)


REF = "ACDEFGHIKL" * 2  # 20 aa 的合成參考序列（測試用，非 1273）


@pytest.fixture
def synth_fasta(tmp_path):
    """寫出一個小型 FASTA：含正常、過短、日期壞掉、含終止碼、過多 X 與重複序列。"""
    def mut(s, pos, aa):
        return s[:pos] + aa + s[pos + 1:]
    recs = [
        ("REF.1", "China", "2019-12", REF),
        ("A1", "USA", "2020-03-01", mut(REF, 3, "A")),
        ("A2", "UK", "2020-05-10", mut(REF, 3, "A")),       # 與 A1 重複
        ("A3", "UK", "2021-05-10", mut(REF, 5, "W")),
        ("BAD_LEN", "USA", "2020-03-01", REF[:-1]),
        ("BAD_DATE", "USA", "xx", REF),
        ("STOP", "USA", "2020-03-01", mut(REF, 2, "*")),
        ("MANYX", "USA", "2020-03-01", "XX" + REF[2:]),     # 20 aa 中 2 個 X = 10% > 1%
    ]
    p = tmp_path / "s.fasta"
    p.write_text("".join(f">{a} |{c}|{d}\n{s}\n" for a, c, d, s in recs))
    return p


@pytest.fixture
def synth_dataset(tmp_path):
    """小型合成資料集目錄（L=64，1273 位點整數編碼），欄位同 build_dataset 的輸出。"""
    import pandas as pd
    rng = np.random.default_rng(0)
    n, L = 120, 64
    ref = rng.integers(0, 20, 1273).astype(np.uint8)
    X_int = np.tile(ref, (n, 1))
    sites = np.sort(rng.choice(1273, L, replace=False))
    for r in range(n):
        for s in rng.choice(sites[:20], 3, replace=False):
            X_int[r, s] = (ref[s] + 1 + r % 3) % 20
    split = np.array(["train"] * 60 + ["gap"] * 10 + ["test"] * 50)
    X = ((X_int[:, sites] != ref[sites]) & (X_int[:, sites] != 20)).astype(np.uint8)
    lin = np.array(["Wuhan-like (D614)"] * 100 + ["Delta-like (L452R+T478K)"] * 20)
    np.savez(tmp_path / "vectors.npz", X=X, X_int=X_int, count=rng.integers(1, 5, n), split=split,
             lineage_proxy=lin, ref=ref, first_date=np.zeros(n, dtype="datetime64[D]"))
    pd.DataFrame({"index": range(L), "spike_pos": sites + 1, "origin": "train_mutation",
                  "train_freq": rng.random(L) * 0.1}).to_csv(tmp_path / "position_map.csv", index=False)
    pd.DataFrame({"spike_pos": range(1, 1274), "slope_per_month": rng.random(1273) * 0.01,
                  "freq_end_of_train": 0.0, "pred_freq_test_start": rng.random(1273) * 0.1}).to_csv(tmp_path / "trend.csv", index=False)
    return tmp_path
