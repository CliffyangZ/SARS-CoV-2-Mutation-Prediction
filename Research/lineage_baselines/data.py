"""譜系資料集載入與前處理：共同長度、截斷、one-hot 與 Spike2Signal 編碼。

規則（規格 FR-006、研究文件 D3）：
- 共同長度 L 只由 train 切分決定，至少 99.9% 的 train 序列不短於 L。
- 截斷到 L，不填充；比 L 短的序列排除並分別計數。
- 未知殘基 X 保留為第 21 個符號。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ALPHABET = tuple(sorted("ACDEFGHIKLMNPQRSTVWY" + "X"))  # 21 個符號，固定字母順序
_INDEX = {c: i for i, c in enumerate(ALPHABET)}
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "Data" / "spike-lineage" / "dataset.tsv"


@dataclass
class Prepared:
    """前處理後的資料：整數編碼矩陣、標籤與排除統計。"""

    x_train: np.ndarray  # (n, L) int8，值為 ALPHABET 的索引
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels: list  # 譜系名稱，索引即標籤編號
    length: int  # 共同長度 L
    n_excluded_train: int
    n_excluded_test: int

    def preprocessing_info(self):
        """回傳寫入結果檔的前處理描述。"""
        return {
            "common_length": self.length,
            "n_excluded_train": self.n_excluded_train,
            "n_excluded_test": self.n_excluded_test,
            "x_symbol": True,
            "n_train": int(len(self.y_train)),
            "n_test": int(len(self.y_test)),
            "label_map": {name: i for i, name in enumerate(self.labels)},
        }


def common_length(lengths, coverage=0.999):
    """回傳最大的 L，使不短於 L 的序列比例至少為 coverage。"""
    s = np.sort(np.asarray(lengths))
    allowed_short = int(np.floor(len(s) * (1.0 - coverage) + 1e-9))
    return int(s[allowed_short])


def encode(seqs, length):
    """把序列字串截斷到 length 並編成 int8 索引矩陣（呼叫端需先確保長度足夠）。"""
    out = np.empty((len(seqs), length), dtype=np.int8)
    for i, s in enumerate(seqs):
        out[i] = [_INDEX[c] for c in s[:length]]
    return out


def prepare_from_frame(df, limit=None, seed=0, coverage=0.999):
    """由資料表產生 Prepared；limit 以固定種子隨機抽樣（冒煙與測試用）。"""
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    labels = sorted(df["lineage"].unique())
    lab = {n: i for i, n in enumerate(labels)}
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    L = common_length(tr["sequence"].str.len().to_numpy(), coverage)
    tr_ok, te_ok = tr[tr["sequence"].str.len() >= L], te[te["sequence"].str.len() >= L]
    return Prepared(
        x_train=encode(tr_ok["sequence"].tolist(), L),
        y_train=tr_ok["lineage"].map(lab).to_numpy(dtype=np.int64),
        x_test=encode(te_ok["sequence"].tolist(), L),
        y_test=te_ok["lineage"].map(lab).to_numpy(dtype=np.int64),
        labels=labels,
        length=L,
        n_excluded_train=len(tr) - len(tr_ok),
        n_excluded_test=len(te) - len(te_ok),
    )


def load_prepared(path=DEFAULT_DATA, limit=None, seed=0):
    """讀取 dataset.tsv 並前處理。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔：{path}")
    df = pd.read_csv(path, sep="\t", dtype={"date": str})
    return prepare_from_frame(df, limit=limit, seed=seed)


def one_hot_sparse(ids):
    """把 (n, L) 索引矩陣轉成 (n, L*21) 的稀疏 one-hot（CSR）。"""
    n, L = ids.shape
    cols = (np.arange(L)[None, :] * len(ALPHABET) + ids).ravel()
    rows = np.repeat(np.arange(n), L)
    return sp.csr_matrix((np.ones(n * L, dtype=np.float32), (rows, cols)), shape=(n, L * len(ALPHABET)))


def to_signal(ids):
    """Spike2Signal：符號依固定字母順序對應 1–21，再線性映到 [-1, 1]（研究文件 D5）。"""
    ints = ids.astype(np.float32) + 1.0
    return ((ints - 1.0) / (len(ALPHABET) - 1) * 2.0 - 1.0).astype(np.float32)
