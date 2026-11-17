"""突變集合的陣列資料集（TokenSet）與資料載入、validation 切分。"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mutations import ROOT, SYMBOLS, mutations_of  # noqa: F401  ROOT 供測試取用

MASK_SYM = len(SYMBOLS)  # 遮蔽符號
N_SYM = len(SYMBOLS) + 1
DEFAULT_DATA = ROOT / "Data" / "spike-lineage" / "dataset.tsv"
CACHE_DIR = ROOT / "Results" / "quspike_jepa"
VAL_FRACTION = 0.15
RULE_VERSION = "val15-latest-per-lineage-v1"


@dataclass
class TokenSet:
    """每列一條序列的突變集合，補齊到 T 個 token。month=-1 表示無日期。"""

    pos: np.ndarray  # (n, T) int16，1 起算
    sym: np.ndarray  # (n, T) int8
    valid: np.ndarray  # (n, T) bool
    y: np.ndarray  # (n,)
    month: np.ndarray  # (n,) int32，year*12+month-1
    labels: list

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        """取子集（保留標籤表）。"""
        idx = np.asarray(idx)
        return TokenSet(self.pos[idx], self.sym[idx], self.valid[idx], self.y[idx], self.month[idx], self.labels)


def parse_month(d):
    """日期字串 → 月序號；只有年者以該年 1 月，缺值回 -1。"""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return -1
    p = str(d).split("-")
    try:
        y = int(p[0])
        m = int(p[1]) if len(p) > 1 else 1
    except ValueError:
        return -1
    return y * 12 + m - 1


def full_date_mask(dates):
    """回傳哪些序列有完整年月日（供親緣鄰近的日期窗）。"""
    return np.array([len(str(d)) == 10 for d in dates])


def build_tokenset(seqs, y, dates, labels):
    """由序列字串建立 TokenSet。"""
    muts = [mutations_of(s) for s in seqs]
    T = max(1, max(len(m) for m in muts))
    n = len(muts)
    pos = np.zeros((n, T), np.int16)
    sym = np.zeros((n, T), np.int8)
    valid = np.zeros((n, T), bool)
    for i, m in enumerate(muts):
        for j, (p, s) in enumerate(m):
            pos[i, j], sym[i, j], valid[i, j] = p, s, True
    return TokenSet(pos, sym, valid, np.asarray(y), np.array([parse_month(d) for d in dates], np.int32), list(labels))


def file_hash(path):
    """資料檔雜湊（split_version 的一部分）。"""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:12]


def load_splits(path=DEFAULT_DATA, limit=None, cache=True):
    """讀 dataset.tsv，回傳 (train, test, split_version)。快取突變集合以免重複比對。"""
    import pandas as pd

    path = Path(path)
    fh = file_hash(path)
    cache_f = CACHE_DIR / f"tokens_{fh}.npz"
    df = pd.read_csv(path, sep="\t", dtype={"date": str})
    labels = sorted(df.lineage.unique())
    idx = {l: i for i, l in enumerate(labels)}
    out = {}
    for sp in ("train", "test"):
        d = df[df.split == sp].reset_index(drop=True)
        if limit:
            d = d.sample(n=min(limit, len(d)), random_state=0).reset_index(drop=True) if sp == "train" else d.head(limit)
        out[sp] = d
    key = f"{fh}_{limit}"
    cache_f = CACHE_DIR / f"tokens_{key}.npz"
    if cache and cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        sets = {sp: TokenSet(z[f"{sp}_pos"], z[f"{sp}_sym"], z[f"{sp}_valid"], z[f"{sp}_y"], z[f"{sp}_month"], labels) for sp in out}
    else:
        sets = {sp: build_tokenset(d.sequence.tolist(), d.lineage.map(idx).values, d.date.tolist(), labels) for sp, d in out.items()}
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_f, **{f"{sp}_{k}": getattr(ts, k) for sp, ts in sets.items() for k in ("pos", "sym", "valid", "y", "month")})
    return sets["train"], sets["test"], f"{fh}+{RULE_VERSION}"


def split_validation(ts, frac=VAL_FRACTION):
    """train 內每譜系日期最晚的 frac 為 validation；無日期者不進 validation。回傳 (fit_idx, val_idx)。"""
    val = np.zeros(len(ts), bool)
    for k in np.unique(ts.y):
        ids = np.where((ts.y == k) & (ts.month >= 0))[0]
        if len(ids) == 0:
            continue
        n_val = max(1, int(round(len(ids) * frac))) if len(ids) >= 2 else 0
        order = ids[np.argsort(ts.month[ids], kind="stable")]
        val[order[len(order) - n_val:]] = True
    return np.where(~val)[0], np.where(val)[0]


N_TOK = 1300 * N_SYM  # token id = pos·N_SYM + sym 的上界（N_POS=1273）


def mutation_vocab(ts):
    """由 ts（應傳 train）建立突變詞彙表：排除 X 與 MASK。回傳 (lut, V)；lut[token id] = 索引或 -1。"""
    from .mutations import UNKNOWN

    keep = ts.valid & (ts.sym != UNKNOWN) & (ts.sym != MASK_SYM)
    tok = np.unique(ts.pos[keep].astype(np.int64) * N_SYM + ts.sym[keep].astype(np.int64))
    lut = np.full(N_TOK, -1, np.int64)
    lut[tok] = np.arange(len(tok))
    return lut, len(tok)
