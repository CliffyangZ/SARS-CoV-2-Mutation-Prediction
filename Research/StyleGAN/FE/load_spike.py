"""讀 spike 胺基酸 FASTA、品質過濾、去重（重現 notebook 1.1–1.2，FR-001）。"""
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

REF_ID = "YP_009724390.1"   # Wuhan-Hu-1 spike
L_FULL = 1273
AA = "ACDEFGHIKLMNPQRSTVWY"
AA21 = AA + "X"


def default_fasta():
    """找資料檔：優先 data/、其次 Data/（專案目前資料夾為大寫 Data）。"""
    root = Path(__file__).resolve().parents[3]
    for d in ("data", "Data"):
        p = root / d / "spike-protein" / "sequences.fasta"
        if p.exists():
            return p
    raise FileNotFoundError("找不到 spike-protein/sequences.fasta（已找 data/ 與 Data/）")


def read_fasta(path):
    """解析標頭 `登錄號 |國家|日期`，回傳 DataFrame(acc, country, date, seq)。"""
    rows = []
    for r in SeqIO.parse(path, "fasta"):
        parts = [p.strip() for p in r.description.split("|")]
        rows.append((r.id, parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "", str(r.seq)))
    df = pd.DataFrame(rows, columns=["acc", "country", "date_raw", "seq"])
    df["date"] = pd.to_datetime(df.date_raw, errors="coerce", format="mixed")  # 混有 YYYY-MM / YYYY-MM-DD
    return df


def quality_filter(df, length=L_FULL, max_x_frac=0.01):
    """長度、日期可解析、不確定殘基 ≤1%、無終止密碼子（FR-001）。"""
    n_unc = df.seq.str.count("[XBZJUO]")
    ok = (df.seq.str.len() == length) & df.date.notna() & (n_unc <= max_x_frac * length) \
        & ~df.seq.str.contains(r"\*", regex=True)
    return df[ok].reset_index(drop=True)


def dedupe(df):
    """相同序列合併，保留代表登錄號 acc／count／first_date／last_date／n_country，依 first_date 排序。"""
    return (df.groupby("seq")
              .agg(acc=("acc", "first"), count=("acc", "size"), first_date=("date", "min"),
                   last_date=("date", "max"), n_country=("country", "nunique"))
              .reset_index().sort_values(["first_date", "seq"]).reset_index(drop=True))


def accession_lookup(fasta=None):
    """序列字串 → 代表登錄號（同序列取首次出現；供熱圖 Y 軸，對應論文圖 10-12）。"""
    df = quality_filter(read_fasta(fasta or default_fasta()))
    return df.groupby("seq", sort=False).acc.first().to_dict()


def decode_aa(X_int):
    """整數編碼 (n,1273) → 胺基酸字串陣列。"""
    aa = np.array(list(AA21))
    return np.array(["".join(aa[row]) for row in np.asarray(X_int)])


def accession_labels(X_int, lookup=None, fasta=None):
    """為唯一序列列標上代表登錄號；找不到時退回 seq{i}。"""
    lut = lookup if lookup is not None else accession_lookup(fasta)
    labels = []
    for i, s in enumerate(decode_aa(X_int)):
        labels.append(str(lut.get(s, f"seq{i}")))
    return labels


def to_int(seqs, length=L_FULL):
    """序列字串 → (n, length) 的 uint8 編碼（A..Y=0..19，其他=20 即 X）。"""
    lut = np.full(256, 20, dtype=np.uint8)
    for i, a in enumerate(AA):
        lut[ord(a)] = i
    return lut[np.frombuffer("".join(seqs).encode(), dtype=np.uint8)].reshape(len(seqs), length)


def load_all(path=None, ref_id=REF_ID, length=L_FULL):
    """完整流程：讀檔 → 過濾 → 去重。回傳 (df_filtered, uniq, ref_seq, counts) ；counts 為各階段數量。"""
    df = read_fasta(path or default_fasta())
    n_raw = len(df)
    ref_seq = df.loc[df.acc == ref_id, "seq"].iloc[0] if (df.acc == ref_id).any() else df.seq.iloc[0]
    df = quality_filter(df, length)
    uniq = dedupe(df)
    return df, uniq, ref_seq, dict(raw=n_raw, filtered=len(df), unique=len(uniq))
