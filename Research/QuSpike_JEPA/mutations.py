"""序列 → 突變集合（相對 Wuhan-Hu-1）。

位置為 1 起算。取代記為 (位置, 胺基酸索引)；`X` 保留為獨立的未知符號；
序列內部的刪除記為符號 "-"；插入忽略。序列首尾比參考短的部分視為未觀測，不算刪除。
"""
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF_ID = "YP_009724390.1"
FASTA = ROOT / "Data" / "spike-protein" / "sequences.fasta"

SYMBOLS = list("ACDEFGHIKLMNPQRSTVWY") + ["X", "-"]
SYM_INDEX = {c: i for i, c in enumerate(SYMBOLS)}
UNKNOWN = SYM_INDEX["X"]
DELETION = SYM_INDEX["-"]


def _load_reference():
    """從 fasta 讀 Wuhan-Hu-1；找不到時明確報錯。"""
    if not FASTA.exists():
        raise FileNotFoundError(f"缺少參考序列檔 {FASTA}")
    seq, on = [], False
    for line in FASTA.read_text().splitlines():
        if line.startswith(">"):
            if on:
                break
            on = line[1:].split()[0] == REF_ID
        elif on:
            seq.append(line.strip())
    if not seq:
        raise ValueError(f"fasta 中找不到 {REF_ID}")
    return "".join(seq)


REFERENCE = _load_reference()
N_POS = len(REFERENCE)


def _direct(seq):
    """等長序列逐位比較。"""
    return [(i + 1, SYM_INDEX.get(c, UNKNOWN)) for i, (c, r) in enumerate(zip(seq, REFERENCE)) if c != r]


def mutations_of(seq):
    """回傳 [(位置, 符號索引)]，依位置排序。"""
    if len(seq) == N_POS:
        return _direct(seq)
    sm = SequenceMatcher(None, REFERENCE, seq, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            k = min(i2 - i1, j2 - j1)
            out += [(i1 + t + 1, SYM_INDEX.get(seq[j1 + t], UNKNOWN)) for t in range(k)]
            i1 += k
            tag = "delete" if i2 > i1 else "insert"
        if tag == "delete" and 0 < i1 and i2 < N_POS:
            out += [(p, DELETION) for p in range(i1 + 1, i2 + 1)]
    return sorted(out)


def site_variability(pos, sym, valid):
    """各 (位置, 符號) token 在序列中的出現比例（只傳入 train 的 TokenSet 陣列）。回傳 dict token_id→比例。"""
    from collections import Counter

    from .dataset import N_SYM

    ids = (pos.astype("int64") * N_SYM + sym.astype("int64"))[valid]
    c = Counter(ids.tolist())
    n = max(len(pos), 1)
    return {k: v / n for k, v in c.items()}
