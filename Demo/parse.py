"""輸入解析：spike 序列／FASTA／突變清單 → 突變集合（與訓練特徵同一表示）。

突變以 (位置, 符號索引) 表示，位置 1 起算、相對 Wuhan-Hu-1。
特徵向量與 `Research.QuSpike_PQK.features.mutation_matrix` 同一規則：排除 X，其餘依詞彙轉 0/1。
"""
import re
from dataclasses import dataclass, field

import numpy as np

from Research.QuSpike_JEPA.dataset import N_SYM
from Research.QuSpike_JEPA.mutations import (DELETION, N_POS, REFERENCE, SYM_INDEX, SYMBOLS,
                                              UNKNOWN, mutations_of)

MIN_LEN = 200       # 低於此長度直接拒絕（無法可靠比對）
WARN_LEN = 1200     # 低於此長度視為截斷，首尾未觀測部分不算突變
MAX_LEN = 1400
AMINO = set(SYMBOLS)


class InputError(ValueError):
    """輸入無效。details 為 [{item, problem}]，供介面指出問題位置或項目。"""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or []


@dataclass
class ParsedInput:
    kind: str
    mutations: list                      # [(pos, sym)]，依位置排序
    n_unknown: int = 0                   # X 位置數
    length: int = None
    warnings: list = field(default_factory=list)
    n_out_of_vocab: int = 0


def clean_sequence(text):
    """去除 FASTA 標頭、空白與換行並轉大寫。只取第一筆記錄。"""
    lines, seen = [], False
    for ln in str(text).splitlines():
        ln = ln.strip()
        if ln.startswith(">"):
            if seen:
                break
            seen = True
            continue
        lines.append(ln)
    return re.sub(r"\s+", "", "".join(lines)).upper()


def parse_sequence(text):
    seq = clean_sequence(text)
    if not seq:
        raise InputError("輸入是空的")
    bad = [{"item": f"位置 {i + 1}", "problem": f"不允許的字元 {c!r}"}
           for i, c in enumerate(seq) if c not in AMINO]
    if bad:
        raise InputError(f"序列含 {len(bad)} 個不允許的字元（只接受 20 種胺基酸、X、-）", bad[:10])
    if len(seq) < MIN_LEN:
        raise InputError(f"序列只有 {len(seq)} aa，太短，無法可靠比對（spike 約 {N_POS} aa）")
    if len(seq) > MAX_LEN:
        raise InputError(f"序列有 {len(seq)} aa，遠長於 spike（約 {N_POS} aa）")
    warns = []
    if len(seq) < WARN_LEN:
        warns.append(f"序列只有 {len(seq)} aa，可能被截斷；首尾未觀測的部分不算突變")
    muts = mutations_of(seq)
    n_unk = sum(1 for _, s in muts if s == UNKNOWN)
    if n_unk:
        warns.append(f"含 {n_unk} 個 X（未知）位置，已忽略，可能降低準確度")
    if not muts:
        warns.append("與參考株完全相同，可能落在訓練分佈邊緣")
    return ParsedInput("sequence", muts, n_unk, len(seq), warns)


_SUB = re.compile(r"^([A-Z])(\d+)([A-Z\-])$")
_DEL_RANGE = re.compile(r"^(?:DEL|Δ)(\d+)(?:[-–](\d+))?$")
_DEL_AA = re.compile(r"^([A-Z])(\d+)(?:DEL|-|Δ)$")


def parse_mutation_item(item):
    """單一項目 → [(pos, sym)]。例：L452R、del69-70、H69-、Δ69-70。"""
    t = item.strip().upper().replace("Δ", "DEL")
    t = re.sub(r"^S:", "", t)
    m = _SUB.match(t)
    if m:
        ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        _check_pos(item, pos)
        if REFERENCE[pos - 1] != ref:
            raise InputError(f"{item}：位置 {pos} 的參考胺基酸是 {REFERENCE[pos - 1]}，不是 {ref}",
                             [{"item": item, "problem": f"位置 {pos} 的參考胺基酸是 {REFERENCE[pos - 1]}"}])
        if alt == "-":
            return [(pos, DELETION)]
        if alt not in SYM_INDEX:
            raise InputError(f"{item}：未知胺基酸 {alt}")
        return [] if alt == ref else [(pos, SYM_INDEX[alt])]
    m = _DEL_RANGE.match(t)
    if m:
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if b < a:
            a, b = b, a
        _check_pos(item, a)
        _check_pos(item, b)
        return [(p, DELETION) for p in range(a, b + 1)]
    m = _DEL_AA.match(t)
    if m:
        ref, pos = m.group(1), int(m.group(2))
        _check_pos(item, pos)
        if REFERENCE[pos - 1] != ref:
            raise InputError(f"{item}：位置 {pos} 的參考胺基酸是 {REFERENCE[pos - 1]}，不是 {ref}",
                             [{"item": item, "problem": f"位置 {pos} 的參考胺基酸是 {REFERENCE[pos - 1]}"}])
        return [(pos, DELETION)]
    raise InputError(f"{item}：無法解析（格式例：L452R、del69-70、H69-）",
                     [{"item": item, "problem": "無法解析"}])


def _check_pos(item, pos):
    if not 1 <= pos <= N_POS:
        raise InputError(f"{item}：位置 {pos} 超出範圍 1–{N_POS}",
                         [{"item": item, "problem": f"位置 {pos} 超出範圍 1–{N_POS}"}])


def parse_mutation_list(text):
    items = [x for x in re.split(r"[,\s;]+", str(text).strip()) if x]
    if not items:
        raise InputError("輸入是空的")
    muts, problems = {}, []
    for it in items:
        try:
            for p, s in parse_mutation_item(it):
                muts[p] = s
        except InputError as e:
            problems += e.details or [{"item": it, "problem": str(e)}]
    if problems:
        raise InputError(f"{len(problems)} 個項目有問題；整份清單未被使用，請修正後重送", problems)
    out = sorted(muts.items())
    n_unk = sum(1 for _, s in out if s == UNKNOWN)
    warns = [] if out else ["清單沒有任何突變，等同參考株，可能落在訓練分佈邊緣"]
    return ParsedInput("mutations", out, n_unk, None, warns)


def detect_kind(text):
    t = str(text).strip()
    if t.startswith(">"):
        return "sequence"
    body = re.sub(r"\s+", "", t)
    return "sequence" if body and not re.search(r"\d", body) and len(body) >= 50 else "mutations"


def parse_input(text, mode="auto"):
    if text is None or not str(text).strip():
        raise InputError("輸入是空的")
    mode = detect_kind(text) if mode == "auto" else mode
    if mode == "sequence":
        return parse_sequence(text)
    if mode == "mutations":
        return parse_mutation_list(text)
    raise InputError(f"未知的輸入模式 {mode}")


def to_vector(parsed, vocab):
    """0/1 突變向量 (1, V)；回傳 (向量, 詞彙外突變數)。與 features.mutation_matrix 同規則（排除 X）。"""
    toks = np.array([p * N_SYM + s for p, s in parsed.mutations if s != UNKNOWN], np.int64)
    x = np.zeros((1, len(vocab)), np.float32)
    n_oov = 0
    if len(toks):
        j = np.searchsorted(vocab, toks)
        ok = (j < len(vocab)) & (vocab[np.minimum(j, len(vocab) - 1)] == toks)
        x[0, j[ok]] = 1.0
        n_oov = int((~ok).sum())
    parsed.n_out_of_vocab = n_oov
    return x, n_oov


def mutations_to_text(muts):
    """突變集合 → 清單文字（測試與等價性檢查用）。"""
    out = []
    for p, s in muts:
        ref = REFERENCE[p - 1]
        alt = "-" if s == DELETION else SYMBOLS[s]
        out.append(f"{ref}{p}{alt}")
    return ", ".join(out)
