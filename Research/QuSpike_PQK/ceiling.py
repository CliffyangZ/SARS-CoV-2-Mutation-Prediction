"""困難群的天花板報告（Q5，FR-006／SC-004）。

「只看 spike 突變」時，突變指標向量完全相同的序列在特徵空間裡是同一個點：
若它們的標籤不同，任何只吃這些特徵的模型（古典或量子）都不可能全部答對。
本模組量化這個資料本身的限制，回答「Delta 與 B.1 家族的錯誤有多少比例在只看 spike 時無法避免」。

注意：群內 macro-F1 上限是**樂觀上限**——每個簽章猜其多數標籤，而多數標籤是看了測試標籤才決定的，
所以它不是可達分數。用途只是回答「距離天花板還有多遠」，避免把資料本身的歧義當成模型缺陷。
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from . import features as F


def _keys(x):
    """每列的突變簽章：把 0/1 向量打包成 bytes。"""
    packed = np.packbits(np.asarray(x) > 0, axis=1)
    return [r.tobytes() for r in packed]


def _conflict(keys, y):
    """回傳 (簽章數, 含多標籤的簽章數, 位於衝突簽章內的列比例, 衝突簽章的布林遮罩)。"""
    by = {}
    for i, k in enumerate(keys):
        by.setdefault(k, []).append(i)
    conflicting = np.zeros(len(keys), bool)
    n_conf = 0
    for idx in by.values():
        if len(np.unique(y[idx])) > 1:
            conflicting[idx] = True
            n_conf += 1
    frac = float(conflicting.mean()) if len(keys) else 0.0
    return len(by), n_conf, frac, conflicting


def _oracle_pred(keys, y):
    """每個簽章猜其多數標籤（偷看標籤，故為樂觀上限）。"""
    by = {}
    for i, k in enumerate(keys):
        by.setdefault(k, []).append(i)
    pred = np.empty(len(keys), np.int64)
    for idx in by.values():
        pred[idx] = np.bincount(y[idx]).argmax()
    return pred


def _confusions(keys, y, labels):
    """每類的主要混淆對象：同簽章下最常出現的其他標籤。"""
    by = {}
    for i, k in enumerate(keys):
        by.setdefault(k, []).append(i)
    pair = {}
    for idx in by.values():
        ys, cnt = np.unique(y[idx], return_counts=True)
        if len(ys) < 2:
            continue
        for a, ca in zip(ys, cnt):
            for b, cb in zip(ys, cnt):
                if a != b:
                    pair[(int(a), int(b))] = pair.get((int(a), int(b)), 0) + int(ca * 1)  # a 被 b 擋住的列數
    out = {}
    for c in sorted({a for a, _ in pair}):
        others = {b: v for (a, b), v in pair.items() if a == c}
        b, v = max(others.items(), key=lambda kv: kv[1])
        out[labels[c]] = dict(confused_with=labels[b], rows_in_shared_signature=v)
    return out


def ceiling_report(fs, groups, c0_per_class=None):
    """每個困難群一筆報告。c0_per_class：C0 的 per_class_test，用來換算「無法避免的錯誤占比」。"""
    out = {}
    for g in groups:
        if not g.is_quantum:
            continue
        ids = np.asarray(g.class_ids)
        tr = np.where(np.isin(fs.y_train, ids))[0]
        te = np.where(np.isin(fs.y_test, ids))[0]
        k_tr, k_te = _keys(fs.x_train[tr]), _keys(fs.x_test[te])
        y_tr, y_te = fs.y_train[tr], fs.y_test[te]
        ns_tr, nc_tr, cf_tr, _ = _conflict(k_tr, y_tr)
        ns_te, nc_te, cf_te, conf_te = _conflict(k_te, y_te)
        pred = _oracle_pred(k_te, y_te)
        f1 = float(f1_score(y_te, pred, labels=ids, average="macro", zero_division=0))
        oracle_errors = int((pred != y_te).sum())
        unavoidable = None
        if c0_per_class:
            by = {c["lineage"]: c for c in c0_per_class}
            actual = sum(round(by[fs.labels[c]]["support"] * (1 - by[fs.labels[c]]["recall"]))
                         for c in ids if fs.labels[c] in by)
            unavoidable = dict(oracle_errors=oracle_errors, c0_errors=int(actual),
                               fraction=float(min(1.0, oracle_errors / actual)) if actual else None)
        out[g.name] = dict(
            n_train=int(len(tr)), n_test=int(len(te)),
            n_signatures_train=ns_tr, n_conflicting_signatures_train=nc_tr,
            conflict_fraction_train=cf_tr,
            n_signatures_test=ns_te, n_conflicting_signatures_test=nc_te,
            conflict_fraction_test=cf_te,
            macro_f1_upper_bound=f1, upper_bound_is_optimistic=True,
            oracle_errors=oracle_errors, unavoidable_error=unavoidable,
            confusions=_confusions(k_te, y_te, fs.labels))
    return out


def run_report(data, c0_result_path=None):
    """完整報告（含詞彙外突變比例）。"""
    fs = F.load_features(source="mutation", data=data)
    groups = F.make_groups(fs)
    c0 = None
    if c0_result_path and Path(c0_result_path).exists():
        c0 = json.loads(Path(c0_result_path).read_text(encoding="utf-8")).get("per_class_test")
    return dict(split_version=fs.split_version, ood_mutation_fraction=fs.ood_fraction,
                c0_source=str(c0_result_path) if c0 else None,
                groups=ceiling_report(fs, groups, c0))


def to_markdown(rep):
    """人類可讀的天花板報告。"""
    lines = ["# 困難群天花板報告（只看 spike 突變）", "",
             f"- 切分：`{rep['split_version']}`",
             f"- 測試序列含訓練詞彙外突變的比例：{rep['ood_mutation_fraction']:.4f}", ""]
    for g, r in rep["groups"].items():
        lines += [f"## 群 {g}", "",
                  f"- 訓練列 {r['n_train']}、測試列 {r['n_test']}",
                  f"- 測試：簽章 {r['n_signatures_test']} 種，其中 {r['n_conflicting_signatures_test']} 種含多個標籤；"
                  f"位於衝突簽章的序列比例 **{r['conflict_fraction_test']:.3f}**（訓練 {r['conflict_fraction_train']:.3f}）",
                  f"- 群內 macro-F1 上限 **{r['macro_f1_upper_bound']:.3f}**（**樂觀上限**：偷看標籤，非可達分數）",
                  f"- 簽章 oracle 的最少錯誤數：{r['oracle_errors']}"]
        u = r["unavoidable_error"]
        if u:
            lines.append(f"- C0 在此群錯 {u['c0_errors']} 筆，其中**至少 {u['fraction']:.1%}** 在只看 spike 時無法避免（下界：單例簽章在 oracle 下計為零錯誤）")
        lines += ["", "| 類別 | 主要混淆對象 | 共用簽章列數 |", "|---|---|---|"]
        lines += [f"| {c} | {v['confused_with']} | {v['rows_in_shared_signature']} |"
                  for c, v in r["confusions"].items()]
        lines.append("")
    return "\n".join(lines)
