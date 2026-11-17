"""標籤歧義上限（§5 修訂）：相同突變組合的序列無法被任何只看 Spike 的模型分開。

把突變指標相同的序列歸為一組，每組都猜該組的多數標籤。這會偷看標籤，所以是**樂觀上限**，
不是可達分數；用途是回答「距離天花板還有多遠」，避免把資料本身的歧義當成模型缺陷。
"""
import numpy as np

from Research.lineage_baselines.metrics import compute_metrics, per_class_report

from .dataset import MASK_SYM, N_SYM
from .mutations import UNKNOWN

# 譜系群（§5 修訂：按群分解損失）。Delta 母譜系與其子譜系、B.1 系，是損失最集中的兩群。
GROUPS = {
    "delta_parent": ("B.1.617.2",),
    "delta_sub": ("AY.103", "AY.2", "AY.25", "AY.3", "AY.4", "AY.44"),
    "b1_family": ("B.1", "B.1.1", "B.1.2", "B.1.177"),
}


def signatures(ts, min_count=1, train_ts=None):
    """回傳每條序列的突變簽章 id（相同突變集合者相同）。

    min_count：只保留在 train_ts（預設為 ts）出現至少這麼多次的 token，模擬「稀有突變不可靠」的情形。
    """
    src = train_ts if train_ts is not None else ts

    def toks(t):
        """回傳 (token id 矩陣, 有效遮罩)；token id = 位置·N_SYM + 符號。"""
        keep = t.valid & (t.sym != UNKNOWN) & (t.sym != MASK_SYM)
        return t.pos.astype(np.int64) * N_SYM + t.sym.astype(np.int64), keep

    tok_s, keep_s = toks(src)
    vals, cnt = np.unique(tok_s[keep_s], return_counts=True)
    vocab = vals[cnt >= min_count]
    tok, keep = toks(ts)
    inside = keep & np.isin(tok, vocab)
    rows = []
    for i in range(len(ts)):
        rows.append(np.sort(tok[i][inside[i]]).tobytes())
    _, inv = np.unique(np.array(rows), return_inverse=True)
    return inv


def oracle(ts, n_classes=None, min_count=1, train_ts=None, labels=None):
    """簽章 oracle 的指標：每個簽章猜其多數標籤。回傳 compute_metrics 的字典，另含簽章數。"""
    labels = labels if labels is not None else ts.labels
    n_classes = n_classes or len(labels)
    sig = signatures(ts, min_count=min_count, train_ts=train_ts)
    y = np.asarray(ts.y)
    pred = np.empty(len(y), np.int64)
    for s in np.unique(sig):
        m = sig == s
        pred[m] = np.bincount(y[m], minlength=n_classes).argmax()
    proba = np.eye(n_classes)[pred]
    out = compute_metrics(y, pred, proba, n_classes)
    out["n_signatures"] = int(len(np.unique(sig)))
    out["min_count"] = int(min_count)
    out["per_class"] = per_class_report(y, pred, n_classes, labels)
    return out


def group_macro_f1(per_class):
    """由 per_class 報告算各譜系群的 macro F1（§5 修訂）。回傳 {群名: (macro F1, 類別數)}。"""
    by = {c["lineage"]: c["f1"] for c in per_class}
    out = {}
    for g, names in GROUPS.items():
        f = [by[n] for n in names if n in by]
        if f:
            out[g] = (float(np.mean(f)), len(f))
    rest = [c["f1"] for c in per_class if c["lineage"] not in {n for ns in GROUPS.values() for n in ns}]
    if rest:
        out["others"] = (float(np.mean(rest)), len(rest))
    return out


def main(argv=None):
    """命令列：印出指定資料集的上限與分群。"""
    import argparse

    from . import dataset as D

    p = argparse.ArgumentParser(description="標籤歧義上限（簽章 oracle）")
    p.add_argument("--data", default=str(D.DEFAULT_DATA))
    p.add_argument("--min-counts", type=int, nargs="+", default=[1, 3, 10])
    a = p.parse_args(argv)
    train, test, _ = D.load_splits(a.data)
    print(f"{'split':6s} {'minc':>5s} {'簽章數':>8s} {'accuracy':>9s} {'macro F1':>9s}")
    for mc in a.min_counts:
        for nm, ts in (("train", train), ("test", test)):
            r = oracle(ts, min_count=mc, train_ts=train)
            print(f"{nm:6s} {mc:5d} {r['n_signatures']:8d} {r['accuracy']:9.3f} {r['f1_macro']:9.3f}")
    r = oracle(test, min_count=1, train_ts=train)
    print("\n分群 macro F1（上限）：")
    for g, (v, n) in group_macro_f1(r["per_class"]).items():
        print(f"  {g:14s} {v:.3f}（{n} 類）")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
