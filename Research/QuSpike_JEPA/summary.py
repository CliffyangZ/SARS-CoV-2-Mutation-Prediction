"""修訂版報告（§4.2／§5）：跳接增量、預訓練貢獻、距參考線與天花板的差距、譜系群分解。

與 report.py（v1 格式）並存；本模組只處理修訂後的假說 H3′／H4′／H5′ 所需的量。
判準一律「差異是否超過種子標準差」，且差異小於 NOISE_FLOOR 視為雜訊內（§7-R7：GPU 重跑可差 0.06）。
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from Research.lineage_baselines.results_io import read_results

NOISE_FLOOR = 0.02  # §9 停止條件：增量需超過此值才算有增量


def _ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return (float("nan"), float("nan"))
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(ms):
    """(平均, 標準差) → 字串。"""
    return "—" if not np.isfinite(ms[0]) else f"{ms[0]:.3f}±{ms[1]:.3f}"


def collect(out_dir, split="test", key="f1_macro"):
    """讀結果檔，回傳 {設定名: {...}}；略過未完成者。

    `collapsed` 是**預訓練表徵**的診斷：凍結 probe 的分數會因此失效，但 fine-tune 的任務分數不會
    （fine-tune 會重新訓練整個編碼器）。故只在非 fine-tune 的結果上據此排除，否則僅記錄旗標。
    """
    g = defaultdict(lambda: dict(vals=[], cfg=None, per_class=None, seconds=[], collapsed=[]))
    for r in read_results(out_dir):
        if r.get("status") != "complete":
            continue
        if r.get("collapsed") and not r["config"].get("finetune"):
            continue
        m = r.get(f"metrics_{split}")
        if not m:
            continue
        name = r["config"]["name"]
        g[name]["vals"].append(m[key])
        g[name]["cfg"] = r["config"]
        g[name]["collapsed"].append(bool(r.get("collapsed")))
        g[name]["seconds"].append(r.get("train_seconds", 0.0) + r.get("ft_seconds", 0.0))
        if r.get(f"per_class_{split}"):
            g[name]["per_class"] = r[f"per_class_{split}"]
    return g


def _find(g, **want):
    """找出 config 符合 want 的設定名（唯一時回傳，否則 None）。"""
    hits = [n for n, d in g.items() if d["cfg"] and all(d["cfg"].get(k) == v for k, v in want.items())]
    return hits[0] if len(hits) == 1 else None


def verdict(a, b, floor=NOISE_FLOOR):
    """a 相對 b 的增量與判定。回傳 (增量, 說明)。"""
    if not (np.isfinite(a[0]) and np.isfinite(b[0])):
        return float("nan"), "資料不足"
    d = a[0] - b[0]
    sd = max(a[1], b[1], floor)
    if abs(d) <= sd:
        return d, f"在雜訊內（±{sd:.3f}）"
    return d, "較佳" if d > 0 else "較差"


QUANTUM_HEADS = ("quantum_multi", "quantum", "quantum_1layer")
CLASSICAL_WIDE = "classical_wide"


def _want(head, ch, **kw):
    """組出 _find 的條件；head=None 表示不限 head（舊結果檔沒有這個欄位）。"""
    w = dict(channels=ch, finetune=True, **kw)
    if head is not None:
        w["head"] = head
    return w


def _pre_scr(g, head, ch):
    """同 head、同通道下的 (預訓練設定名, scratch 設定名)。"""
    pre = next((_find(g, **_want(head, ch, epochs=e)) for e in (20, 10)
                if _find(g, **_want(head, ch, epochs=e))), None)
    return pre, _find(g, **_want(head, ch, epochs=0))


def table_increments(g, split="test"):
    """H3′ 跳接增量、等容量量子對古典，以及 H4′ 預訓練貢獻。"""
    L = [f"## 增量（{split} macro F1；§4.2 H3′／H4′）", "",
         "| 比較 | 設定 A | 設定 B | A − B | 判定 |", "|---|---|---|---|---|"]
    skip = _find(g, channels="skip")
    heads = [h for h in (*QUANTUM_HEADS, CLASSICAL_WIDE) if any(d["cfg"] and d["cfg"].get("head") == h for d in g.values())]
    rows = []
    for head in heads or [None]:
        lab = head or "all"
        for ch in ("both", "q"):
            pre, scr = _pre_scr(g, head, ch)
            if pre and scr:  # H4′：預訓練貢獻（同 head、同通道）
                a, b = _ms(g[pre]["vals"]), _ms(g[scr]["vals"])
                d, v = verdict(a, b)
                rows.append((f"H4′ 預訓練貢獻（{lab}／{ch}）", fmt(a), fmt(b), d, v))
    # H3′：跳接增量（各 head 的最佳 both 設定 − 參考線）
    for head in heads or [None]:
        cand = [n for n in g if g[n]["cfg"] and g[n]["cfg"].get("channels") == "both"
                and (head is None or g[n]["cfg"].get("head") == head)]
        best = max(cand, key=lambda n: _ms(g[n]["vals"])[0], default=None)
        if best and skip:
            a, b = _ms(g[best]["vals"]), _ms(g[skip]["vals"])
            d, v = verdict(a, b)
            rows.append((f"H3′ 跳接增量（{head or 'both'} − 參考線）", fmt(a), fmt(b), d, v))
    # H3′ 的正式對照：等容量、同通道、同視角下量子頭 vs 寬度匹配古典頭
    for ch in ("both", "q"):
        for ep in (0, 10, 20):
            q = next((_find(g, head=h, channels=ch, finetune=True, epochs=ep) for h in QUANTUM_HEADS
                      if _find(g, head=h, channels=ch, finetune=True, epochs=ep)), None)
            c = _find(g, head=CLASSICAL_WIDE, channels=ch, finetune=True, epochs=ep)
            if q and c:
                a, b = _ms(g[q]["vals"]), _ms(g[c]["vals"])
                d, v = verdict(a, b)
                tag = "scratch" if ep == 0 else f"預訓練{ep}"
                rows.append((f"**量子 − 寬度匹配古典**（{ch}／{tag}）", fmt(a), fmt(b), d, v))
    for lab, a, b, d, v in rows:
        L.append(f"| {lab} | {a} | {b} | {d:+.3f} | {v} |")
    if not rows:
        L.append("| （缺少可比較的設定） | — | — | — | — |")
    return L


def table_main(g, split="test", reference=None, ceiling=None):
    """主表：各設定分數，附距參考線與距天花板的差距（§5 修訂）。"""
    L = [f"## 主結果（{split} macro F1）", "",
         "| 設定 | 種子 | macro F1 | 距參考線 | 距天花板 | 秒 | 表徵塌陷 |", "|---|---|---|---|---|---|---|"]
    for name in sorted(g):
        ms = _ms(g[name]["vals"])
        dr = f"{ms[0] - reference:+.3f}" if reference and np.isfinite(ms[0]) else "—"
        dc = f"{ms[0] - ceiling:+.3f}" if ceiling and np.isfinite(ms[0]) else "—"
        flags = g[name]["collapsed"]
        nc = sum(flags)
        col = f"{nc}/{len(flags)}" if nc else "—"
        L.append(f"| {name} | {len(g[name]['vals'])} | {fmt(ms)} | {dr} | {dc} | "
                 f"{np.mean(g[name]['seconds']):.0f} | {col} |")
    return L


def table_groups(g, split="test"):
    """譜系群分解（§5 修訂）。"""
    from .ceiling import group_macro_f1

    names = [n for n in sorted(g) if g[n]["per_class"]]
    if not names:
        return []
    keys = ["delta_parent", "delta_sub", "b1_family", "others"]
    L = [f"## 譜系群 macro F1（{split}；§5 修訂）", "", "| 設定 | " + " | ".join(keys) + " |",
         "|---" * (len(keys) + 1) + "|"]
    for n in names:
        gm = group_macro_f1(g[n]["per_class"])
        L.append(f"| {n} | " + " | ".join(f"{gm[k][0]:.3f}" if k in gm else "—" for k in keys) + " |")
    return L


def build(out_dir, split="test", reference=None, ceiling=None):
    """產生報告文字。"""
    g = collect(out_dir, split)
    L = ["# QuSpike-JEPA 修訂版結果", "",
         f"資料來源：`{out_dir}`；{split} macro F1；只計入 status=complete 且未塌陷者。",
         f"參考線（監督線性）={reference if reference else '未給'}；天花板（簽章 oracle）={ceiling if ceiling else '未給'}。",
         f"增量需超過 {NOISE_FLOOR}（§7-R7 的 GPU 噪聲下限）才算有增量。", ""]
    L += table_main(g, split, reference, ceiling) + [""]
    L += table_increments(g, split) + [""]
    L += table_groups(g, split)
    return "\n".join(L) + "\n"


def main(argv=None):
    """命令列入口。"""
    from .run import DEFAULT_OUT

    p = argparse.ArgumentParser(description="QuSpike-JEPA 修訂版報告")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--reference", type=float, default=None, help="監督線性參考線的 macro F1")
    p.add_argument("--ceiling", type=float, default=None, help="簽章 oracle 上限")
    p.add_argument("--write", default=None, help="寫入檔案（預設只印出）")
    a = p.parse_args(argv)
    txt = build(a.out, a.split, a.reference, a.ceiling)
    print(txt)
    if a.write:
        Path(a.write).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
