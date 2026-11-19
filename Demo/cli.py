"""命令列：python -m Demo.cli build | predict | serve。"""
import argparse
import json
import sys
from pathlib import Path

from . import bundle as B, parse
from .predict import Predictor


def _fmt(pred):
    lines = ["Top-k 譜系（機率為融合後；參考機率為僅參考分類器；殘差 = 融合 logit − 參考 logit）",
             f"{'#':>2} {'譜系':<10}{'機率':>8}{'參考機率':>10}{'殘差':>9}{'參考名次':>8}"]
    for t in pred.top_k:
        lines.append(f"{t.rank:>2} {t.lineage:<10}{t.prob:>8.4f}{t.ref_prob:>10.4f}{t.residual:>+9.4f}{t.ref_rank:>8}")
    r = pred.routing
    probs = "  ".join(f"{g}={v:.3f}" for g, v in r.probs.items())
    lines += ["", f"路由群組：{r.group}（{probs}）｜專家{'已套用' if r.expert_applied else '未套用'}（s={pred.s_used:g}）",
              "專家改變了 Top-1" if pred.top1_changed_by_expert else "專家未改變 Top-1"]
    for w in pred.parsed["warnings"]:
        lines.append(f"警告：{w}")
    if pred.confidence_note:
        lines.append(f"注意：{pred.confidence_note}")
    lines += ["", pred.disclaimer]
    return "\n".join(lines)


def cmd_build(a):
    B.build_bundle(seed=a.seed, out=a.out, limit=a.limit)
    return 0


def cmd_predict(a):
    if a.seq_file:
        text, mode = Path(a.seq_file).read_text(encoding="utf-8"), "sequence"
    elif a.seq is not None:
        text, mode = a.seq, "sequence"
    else:
        text, mode = a.mut, "mutations"
    try:
        pred = Predictor(B.load_bundle(a.bundle)).predict(text, mode, k=a.k, s=a.s)
    except B.BundleMissing as e:
        print(e, file=sys.stderr)
        return 3
    except parse.InputError as e:
        print(f"輸入無效：{e}", file=sys.stderr)
        for d in e.details:
            print(f"  - {d['item']}：{d['problem']}", file=sys.stderr)
        return 2
    print(json.dumps(pred.to_dict(), ensure_ascii=False, indent=2) if a.json else _fmt(pred))
    return 0


def cmd_serve(a):
    from .server import serve
    return serve(a.host, a.port, a.bundle)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m Demo.cli", description="即時譜系預測器 Demo")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="訓練並凍結模型包")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=str(B.DEFAULT_BUNDLE))
    b.add_argument("--limit", type=int, default=None, help="僅供測試：隨機子集大小")
    b.set_defaults(fn=cmd_build)
    q = sub.add_parser("predict", help="單筆預測")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq")
    g.add_argument("--seq-file")
    g.add_argument("--mut")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--s", type=float, default=None)
    q.add_argument("--json", action="store_true")
    q.add_argument("--bundle", default=str(B.DEFAULT_BUNDLE))
    q.set_defaults(fn=cmd_predict)
    s = sub.add_parser("serve", help="啟動本機網頁")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--bundle", default=str(B.DEFAULT_BUNDLE))
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
