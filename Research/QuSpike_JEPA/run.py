"""命令列入口：執行單一設定或單一消融軸（契約見 specs/006-quspike-jepa/contracts/cli.md）。

預設只在 validation 評估（調參用，不讀 test）；加 --final 才以完整 train 重訓並在 test 評一次。
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

from Research.lineage_baselines.metrics import compute_metrics, per_class_report
from Research.lineage_baselines.results_io import environment_info, is_complete, result_path, write_result

from . import baselines, dataset as D, finetune as FT, loss as L, pairs, probe as PR, pretrain as PT

DEFAULT_OUT = D.ROOT / "Results" / "quspike_jepa"
# 設定項來自文件者標 doc，其餘為自行決定（FR-018）
DOC_KEYS = {"variant", "n_qubits", "n_layers", "entangle", "trainable_obs", "trainable_freq", "mask_mut", "mask_pos",
            "trim", "local_trim", "min_mut", "mask_prior", "whiten", "head"}

ABLATE_AXES = {
    "n_qubits": int, "n_layers": int, "entangle": lambda s: s.lower() in ("1", "true", "yes"),
    "trainable_obs": lambda s: s.lower() in ("1", "true", "yes"), "variant": str, "min_mutations": int,
    "whitening": str, "encoder": str, "mask_ratio": float, "mask_level": str, "mask_dist": str, "anchor_sampling": str,
    "n_heads": int, "recon": float, "channels": str, "enc_dim": int,
}


def build_parser():
    """建立命令列參數解析器。"""
    p = argparse.ArgumentParser(description="QuSpike-JEPA 譜系分類")
    p.add_argument("--data", default=str(D.DEFAULT_DATA))
    p.add_argument("--variant", default="same_lineage", choices=sorted(pairs.VARIANTS))
    p.add_argument("--head", default="classical",
                   choices=["classical", "quantum", "classical_matched", "quantum_1layer", "random_encoder", "mutset_nn", "mutset_linear",
                            "quantum_multi", "classical_wide"])
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=1, help="並聯量子頭數 H（§3.2 修訂）；只對 quantum_multi/classical_wide 有效")
    p.add_argument("--enc-dim", type=int, default=64, help="編碼器寬度 d_z（§3.1 修訂：與 n 解開）")
    p.add_argument("--recon", type=float, default=0.0, help="重建項權重 μ（§3.3 修訂）")
    p.add_argument("--channels", default="q", choices=FT.CHANNELS, help="分類通道：q／skip（參考線）／both（跳接，§3.2）")
    p.add_argument("--entangle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trainable-obs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fixed-obs", action="store_true")
    p.add_argument("--trainable-freq", action="store_true")
    p.add_argument("--encoder", default="deepsets", choices=["deepsets", "cnn1d"])
    p.add_argument("--mask-dist", default="uniform", choices=["uniform", "variability"])
    p.add_argument("--mask-level", default="both", choices=["mutation", "position", "both", "span"])
    p.add_argument("--mask-ratio", type=float, default=0.15)
    p.add_argument("--min-mutations", type=int, default=3)
    p.add_argument("--whitening", default="vicreg", choices=["vicreg", "identity", "none"])
    p.add_argument("--anchor-sampling", default="natural", choices=["natural", "balanced"])
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--ablate", default=None, choices=sorted(ABLATE_AXES))
    p.add_argument("--values", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--final", action="store_true")
    p.add_argument("--finetune", action="store_true", help="預訓練後以標籤 fine-tune 編碼器＋預測頭＋線性分類層")
    p.add_argument("--ft-epochs", type=int, default=FT.FT_DEFAULTS["ft_epochs"])
    p.add_argument("--ft-lr", type=float, default=FT.FT_DEFAULTS["ft_lr"])
    p.add_argument("--ft-head-lr", type=float, default=FT.FT_DEFAULTS["ft_head_lr"])
    p.add_argument("--ft-balanced", action="store_true", help="fine-tune 交叉熵依類別頻率加權")
    p.add_argument("--ft-freeze", action="store_true", help="fine-tune 時凍結編碼器與預測頭（只訓練分類層）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def make_cfg(a):
    """由參數組出預訓練設定。"""
    cfg = copy.deepcopy(PT.DEFAULTS)
    lv, r = a.mask_level, a.mask_ratio
    cfg.update(
        variant=pairs.canonical_variant(a.variant), encoder=a.encoder, head=a.head, n_qubits=a.n_qubits, n_layers=a.n_layers,
        entangle=a.entangle, trainable_obs=a.trainable_obs and not a.fixed_obs, trainable_freq=a.trainable_freq,
        mask_mut=r if lv in ("mutation", "both", "span") else 0.0, mask_pos=r if lv in ("position", "both", "span") else 0.0,
        span_prob=0.5 if lv == "span" else 0.0, mask_prior="P1" if a.mask_dist == "variability" else "P0",
        min_mut=a.min_mutations, whiten=a.whitening, anchor_sampling=a.anchor_sampling, epochs=a.epochs, batch_size=a.batch_size,
        n_heads=a.n_heads, recon=a.recon, enc_dim=a.enc_dim,
    )
    if a.head == "random_encoder":
        cfg.update(head="classical_matched", epochs=0)
    if a.finetune:
        cfg.update(FT.FT_DEFAULTS)
        cfg.update(finetune=True, ft_epochs=a.ft_epochs, ft_lr=a.ft_lr, ft_head_lr=a.ft_head_lr,
                   ft_balanced=a.ft_balanced, ft_freeze=a.ft_freeze, channels=a.channels)
    return cfg


def config_name(a, tag=""):
    """結果檔的設定名（不含種子）。"""
    q = f"_n{a.n_qubits}L{a.n_layers}" if a.head in ("quantum", "quantum_1layer", "quantum_multi") else ""
    if a.head in PT.MULTI_HEADS:
        q += f"H{a.n_heads}"
    ft = ""
    if getattr(a, "finetune", False):
        ft = "_ft" + ("frozen" if a.ft_freeze else "") + ("_bal" if a.ft_balanced else "") + ("_scratch" if a.epochs == 0 else "")
        if a.channels != "q":
            ft += f"_{a.channels}"
    if a.recon:
        ft += f"_mu{a.recon:g}"
    if a.enc_dim != 64:
        ft += f"_dz{a.enc_dim}"
    return f"jepa_{pairs.canonical_variant(a.variant)}_{a.head}{q}{ft}{tag}"


def _metrics_block(res):
    return dict(metrics=res["linear"], metrics_knn=res["knn"])


def _phase(train, evals, cfg, seed, a, ckpt, noisy=False):
    """預訓練於 train，probe 評估 evals（dict 名稱→TokenSet）。回傳 (輸出 dict, 預訓練資訊, 有效秩)。"""
    model, info = PT.pretrain(train, cfg, seed, ckpt_path=ckpt, device=a.device, resume=a.resume, log=lambda s: print("  " + s, flush=True))
    qtr, _ = PR.embed(model, train, a.device)
    n_cls = len(train.labels)
    sub = qtr[np.random.default_rng(seed).permutation(len(qtr))[:2048]]
    rank = L.effective_rank(torch.as_tensor(sub))
    info["min_dim_std"] = L.min_dim_std(sub)
    info["dead_dim_frac"] = L.dead_dim_frac(sub)
    feats = {k: PR.embed(model, ts, a.device)[0] for k, ts in evals.items()}
    out = {}
    if cfg.get("finetune"):
        probe_res = {k: PR.probe(qtr, train.y, [feats[k]], ts.y, n_cls, train.labels, seed=seed)[0][0] for k, ts in evals.items()}
        clf, ft_info = FT.finetune(model, train, cfg, seed, device=a.device, log=lambda s: print("  " + s, flush=True))
        info.update(ft_info)
        for k, ts in evals.items():
            e = FT.evaluate(clf, ts, a.device)
            out[k] = dict(linear=e["metrics"], knn=probe_res[k]["linear"], per_class=e["per_class"])
            if noisy and (a.shots or a.noise) and cfg["head"].startswith("quantum"):
                out[k + "_noisy"] = dict(linear=FT.evaluate(clf, ts, a.device, shots=a.shots, noise=a.noise)["metrics"])
        return out, info, rank
    for k, ts in evals.items():
        (res,), _ = PR.probe(qtr, train.y, [feats[k]], ts.y, n_cls, train.labels, seed=seed)
        out[k] = res
        if noisy and (a.shots or a.noise) and cfg["head"].startswith("quantum"):
            kw = dict(shots=a.shots, noise=a.noise)
            fn, _ = PR.embed(model, ts, a.device, **kw)
            (rn,), _ = PR.probe(qtr, train.y, [fn], ts.y, n_cls, train.labels, seed=seed)
            out[k + "_noisy"] = rn
    return out, info, rank


def run_one(a, cfg, name, seed, data):
    """執行一個（設定，種子），寫結果檔；已完成則跳過。"""
    path = result_path(a.out, name, "mut", seed)
    if is_complete(path):
        print(f"[跳過] {path.name} 已完成")
        return None
    train, test, split_version = data
    n_cls = len(train.labels)
    fit_idx, val_idx = D.split_validation(train)
    fit, val = train.subset(fit_idx), train.subset(val_idx)
    ck = Path(a.out) / "checkpoints"
    res = dict(
        config=dict(cfg, name=name, shots=a.shots, noise=a.noise, ablate=a.ablate, data=Path(a.data).name, n_classes=n_cls),
        seed=seed, split_version=split_version,
        config_origin={k: ("doc" if k in DOC_KEYS else "chosen") for k in cfg}, uses_label=cfg["variant"] == "VA",
        status="complete", environment=environment_info(),
    )
    try:
        if a.head == "mutset_linear":
            pred, pr = baselines.mutset_linear(fit, val, n_cls, seed=seed, device=a.device)
            res["metrics_val"] = compute_metrics(val.y, pred, pr, n_cls)
            res.update(train_seconds=0.0, effective_rank=None, collapsed=False)
            if a.final:
                pred, pr = baselines.mutset_linear(train, test, n_cls, seed=seed, device=a.device)
                res["metrics_test"] = compute_metrics(test.y, pred, pr, n_cls)
                res["per_class_test"] = per_class_report(test.y, pred, n_cls, train.labels)
        elif a.head == "mutset_nn":
            pred, pr = baselines.nearest_mutset(fit, val, n_cls)
            res["metrics_val"] = compute_metrics(val.y, pred, pr, n_cls)
            res.update(train_seconds=0.0, effective_rank=None, collapsed=False)
            if a.final:
                pred, pr = baselines.nearest_mutset(train, test, n_cls)
                res["metrics_test"] = compute_metrics(test.y, pred, pr, n_cls)
        else:
            out, info, rank = _phase(fit, {"val": val}, cfg, seed, a, ck / f"{name}__seed{seed}__val.pt", noisy=True)
            # §5 修訂：有效秩不可單獨判斷塌陷，另加方差下限與多數類基準
            maj = L.majority_macro_f1(val.y, n_cls)
            probe_f1 = out["val"].get("knn", out["val"]["linear"])["f1_macro"]
            # 塌陷診斷只在「有預訓練且使用 q 路徑」時有意義：channels='skip' 不用 q 路徑，
            # epochs=0（scratch 對照）的編碼器本來就是隨機的，其凍結 probe 低不代表塌陷。
            uses_q = cfg.get("channels", "q") != "skip"
            diagnosable = uses_q and cfg["epochs"] > 0
            res.update(metrics_val=out["val"]["linear"], metrics_val_knn=out["val"]["knn"], per_class_val=out["val"]["per_class"],
                       effective_rank=rank, min_dim_std=info["min_dim_std"], dead_dim_frac=info["dead_dim_frac"],
                       majority_macro_f1=maj,
                       below_majority=bool(probe_f1 <= maj * 1.5), q_channel_used=uses_q, collapse_diagnosable=diagnosable,
                       collapsed=bool(diagnosable and (rank < cfg["collapse_rank"] or not np.isfinite(rank)
                                                       or info["dead_dim_frac"] > 0.5 or probe_f1 <= maj * 1.5)),
                       pair_stats=info["view_stats"], train_seconds=info["train_seconds"], n_params=info["n_params"],
                       batch_size=info["batch_size"], loss_history=info["loss_history"])
            if cfg.get("finetune"):
                res.update(metrics_val_probe=out["val"]["knn"], ft_seconds=info["ft_seconds"], ft_loss_history=info["ft_loss_history"])
                res.pop("metrics_val_knn")
            if "val_noisy" in out:
                res["metrics_val_noisy"] = out["val_noisy"]["linear"]
            if a.final:
                out2, info2, rank2 = _phase(train, {"test": test}, cfg, seed, a, ck / f"{name}__seed{seed}__final.pt", noisy=True)
                res.update(metrics_test=out2["test"]["linear"], metrics_test_knn=out2["test"]["knn"], per_class_test=out2["test"]["per_class"],
                           effective_rank_final=rank2, train_seconds_final=info2["train_seconds"])
                if cfg.get("finetune"):
                    res["metrics_test_probe"] = res.pop("metrics_test_knn")
                    res["ft_seconds_final"] = info2["ft_seconds"]
                res["collapsed"] = res["collapsed"] or bool(diagnosable and rank2 < cfg["collapse_rank"])
                if "test_noisy" in out2:
                    res["metrics_test_noisy"] = out2["test_noisy"]["linear"]
    except MemoryError as e:
        res.update(status="skipped", reason=str(e))
    write_result(path, res)
    m = res.get("metrics_val", {})
    print(f"[{name} seed{seed}] val macroF1={m.get('f1_macro', float('nan')):.4f} "
          f"test macroF1={res.get('metrics_test', {}).get('f1_macro', float('nan')):.4f} rank={res.get('effective_rank')} "
          f"collapsed={res.get('collapsed')} status={res['status']}", flush=True)
    return res


def plan(a):
    """展開成 [(args, tag)]；--ablate 時依值逐一覆寫單一設定。"""
    if not a.ablate:
        return [(a, "")]
    if not a.values:
        raise SystemExit("--ablate 需要搭配 --values")
    out = []
    attr = {"min_mutations": "min_mutations", "mask_ratio": "mask_ratio"}.get(a.ablate, a.ablate)
    for v in a.values:
        b = copy.copy(a)
        try:
            setattr(b, attr, ABLATE_AXES[a.ablate](v))
        except ValueError:
            raise SystemExit(f"消融軸 {a.ablate} 的取值無法解析：{v}")
        out.append((b, f"__{a.ablate}={v}"))
    return out


def main(argv=None):
    """命令列入口。"""
    a = build_parser().parse_args(argv)
    if not Path(a.data).exists():
        print(f"找不到資料檔 {a.data}", file=sys.stderr)
        return 2
    jobs = plan(a)
    data = D.load_splits(a.data, limit=a.limit)
    for b, tag in jobs:
        if b.anchor_sampling == "balanced" and not a.ablate:
            print("[注意] balanced 錨點取樣只供消融，結果名稱已加標示", flush=True)
            tag = tag or "__anchor_sampling=balanced"
        if not a.ablate:
            tag += ("__encoder=" + b.encoder if b.encoder != "deepsets" else "")
            tag += ("__entangle=False" if not b.entangle else "") + ("__trainable_obs=False" if b.fixed_obs else "")
        if (b.shots or b.noise) and "noise=" not in tag:
            tag += f"__noise=s{b.shots}_p{b.noise}"
        cfg, name = make_cfg(b), config_name(b, tag)
        for s in a.seeds:
            run_one(b, cfg, name, s, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
