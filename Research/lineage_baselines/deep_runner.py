"""把深度模型接到 run.py：準備輸入轉換、訓練、評估並組成結果字典。"""
from pathlib import Path

import numpy as np
import torch

from . import deep_models as DM
from . import metrics as M
from . import trainer as TR
from . import transforms as T


RECIPE_VERSION = 2  # 2：評估前重新校準 BatchNorm；循環模型加梯度裁剪。版本 1 的深度結果不可信，會被重新評估


def make_input_fn(rep):
    """回傳 ids(np.int8) → 模型輸入張量的函式：影像為 (B,3,360,360)，訊號為 (B,L)。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_input(ids):
        """一批符號索引 → 訊號或影像張量（在 GPU 上即時產生）。"""
        sig = T.signal_from_ids(torch.from_numpy(ids).to(dev))
        return sig if rep == "signal" else T.to_image(rep, sig)

    return _to_input


def run_one(model, rep, seed, prep, args, ctx):
    """訓練並評估一個深度模型（單一種子）；未訓練完則回報 incomplete。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_cls = len(prep.labels)
    pretrained = bool(getattr(args, "pretrained", True)) and model == "resnet34"
    net = DM.build(model, n_cls, pretrained=pretrained)
    bs = args.batch_size or DM.DEFAULT_BATCH[model]
    clip = 1.0 if model in ("lstm3bi", "mwdn") else None  # 循環模型易爆炸：LSTM 實測第 85 輪權重變 NaN
    ckpt = Path(args.out) / "checkpoints" / f"{model}__{rep}__seed{seed}.pt"
    r = TR.fit(net, prep.x_train, prep.y_train, prep.x_test, input_fn=make_input_fn(rep), epochs=args.epochs,
               batch_size=bs, lr=1e-3, seed=seed, ckpt_path=ckpt, resume=args.resume,
               eval_batch_size=bs, stop_after=getattr(args, "stop_after", None), clip_grad=clip)
    hp = {"epochs": args.epochs, "batch_size": bs, "lr": 1e-3, "optimizer": "Adam", "loss": "NLL",
          "amp": "bfloat16", "pretrained_imagenet": pretrained, "grad_clip_norm": clip, "bn_recalibration_samples": 4096, "architecture": DM.ARCH_NOTES[model]}
    prov = {"optimizer": "paper", "loss": "paper", "epochs": "chosen (論文未說明深度 baseline 的輪數，取與 SNN 相同的 100)",
            "lr": "chosen", "batch_size": "chosen", "amp": "chosen", "grad_clip_norm": "chosen", "bn_recalibration_samples": "chosen（評估前以 train 子集重算 BatchNorm 統計）", "architecture": DM.ARCH_PROVENANCE[model]}
    out = {"recipe_version": RECIPE_VERSION, "hyperparameters": hp, "provenance": prov, "status": r["status"], "epochs_done": r["epochs_done"], "skipped_steps": r.get("skipped_steps", 0),
           "train_seconds": r["train_seconds"], "train_losses": r["train_losses"], "peak_mem_mb": r["peak_mem_mb"]}
    if r["status"] == "complete":
        proba = r["proba"]
        pred = proba.argmax(1)
        out["metrics"] = M.compute_metrics(prep.y_test, pred, proba, n_cls)
        out["per_class"] = M.per_class_report(prep.y_test, pred, n_cls, prep.labels)
    else:
        out["metrics"] = {"accuracy": float("nan"), "f1_macro": float("nan")}
    return out
