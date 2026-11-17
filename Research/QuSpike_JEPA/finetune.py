"""監督式 fine-tune：在預訓練後的編碼器＋預測頭（量子或古典）之後接線性分類層，整體以交叉熵訓練。

不做早停（固定 epoch 數），validation 與 test 階段用同一套設定，避免以 test 調參。
"""
import time

import numpy as np
import torch
from torch import nn

from Research.lineage_baselines.metrics import compute_metrics, per_class_report

from .dataset import N_SYM, mutation_vocab
from .pretrain import effective_batch

FT_DEFAULTS = dict(ft_epochs=30, ft_lr=1e-3, ft_head_lr=1e-2, ft_wd=1e-4, ft_balanced=False, ft_freeze=False,
                   channels="q", ft_wd_skip=1e-2)
CHANNELS = ("q", "skip", "both")  # §3.2 跳接消融：只有 W_q／只有 W_s／兩者


class Classifier(nn.Module):
    """編碼器 → 預測頭（q，m 維）＋ 突變指標跳接 → 線性層（類別數）。

    §3.2 修訂：`logits = W_q·q + W_s·x_mut + b`。channels='skip' 時 q 路徑不參與（監督線性參考線），
    'q' 時退回原本只有量子/古典路徑的設定，'both' 為跳接架構。
    """

    def __init__(self, model, n_classes, lut=None, n_vocab=0, channels="q"):
        super().__init__()
        if channels not in CHANNELS:
            raise ValueError(f"未知的 channels：{channels}")
        self.model, self.channels = model, channels
        d = model.pred.C.out_features
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, n_classes) if channels in ("q", "both") else None
        self.skip = nn.Linear(n_vocab, n_classes, bias=False) if channels in ("skip", "both") else None
        if self.skip is not None:
            self.register_buffer("lut", torch.as_tensor(lut))
            self.n_vocab = n_vocab

    def mut_indicator(self, pos, sym, valid):
        """(B, V) 的 0/1 突變指標（排除 X 與詞彙外 token）。"""
        ids = torch.where(valid, self.lut[pos * N_SYM + sym], torch.full_like(pos, -1))
        out = torch.zeros(pos.shape[0], self.n_vocab, device=pos.device)
        return out.scatter_add_(1, ids.clamp(min=0), (ids >= 0).float()).clamp_(max=1.0)

    def forward(self, pos, sym, valid, **kw):
        """回傳 logits。"""
        out = 0.0
        if self.fc is not None:
            q, _ = self.model.embed(pos, sym, valid, **kw)
            out = out + self.fc(self.norm(q))
        if self.skip is not None:
            out = out + self.skip(self.mut_indicator(pos, sym, valid))
        return out


def _batch(ts, idx, device):
    return (torch.as_tensor(ts.pos[idx].astype(np.int64), device=device),
            torch.as_tensor(ts.sym[idx].astype(np.int64), device=device),
            torch.as_tensor(ts.valid[idx], device=device))


def finetune(model, ts, cfg, seed, device="cpu", log=print):
    """在 ts 上 fine-tune；回傳 (Classifier, info)。cfg 需含 FT_DEFAULTS 的鍵。"""
    n_cls = len(ts.labels)
    ch = cfg.get("channels", "q")
    lut, n_vocab = mutation_vocab(ts) if ch in ("skip", "both") else (None, 0)
    torch.manual_seed(int(seed))
    clf = Classifier(model, n_cls, lut=lut, n_vocab=n_vocab, channels=ch).to(device)
    if clf.fc is not None:
        nn.init.normal_(clf.fc.weight, std=0.01)
        nn.init.zeros_(clf.fc.bias)
    if clf.skip is not None:
        nn.init.zeros_(clf.skip.weight)
    if cfg["ft_freeze"]:
        for p in model.parameters():
            p.requires_grad_(False)
    body = [p for p in model.parameters() if p.requires_grad] if clf.fc is not None else []
    head = list(clf.norm.parameters()) + (list(clf.fc.parameters()) if clf.fc is not None else [])
    groups = ([dict(params=head, lr=cfg["ft_head_lr"])] if head else []) + ([dict(params=body, lr=cfg["ft_lr"])] if body else [])
    # 跳接權重另給較強的 weight decay（§3.2 風險：W_s 容量大，會搶走梯度）
    if clf.skip is not None:
        groups.append(dict(params=list(clf.skip.parameters()), lr=cfg["ft_head_lr"], weight_decay=cfg.get("ft_wd_skip", 1e-2)))
    opt = torch.optim.AdamW(groups, weight_decay=cfg["ft_wd"])
    epochs = cfg["ft_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    w = None
    if cfg["ft_balanced"]:
        c = np.bincount(ts.y, minlength=n_cls).astype(np.float32)
        w = torch.as_tensor(c.sum() / (n_cls * np.maximum(c, 1)), device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    y = torch.as_tensor(np.asarray(ts.y, np.int64), device=device)
    bs = effective_batch(cfg)
    rng = np.random.default_rng([seed, 4242])
    t0, hist = time.time(), []
    for ep in range(epochs):
        clf.train()
        order = rng.permutation(len(ts))
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            if len(idx) < 2:
                continue
            loss = lossf(clf(*_batch(ts, idx, device)), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in clf.parameters() if p.requires_grad], cfg.get("clip", 1.0))
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        hist.append(tot / max(nb, 1))
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            log(f"ft epoch {ep + 1}/{epochs} loss={hist[-1]:.4f} t={time.time() - t0:.0f}s")
    return clf, dict(ft_seconds=time.time() - t0, ft_loss_history=hist, ft_batch_size=bs)


@torch.no_grad()
def predict_proba(clf, ts, device="cpu", batch=512, **kw):
    """回傳 (n, 類別數) 的機率。kw 傳給量子頭（shots、noise）。"""
    clf.eval()
    out = []
    for s in range(0, len(ts), batch):
        idx = np.arange(s, min(len(ts), s + batch))
        out.append(torch.softmax(clf(*_batch(ts, idx, device), **kw), 1).cpu().numpy())
    return np.concatenate(out)


def evaluate(clf, ts, device="cpu", **kw):
    """回傳 dict(metrics, per_class)。"""
    n_cls = len(ts.labels)
    p = predict_proba(clf, ts, device, **kw)
    pred = p.argmax(1)
    return dict(metrics=compute_metrics(ts.y, pred, p, n_cls), per_class=per_class_report(ts.y, pred, n_cls, ts.labels))
