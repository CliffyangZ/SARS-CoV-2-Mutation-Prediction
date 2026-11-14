"""深度模型訓練迴圈：混合精度、每輪檢查點、可續跑、累計牆鐘耗時。

規則（規格 FR-012、研究文件 D6、D10）：
- 優化器 Adam、損失 negative log-likelihood（log_softmax 加 nll_loss）；
- 每個 epoch 的洗牌順序由（seed, epoch）決定，使續跑與不中斷的結果一致；
- 只用最後一輪的模型在 test 上評估一次，不用 test 挑檢查點或調參。
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def _save(path, state):
    """原子寫入檢查點。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def recalibrate_bn(model, x_train, input_fn, device, batch_size, n_samples, seed, amp):
    """用 train 資料的隨機子集重算所有 BatchNorm 的統計（累積平均，不更新權重），回傳使用的樣本數。

    只使用 train 資料，不會洩漏 test。模型沒有 BatchNorm 或 n_samples<=0 時不做任何事。
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns or n_samples <= 0:
        return 0
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # 累積平均
    idx = np.sort(np.random.default_rng([seed, 999_983]).choice(len(x_train), min(n_samples, len(x_train)), replace=False))
    model.train()
    for i in range(0, len(idx), batch_size):
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            model(input_fn(x_train[idx[i : i + batch_size]]).to(device))
    for m, mo in zip(bns, saved):
        m.momentum = mo
    model.eval()
    return len(idx)


@torch.no_grad()
def predict_proba(model, x, input_fn, device, batch_size, amp):
    """分批計算 softmax 機率 (n, K)。"""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            logits = model(input_fn(x[i : i + batch_size]).to(device))
        out.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def fit(model, x_train, y_train, x_test, *, input_fn, epochs, batch_size, lr, seed, ckpt_path,
        resume=True, device=None, amp=True, stop_after=None, eval_batch_size=None,
        clip_grad=None, bn_recal_samples=4096):
    """訓練 model 並在最後於 x_test 上輸出機率。

    回傳字典：status（complete 或 incomplete）、epochs_done、train_seconds、train_losses、
    proba（僅 complete）、peak_mem_mb、bn_recal_samples。clip_grad 為梯度範數上限（None 代表不裁剪）。stop_after 為測試用：訓練到該輪後停止並回報 incomplete。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = amp and device.startswith("cuda")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start, elapsed, losses = 0, 0.0, []
    ckpt_path = Path(ckpt_path)
    if resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, elapsed, losses = st["epoch"], st["elapsed"], st["losses"]
        torch.set_rng_state(st["rng_cpu"].cpu())
        if device.startswith("cuda") and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"].cpu())
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    n = len(x_train)
    skipped = 0
    n_steps = -(-n // batch_size) * max(epochs, 1)
    y_t =torch.from_numpy(np.asarray(y_train)).long()
    for epoch in range(start, epochs):
        if stop_after is not None and epoch >= stop_after:
            break
        t0 = time.time()
        model.train()
        perm = np.random.default_rng([seed, epoch]).permutation(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = np.sort(perm[i : i + batch_size])
            xb = input_fn(x_train[idx]).to(device)
            yb = y_t[idx].to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
                logits = model(xb)
            loss = F.nll_loss(F.log_softmax(logits.float(), dim=1), yb)
            opt.zero_grad(set_to_none=True)
            ok = bool(torch.isfinite(loss))
            if ok:
                loss.backward()
                # 範數不論是否裁剪都要檢查：bf16 偶爾產生 inf/NaN 梯度，直接更新會污染全部權重
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad or float("inf"))
                ok = bool(torch.isfinite(gn))
            if not ok:
                skipped += 1
                if skipped > max(10, n_steps // 100):  # 超過總步數 1% 才視為真正發散
                    raise FloatingPointError(f"第 {epoch + 1} 輪：累計跳過 {skipped} 步非有限的損失或梯度，判定發散；檢查點停在第 {epoch} 輪")
                continue
            opt.step()
            tot += loss.item() * len(idx)
        elapsed += time.time() - t0
        losses.append(tot / n)
        _save(ckpt_path, {
            "model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1, "elapsed": elapsed,
            "losses": losses, "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device.startswith("cuda") else None,
        })
    done = len(losses)
    out = {"epochs_done": done, "skipped_steps": skipped, "train_seconds": elapsed, "train_losses": losses,
           "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None}
    if done >= epochs:
        out["status"] = "complete"
        out["bn_recal_samples"] = recalibrate_bn(model, x_train, input_fn, device, batch_size, bn_recal_samples, seed, amp)
        out["proba"] = predict_proba(model, x_test, input_fn, device, eval_batch_size or batch_size, amp)
    else:
        out["status"] = "incomplete"
    return out
