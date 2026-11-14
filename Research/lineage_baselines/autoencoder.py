"""稠密自編碼器：把展平的 one-hot 序列壓成 128 維嵌入（研究文件 D2）。

只用 train 訓練；test 只經過已訓練好的編碼器，避免洩漏。
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import ALPHABET

AE_HPARAMS = {
    "encoder_dims": [1024, 512, 256, 128],
    "activation": "LeakyReLU(0.2)",
    "batch_norm": True,
    "dropout": 0.1,
    "optimizer": "Adam",
    "loss": "MSE",
    "epochs": 100,
    "lr": 1e-3,
    "batch_size": 128,
}
AE_PROVENANCE = {
    "encoder_layers": "paper",  # 論文：4 層稠密層、LeakyReLU、BatchNorm、Dropout、對稱解碼器
    "optimizer": "paper",
    "loss": "paper",
    "epochs": "paper",
    "encoder_dims": "chosen",
    "dropout": "chosen",
    "lr": "chosen",
    "batch_size": "chosen",
}


def _block(i, o, drop):
    """一個稠密區塊：線性、BatchNorm、LeakyReLU、Dropout。"""
    return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.LeakyReLU(0.2), nn.Dropout(drop)]


class DenseAE(nn.Module):
    """對稱的稠密自編碼器，輸入為展平的 one-hot。"""

    def __init__(self, in_dim, dims=(1024, 512, 256, 128), drop=0.1):
        """建立編碼器與對稱解碼器。"""
        super().__init__()
        enc, d = [], in_dim
        for h in dims:
            enc += _block(d, h, drop)
            d = h
        dec = []
        for h in list(dims[::-1][1:]) + [in_dim]:
            dec += _block(d, h, drop) if h != in_dim else [nn.Linear(d, h)]
            d = h
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        """回傳（重建，嵌入）。"""
        z = self.encoder(x)
        return self.decoder(z), z


def _onehot_batch(ids, device):
    """把一批 int8 索引轉成展平的 float one-hot。"""
    t = torch.from_numpy(ids).to(device).long()
    return F.one_hot(t, len(ALPHABET)).float().flatten(1)


@torch.no_grad()
def _embed(model, ids, device, chunk=512):
    """分批計算嵌入。"""
    model.eval()
    out = [model.encoder(_onehot_batch(ids[i : i + chunk], device)).cpu().numpy() for i in range(0, len(ids), chunk)]
    return np.concatenate(out).astype(np.float32)


def fit_embed(x_train, x_test, seed, epochs=None, device=None, hp=None):
    """只用 x_train 訓練自編碼器，回傳（train 嵌入，test 嵌入，最後一輪重建損失）。"""
    hp = {**AE_HPARAMS, **(hp or {})}
    epochs = hp["epochs"] if epochs is None else epochs
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = DenseAE(x_train.shape[1] * len(ALPHABET), hp["encoder_dims"], hp["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    n, last = len(x_train), float("nan")
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).numpy()
        tot = 0.0
        for i in range(0, n, hp["batch_size"]):
            idx = perm[i : i + hp["batch_size"]]
            if len(idx) < 2:  # BatchNorm 至少需要 2 筆
                continue
            x = _onehot_batch(x_train[idx], device)
            rec, _ = model(x)
            loss = F.mse_loss(rec, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        last = tot / n
    return _embed(model, x_train, device), _embed(model, x_test, device), last
