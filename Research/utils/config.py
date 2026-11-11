"""設定與隨機種子（對應 contracts/cli.md 的設定契約）。

規模一：6 qubit、L=64、3 區塊；規模二：10 qubit、L=1024、5 區塊（論文規模）。
超參數大多是論文未說明、由本專案假設的值（research R7）。
"""
import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# 兩種規模的預設值：n_qubits、區塊邊長、各區塊通道數（高解析度壓到 4–8，R6）、batch、k
SCALES = {
    1: dict(n_qubits=6, L=64, block_sizes=[4, 16, 64], channels=[16, 16, 8], batch_size=16, k=2),
    2: dict(n_qubits=10, L=1024, block_sizes=[4, 16, 64, 256, 1024],
            channels=[16, 16, 8, 4, 4], batch_size=8, k=3),
}


@dataclass
class Config:
    """一次實驗（run）的全部設定；同步寫入 config.json 與 wandb。"""
    stage: str = "A"
    scale: int = 1
    model: str = "classical"            # classical | quantum
    loss: str = "logistic"              # logistic | rahinge
    seed: int = 0
    steps: int = 3500                   # 論文：3500 步
    log_every: int = 74                 # 論文：每 74 步記錄（約 47 點）
    batch_size: int = 16
    lr_g: float = 2e-3
    lr_d: float = 2e-3
    betas: tuple = (0.0, 0.99)
    latent_dim: int = 64
    mapping_layers: int = 8             # 論文：mapping network 8 層 FC
    n_qubits: int = 6
    L: int = 64
    block_sizes: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    k: int = 2
    gamma1: float = 10.0                # 式 5 的 R1 型梯度懲罰（古典 D）
    gamma2: float = 0.0
    r1_rahinge: bool = False            # 是否對 rahinge 的 D 也加 R1 懲罰（對治 D 損失趨近 0 的過強判別器）
    ema_decay: float = 0.0              # 生成器 EMA 衰減率；0 表示關閉
    noise: bool = False                 # 是否在 synthesis 區塊加入 noise injection
    train_end: str = "2021-02-28"
    test_start: str = "2021-04-01"
    git_commit: str = ""

    @property
    def block_schedule(self):
        """各區塊的訓練步數區間 [(start, end), ...]：平均分配（R7）。"""
        n, per = len(self.block_sizes), self.steps / len(self.block_sizes)
        return [(round(i * per), round((i + 1) * per)) for i in range(n)]

    def run_dir(self, root="Results"):
        """輸出目錄 Results/<階段>/scale<N>/<model>_<loss>_seed<S>/（R16）。"""
        return Path(root) / self.stage / f"scale{self.scale}" / f"{self.model}_{self.loss}_seed{self.seed}"


def make_config(scale=1, **kw):
    """依規模取預設值，再用關鍵字參數覆寫。"""
    base = dict(SCALES[scale], scale=scale)
    base.update(kw)
    cfg = Config(**base)
    try:
        cfg.git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                 stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    return cfg


def set_seed(seed):
    """固定 python / numpy / torch 的隨機種子。"""
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def save_config(cfg, path):
    """把設定寫成 config.json（含 block_schedule）。"""
    d = asdict(cfg); d["block_schedule"] = cfg.block_schedule
    Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False))


def load_config(path):
    """讀回 config.json。"""
    d = json.loads(Path(path).read_text()); d.pop("block_schedule", None)
    d["betas"] = tuple(d["betas"])
    return Config(**d)
