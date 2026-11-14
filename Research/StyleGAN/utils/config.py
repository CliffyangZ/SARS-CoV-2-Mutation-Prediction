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

# 兩種規模的預設值：n_qubits、區塊邊長、batch、k（各區塊通道數由 fmap_base / fmap_max 依 reference_gan.py 的 nf(stage) 決定）
SCALES = {
    1: dict(n_qubits=6, L=64, block_sizes=[4, 16, 64], batch_size=16, k=2),
    2: dict(n_qubits=10, L=1024, block_sizes=[4, 16, 64, 256, 1024], batch_size=8, k=3),
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
    fmap_base: int = 8192               # reference_gan.py：nf(stage)=min(fmap_base/2^stage, fmap_max)
    fmap_max: int = 512
    k: int = 2
    gamma1: float = 10.0                # 式 5 的 R1 型梯度懲罰（古典 D）
    gamma2: float = 0.0
    r1_rahinge: bool = False            # 是否對 rahinge 的 D 也加 R1 懲罰（對治 D 損失趨近 0 的過強判別器）
    ema_decay: float = 0.0              # 生成器 EMA 衰減率；0 表示關閉
    noise: bool = True                  # 是否在 synthesis 區塊加入 noise injection（reference 預設 use_noise=True）
    encoding: dict = field(default_factory=dict)   # EncodingSpec 的 dict；空表示 legacy（舊 config 相容）
    train_end: str = "2021-02-28"
    test_start: str = "2021-04-01"
    lr_scale: float = 1.0               # 003：G、D 學習率同乘此值（實際學習率 = lr × lr_scale）
    d_every: int = 1                    # 003：每 d_every 步才更新一次判別器（1 為每步更新）
    last_block_steps: int = None        # 003：最後區塊的訓練步數；None＝與其他區塊均分（其餘區塊步數不動）
    fade_frac: float = 0.5              # 003：淡入長度占區塊長度的比例
    tag: str = ""                       # 003：附加在 run 目錄名稱的設定標籤（空字串不加）
    g_fmap_max: int = None              # 004：生成器特徵圖寬度上限；None＝同 fmap_max
    d_fmap_max: int = None              # 004：判別器特徵圖寬度上限；None＝同 fmap_max
    mbstd_group: int = 4                # 004：判別器 minibatch 標準差的群組大小（預設 4＝reference）
    div_reg: float = 0.0                # 004：多樣性正則權重（批次內兩兩差異式）；0＝關閉，不多做前向
    git_commit: str = ""

    @property
    def block_schedule(self):
        """各區塊的訓練步數區間 [(start, end), ...]：平均分配（R7）。
        last_block_steps 不為 None 時，最後區塊改為該步數，其餘區塊維持均分的區間（總步數隨之改變）。"""
        n, per = len(self.block_sizes), self.steps / len(self.block_sizes)
        sch = [(round(i * per), round((i + 1) * per)) for i in range(n)]
        if self.last_block_steps is not None:
            start = sch[-1][0]
            sch[-1] = (start, start + int(self.last_block_steps))
        return sch

    @property
    def total_steps(self):
        """實際訓練步數＝最後一個區塊的結束步（未指定 last_block_steps 時等於 steps）。"""
        return self.block_schedule[-1][1]

    @property
    def enc(self):
        """本次實驗的 EncodingSpec（延後匯入，避免設定模組一載入就匯入 torch）。"""
        from Research.StyleGAN.FE.encoding import spec_from_dict
        return spec_from_dict(self.encoding)

    def run_dir(self, root="Results"):
        """輸出目錄 Results/<階段>/scale<N>/<model>_<loss>[_<編碼>]_seed<S>/；legacy 不加編碼後綴（與 run01 相容）。"""
        enc = "" if self.enc.name == "legacy" else f"_{self.enc.name}"
        tag = f"_{self.tag}" if self.tag else ""
        return Path(root) / self.stage / f"scale{self.scale}" / f"{self.model}_{self.loss}{enc}{tag}_seed{self.seed}"


def make_config(scale=1, encoding="legacy", **kw):
    """依規模取預設值，再用關鍵字參數覆寫；encoding 為編碼名稱（見 Research.StyleGAN.FE.encoding.get_encoding）。"""
    from Research.StyleGAN.FE.encoding import get_encoding
    base = dict(SCALES[scale], scale=scale, encoding=get_encoding(encoding).to_dict())
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


def get_device(pref="auto"):
    """選運算裝置：auto＝有 CUDA 就用 GPU，否則 CPU；指定 cuda 但不可用時直接報錯（不默默退回 CPU）。"""
    import torch
    if pref == "auto":
        pref = "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("要求使用 GPU，但 torch.cuda.is_available() 為 False（請確認安裝的是 CUDA 版 torch）")
    return torch.device(pref)


def save_config(cfg, path):
    """把設定寫成 config.json（含 block_schedule）。"""
    d = asdict(cfg); d["block_schedule"] = cfg.block_schedule
    Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False))


def load_config(path):
    """讀回 config.json。"""
    d = json.loads(Path(path).read_text()); d.pop("block_schedule", None)
    d["betas"] = tuple(d["betas"])
    return Config(**d)
