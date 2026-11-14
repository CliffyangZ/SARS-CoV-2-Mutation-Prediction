"""深度 baseline 模型：3／4 層 CNN、ResNet34（影像），FCN、雙向 LSTM、mWDN（訊號）。

論文只給層數，其餘結構為自行選用（研究文件 D6），逐項記錄於結果檔的 provenance。
"""
from pathlib import Path

import torch
from torch import nn


def make_cnn(n_layers, n_classes, in_ch=3):
    """n_layers 層 Conv3×3+BatchNorm+ReLU+MaxPool，通道 32→64→128（→256），全域平均池化加線性分類。"""
    chans = [32, 64, 128, 256][:n_layers]
    layers, c = [], in_ch
    for o in chans:
        layers += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        c = o
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_classes))


class _ImageNetNorm(nn.Module):
    """套用 ImageNet 的均值與標準差正規化後交給內部模型（預訓練權重的輸入慣例）。"""

    def __init__(self, inner):
        """包住內部模型並註冊均值與標準差。"""
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """x 為 [0, 1] 的 (B,3,H,W) 影像。"""
        return self.inner((x - self.mean) / self.std)


PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "pretrained"
RESNET34_FILE = PRETRAINED_DIR / "resnet34-b627a593.pth"  # torchvision ResNet34_Weights.IMAGENET1K_V1


def make_resnet34(n_classes, pretrained=True):
    """torchvision 的 ResNet34；pretrained 為真時載入 ImageNet 權重並加上 ImageNet 正規化。

    權重優先讀專案內的 pretrained/ 資料夾；檔案不存在才退回 torchvision 的線上下載。
    """
    import torchvision

    m = torchvision.models.resnet34(weights=None)
    if pretrained:
        if RESNET34_FILE.exists():
            m.load_state_dict(torch.load(RESNET34_FILE, map_location="cpu", weights_only=True))
        else:
            m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return _ImageNetNorm(m) if pretrained else m


class FCN(nn.Module):
    """一維全卷積網路（Wang 等人的標準 FCN）：三個卷積區塊 128／256／128，核 8／5／3。"""

    def __init__(self, n_classes):
        """建立三個 Conv1d+BatchNorm+ReLU 區塊與分類層。"""
        super().__init__()
        blocks, c = [], 1
        for o, k in ((128, 8), (256, 5), (128, 3)):
            blocks += [nn.Conv1d(c, o, k, padding="same"), nn.BatchNorm1d(o), nn.ReLU()]
            c = o
        self.body, self.head = nn.Sequential(*blocks), nn.Linear(c, n_classes)

    def forward(self, x):
        """x 形狀 (B, L)。"""
        return self.head(self.body(x[:, None, :]).mean(-1))


class BiLSTM(nn.Module):
    """3 層雙向 LSTM（隱藏 128），取最後一層正反向的最終隱藏狀態接線性分類。"""

    def __init__(self, n_classes, hidden=128, layers=3):
        """建立 LSTM 與分類層。"""
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        """x 形狀 (B, L)。"""
        _, (h, _) = self.rnn(x[:, :, None])
        return self.head(torch.cat([h[-2], h[-1]], dim=1))


class MWDN(nn.Module):
    """多層小波分解網路（Wang 等人 mWDN 的精簡版）：可訓練的低通與高通濾波加 LSTM 頭。

    每層以步幅 2 的一維卷積做分解，濾波器以 Haar 初始化；各層的高通輸出與最後的低通輸出
    各送入一個 LSTM，取最後隱藏狀態串接後分類。
    """

    def __init__(self, n_classes, levels=3, hidden=64):
        """建立分解層與各分支的 LSTM。"""
        super().__init__()
        self.low = nn.ModuleList([nn.Conv1d(1, 1, 2, stride=2, bias=False) for _ in range(levels)])
        self.high = nn.ModuleList([nn.Conv1d(1, 1, 2, stride=2, bias=False) for _ in range(levels)])
        with torch.no_grad():
            for lo, hi in zip(self.low, self.high):
                lo.weight.copy_(torch.tensor([[[0.5, 0.5]]]))
                hi.weight.copy_(torch.tensor([[[0.5, -0.5]]]))
        self.rnns = nn.ModuleList([nn.LSTM(1, hidden, batch_first=True) for _ in range(levels + 1)])
        self.head = nn.Linear(hidden * (levels + 1), n_classes)

    def forward(self, x):
        """x 形狀 (B, L)。"""
        a, feats = x[:, None, :], []
        for lo, hi in zip(self.low, self.high):
            feats.append(hi(a))
            a = lo(a)
        feats.append(a)
        outs = [rnn(f.transpose(1, 2))[1][0][-1] for rnn, f in zip(self.rnns, feats)]
        return self.head(torch.cat(outs, dim=1))


def build(model, n_classes, pretrained=True):
    """依名稱建立深度模型。"""
    if model == "cnn3":
        return make_cnn(3, n_classes)
    if model == "cnn4":
        return make_cnn(4, n_classes)
    if model == "resnet34":
        return make_resnet34(n_classes, pretrained)
    if model == "fcn":
        return FCN(n_classes)
    if model == "lstm3bi":
        return BiLSTM(n_classes)
    if model == "mwdn":
        return MWDN(n_classes)
    raise ValueError(f"未知深度模型：{model}")


# 各模型的預設批次大小與超參數說明（逐項標 paper 或 chosen）
DEFAULT_BATCH = {"cnn3": 32, "cnn4": 32, "resnet34": 16, "fcn": 64, "lstm3bi": 64, "mwdn": 64}
ARCH_NOTES = {
    "cnn3": {"layers": 3, "channels": [32, 64, 128], "kernel": 3, "pool": "max2", "head": "global_avg+linear"},
    "cnn4": {"layers": 4, "channels": [32, 64, 128, 256], "kernel": 3, "pool": "max2", "head": "global_avg+linear"},
    "resnet34": {"torchvision": "resnet34", "weights": "IMAGENET1K_V1（預訓練時）", "input_norm": "ImageNet 均值與標準差", "fc": "replaced"},
    "fcn": {"blocks": [[128, 8], [256, 5], [128, 3]], "head": "global_avg+linear"},
    "lstm3bi": {"layers": 3, "hidden": 128, "bidirectional": True},
    "mwdn": {"levels": 3, "hidden": 64, "init": "haar", "decomposition_kernel": 2},
}
ARCH_PROVENANCE = {
    "cnn3": {"layers": "paper", "channels": "chosen", "kernel": "chosen", "pool": "chosen", "head": "chosen"},
    "cnn4": {"layers": "paper", "channels": "chosen", "kernel": "chosen", "pool": "chosen", "head": "chosen"},
    "resnet34": {"torchvision": "paper", "weights": "chosen（論文只說預訓練，未說資料集）", "input_norm": "chosen", "fc": "chosen"},
    "fcn": {"blocks": "chosen", "head": "chosen"},
    "lstm3bi": {"layers": "paper", "hidden": "chosen", "bidirectional": "paper"},
    "mwdn": {"levels": "chosen", "hidden": "chosen", "init": "chosen", "decomposition_kernel": "chosen"},
}
