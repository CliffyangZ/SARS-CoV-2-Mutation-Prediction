"""生成器（古典 / 量子判別器共用，FR-008）：8 層 FC mapping ＋ 漸進式 synthesis。

每個 synthesis 區塊：(放大×4) → conv → AdaIN → conv → AdaIN；
區塊邊長 4、16、64（、256、1024）對應判別器 2、4、6（、8、10）qubit。
輸出單通道特徵圖 G，再由 encoding.featuremap_to_rho 轉為合法密度矩陣 ρ（R6）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Research.FE.encoding import featuremap_to_rho


class AdaIN(nn.Module):
    """自適應實例正規化：以風格向量 w 產生通道的縮放與位移（StyleGAN）。"""

    def __init__(self, ch, w_dim):
        super().__init__()
        self.style = nn.Linear(w_dim, ch * 2)

    def forward(self, x, w):
        """以風格 w 調變實例正規化後的特徵。"""
        s, b = self.style(w).chunk(2, dim=1)
        return F.instance_norm(x) * (1 + s[:, :, None, None]) + b[:, :, None, None]


class SynthBlock(nn.Module):
    """一個 synthesis 區塊（每塊 2 個卷積，論文）。noise=True 時在每次卷積後加入 noise injection（StyleGAN）。"""

    def __init__(self, c_in, c_out, w_dim, up, noise=False):
        super().__init__()
        self.up, self.noise = up, noise
        self.c1, self.c2 = nn.Conv2d(c_in, c_out, 3, padding=1), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.n1, self.n2 = AdaIN(c_out, w_dim), AdaIN(c_out, w_dim)
        if noise:   # 每通道的雜訊強度，初始為 0，讓模型自行學習要不要用雜訊
            self.nw1 = nn.Parameter(torch.zeros(1, c_out, 1, 1))
            self.nw2 = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def _add_noise(self, x, nw):
        """加入單通道的逐像素高斯雜訊，強度為學習到的 nw。"""
        if not self.noise:
            return x
        return x + torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) * nw

    def forward(self, x, w):
        """（放大）→ 卷積 → 雜訊 → AdaIN，共兩次。"""
        if self.up:
            x = F.interpolate(x, scale_factor=4, mode="nearest")
        x = self.n1(self._add_noise(F.leaky_relu(self.c1(x), 0.2), getattr(self, "nw1", None)), w)
        return self.n2(self._add_noise(F.leaky_relu(self.c2(x), 0.2), getattr(self, "nw2", None)), w)


class Generator(nn.Module):
    """漸進式 StyleGAN 生成器；forward(z, level, alpha) → 密度矩陣 ρ (B, S, S)。"""

    def __init__(self, block_sizes, channels, latent_dim=64, mapping_layers=8, noise=False):
        super().__init__()
        self.sizes = list(block_sizes)
        layers = []
        for _ in range(mapping_layers):              # mapping network：8 層 FC（論文）
            fc = nn.Linear(latent_dim, latent_dim)
            nn.init.kaiming_normal_(fc.weight, a=0.2, nonlinearity="leaky_relu")   # 保持 8 層後訊號變異不衰減
            nn.init.zeros_(fc.bias)
            layers += [fc, nn.LeakyReLU(0.2)]
        self.mapping = nn.Sequential(*layers)
        self.const = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.blocks = nn.ModuleList(
            SynthBlock(channels[max(i - 1, 0)], channels[i], latent_dim, up=i > 0, noise=noise)
            for i in range(len(channels)))
        self.to_map = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in channels)   # 通道 → 單通道特徵圖

    def param_groups(self, lr, mapping_lr_mult=0.01):
        """優化器參數分組：mapping network 的學習率乘 0.01（StyleGAN 慣例，避免風格向量發散）。"""
        m = list(self.mapping.parameters()); ids = {id(p) for p in m}
        return [{"params": m, "lr": lr * mapping_lr_mult},
                {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr}]

    def feature_map(self, z, level, alpha):
        """回傳當前解析度的特徵圖 G (B,1,S,S)；淡入期與上一層輸出混合。"""
        w = self.mapping(z / (z.norm(dim=1, keepdim=True) + 1e-8) * z.shape[1] ** 0.5)
        x, prev = self.const.expand(z.shape[0], -1, -1, -1), None
        for i in range(level + 1):
            if i == level:
                prev = self.to_map[i - 1](x) if i > 0 else None
            x = self.blocks[i](x, w)
        g = self.to_map[level](x)
        if prev is not None and alpha < 1.0:
            g = alpha * g + (1 - alpha) * F.interpolate(prev, scale_factor=4, mode="nearest")
        return g

    def forward(self, z, level, alpha=1.0):
        """潛在向量 z → 特徵圖 G → 密度矩陣 ρ = GGᵀ / Tr(GGᵀ)（R6）。"""
        return featuremap_to_rho(self.feature_map(z, level, alpha))
