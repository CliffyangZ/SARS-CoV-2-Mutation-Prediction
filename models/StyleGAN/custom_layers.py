"""自訂層（reference_gan.py 的 models.CustomLayers）：equalized lr 的 Linear / Conv、PixelNorm、
Noise、StyleMod、LayerEpilogue、Blur、minibatch-stddev、Truncation。"""
import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNormLayer(nn.Module):
    """潛碼歸一化：x ← x / sqrt(mean(x²) + ε)。"""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


def _equalized_init(shape, fan_in, gain, use_wscale, lrmul):
    """equalized learning rate：回傳 (初始權重, 執行期權重乘數)。"""
    he_std = gain * fan_in ** (-0.5)
    if use_wscale:
        return torch.randn(*shape) / lrmul, he_std * lrmul
    return torch.randn(*shape) * he_std / lrmul, lrmul


class EqualizedLinear(nn.Module):
    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1.0, bias=True):
        super().__init__()
        w, self.w_mul = _equalized_init((output_size, input_size), input_size, gain, use_wscale, lrmul)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(output_size)) if bias else None
        self.b_mul = lrmul

    def forward(self, x):
        bias = self.bias * self.b_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.w_mul, bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, gain=2 ** 0.5, use_wscale=False,
                 lrmul=1.0, bias=True):
        super().__init__()
        w, self.w_mul = _equalized_init((output_channels, input_channels, kernel_size, kernel_size),
                                        input_channels * kernel_size ** 2, gain, use_wscale, lrmul)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(output_channels)) if bias else None
        self.b_mul = lrmul
        self.pad = kernel_size // 2

    def forward(self, x):
        bias = self.bias * self.b_mul if self.bias is not None else None
        return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.pad)


class NoiseLayer(nn.Module):
    """逐像素高斯雜訊，通道強度可學習（初始 0）。"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise


class StyleMod(nn.Module):
    """以風格 w 產生通道縮放與位移（AdaIN 的調變部分）。"""

    def __init__(self, latent_size, channels, use_wscale):
        super().__init__()
        self.lin = EqualizedLinear(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent).view(-1, 2, x.size(1), 1, 1)
        return x * (style[:, 0] + 1.0) + style[:, 1]


class LayerEpilogue(nn.Module):
    """每個卷積後：noise → 激活 → pixel norm → instance norm → style mod。"""

    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm,
                 use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.noise = NoiseLayer(channels) if use_noise else None
        self.act = activation_layer
        self.pixel_norm = PixelNormLayer() if use_pixel_norm else None
        self.instance_norm = nn.InstanceNorm2d(channels) if use_instance_norm else None
        self.style_mod = StyleMod(dlatent_size, channels, use_wscale) if use_styles else None

    def forward(self, x, dlatents_in_slice=None):
        if self.noise is not None:
            x = self.noise(x)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        return x


class BlurLayer(nn.Module):
    """固定（不可訓練）的 depthwise 低通濾波，預設 [1,2,1]。"""

    def __init__(self, kernel=(1, 2, 1), normalize=True):
        super().__init__()
        k = torch.tensor(kernel, dtype=torch.float32)
        k = k[None, :] * k[:, None]
        if normalize:
            k = k / k.sum()
        self.register_buffer("kernel", k[None, None])

    def forward(self, x):
        c = x.size(1)
        return F.conv2d(x, self.kernel.expand(c, -1, -1, -1), padding=(self.kernel.size(2) - 1) // 2, groups=c)


class StddevLayer(nn.Module):
    """minibatch standard deviation：附加 num_new_features 個跨組標準差通道（批次大小需可被組數整除）。"""

    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size, self.num_new_features = group_size, num_new_features

    def forward(self, x):
        b, c, h, w = x.shape
        g = min(self.group_size, b)
        y = x.reshape(g, -1, self.num_new_features, c // self.num_new_features, h, w)
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)
        y = y.expand(g, -1, -1, h, w).clone().reshape(b, self.num_new_features, h, w)
        return torch.cat([x, y], dim=1)


class Truncation(nn.Module):
    """截斷技巧：前 max_layer 層的 w ← lerp(w_avg, w, threshold)；w_avg 為訓練時的滑動平均。"""

    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer, self.threshold, self.beta = max_layer, threshold, beta
        self.register_buffer("avg_latent", avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1.0 - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1), device=x.device) < self.max_layer).view(1, -1, 1)
        return torch.where(do_trunc, interp, x)
