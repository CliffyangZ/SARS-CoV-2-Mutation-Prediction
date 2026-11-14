"""區塊（reference_gan.py 的 models.Blocks）：生成器 InputBlock / GSynthesisBlock，
判別器 DiscriminatorBlock / DiscriminatorTop。每個漸進階段邊長 ×4（上取樣 / 平均池化 4）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.StyleGAN.custom_layers import (BlurLayer, EqualizedConv2d, EqualizedLinear, LayerEpilogue,
                                           StddevLayer)


class InputBlock(nn.Module):
    """生成網絡第 1 個模塊（4×4）：常數（或 dense）→ epilogue → conv → epilogue。"""

    def __init__(self, in_channels, dlatent_size, const_input_layer, gain, use_wscale, use_noise,
                 use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer, self.in_channels = const_input_layer, in_channels
        if const_input_layer:
            self.const = nn.Parameter(torch.ones(1, in_channels, 4, 4))
            self.bias = nn.Parameter(torch.ones(in_channels))
        else:
            self.dense = EqualizedLinear(dlatent_size, in_channels * 16, gain=gain / 4, use_wscale=use_wscale)
        epi = dict(dlatent_size=dlatent_size, use_wscale=use_wscale, use_noise=use_noise,
                   use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm,
                   use_styles=use_styles, activation_layer=activation_layer)
        self.epi1 = LayerEpilogue(in_channels, **epi)
        self.conv = EqualizedConv2d(in_channels, in_channels, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(in_channels, **epi)

    def forward(self, dlatents_in_range):
        b = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(b, -1, -1, -1) + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(b, self.in_channels, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        return self.epi2(x, dlatents_in_range[:, 1])


class GSynthesisBlock(nn.Module):
    """生成網絡一般模塊：上取樣×4 →（blur）→ conv → epilogue → conv → epilogue。"""

    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise,
                 use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.blur = BlurLayer(blur_filter) if blur_filter else None
        self.conv0_up = EqualizedConv2d(in_channels, out_channels, 3, gain=gain, use_wscale=use_wscale)
        epi = dict(dlatent_size=dlatent_size, use_wscale=use_wscale, use_noise=use_noise,
                   use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm,
                   use_styles=use_styles, activation_layer=activation_layer)
        self.epi1 = LayerEpilogue(out_channels, **epi)
        self.conv1 = EqualizedConv2d(out_channels, out_channels, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, **epi)

    def forward(self, x, dlatents_in_range):
        x = F.interpolate(x, scale_factor=4, mode="nearest")
        if self.blur is not None:
            x = self.blur(x)
        x = self.epi1(self.conv0_up(x), dlatents_in_range[:, 0])
        return self.epi2(self.conv1(x), dlatents_in_range[:, 1])


class DiscriminatorBlock(nn.Module):
    """判別器模塊：conv → 激活 →（blur）→ conv → 激活 → 平均池化 4。"""

    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel=None):
        super().__init__()
        self.conv0 = EqualizedConv2d(in_channels, in_channels, 3, gain=gain, use_wscale=use_wscale)
        self.blur = BlurLayer(blur_kernel) if blur_kernel else None
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, gain=gain, use_wscale=use_wscale)
        self.act = activation_layer

    def forward(self, x):
        x = self.act(self.conv0(x))
        if self.blur is not None:
            x = self.blur(x)
        return F.avg_pool2d(self.act(self.conv1(x)), 4)


class DiscriminatorTop(nn.Module):
    """判別器最後一塊（4×4）：minibatch stddev → conv → dense → dense → 分數 (B,1)。"""

    def __init__(self, mbstd_group_size, mbstd_num_features, in_channels, intermediate_channels, gain,
                 use_wscale, activation_layer, resolution=4, in_channels2=None, output_features=1, last_gain=1):
        super().__init__()
        self.stddev = StddevLayer(mbstd_group_size, mbstd_num_features) if mbstd_group_size > 1 else None
        if self.stddev is not None:
            in_channels += mbstd_num_features
        in_channels2 = in_channels2 or in_channels - (mbstd_num_features if self.stddev is not None else 0)
        self.conv = EqualizedConv2d(in_channels, intermediate_channels, 3, gain=gain, use_wscale=use_wscale)
        self.dense0 = EqualizedLinear(intermediate_channels * resolution ** 2, in_channels2, gain=gain,
                                      use_wscale=use_wscale)
        self.dense1 = EqualizedLinear(in_channels2, output_features, gain=last_gain, use_wscale=use_wscale)
        self.act = activation_layer

    def forward(self, x):
        if self.stddev is not None:
            x = self.stddev(x)
        x = self.act(self.conv(x))
        x = self.act(self.dense0(x.flatten(1)))
        return self.dense1(x)
