"""生成器（reference_gan.py 的 GMapping / GSynthesis / Generator，古典 / 量子判別器共用，FR-008）。

resolution = 最終邊長（64 或 1024）；depth = log2(resolution)/2 個漸進階段，邊長 4、16、64、…（每階段 ×4）。
Generator 輸出 min-max 正規化到 [0,1] 的單通道特徵圖 (B,1,S,S)；
由 Research.StyleGAN.FE.encoding.featuremap_to_rho 轉為合法密度矩陣 ρ（R6），不屬於本模型。
"""
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from models.StyleGAN.blocks import GSynthesisBlock, InputBlock
from models.StyleGAN.custom_layers import EqualizedConv2d, EqualizedLinear, PixelNormLayer, Truncation


class GMapping(nn.Module):
    """映射網絡的搭建。"""

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lReLU',
                 use_wscale=True, normalize_latents=True, **kwargs):
        super().__init__()
        self.latent_size = latent_size                  # 潜码 Z 的维度
        self.mapping_fmaps = mapping_fmaps              # 映射层中的特征图数量
        self.dlatent_size = dlatent_size                # 中间潜向量 W 的维度
        self.dlatent_broadcast = dlatent_broadcast      # 中间潜向量 W 是否广播
        act, gain = {'ReLU': (torch.nn.ReLU(), np.sqrt(2)),
                     'lReLU': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        layers = []
        if normalize_latents:                           # 潜码归一化
            layers.append(('pixel_norm', PixelNormLayer()))
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps, gain=gain,
                                                 lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(('dense{:d}'.format(layer_idx), EqualizedLinear(
                fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # 最开始的输入: 潜码 Z [mini_batch, latent_size]
        x = self.map(x)
        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):
    """生成网络的搭建。"""

    def __init__(self, dlatent_size=512, num_channels=1, resolution=1024, fmap_base=8192,
                 fmap_decay=1.0, fmap_max=512, use_styles=True, const_input_layer=True,
                 use_noise=True, nonlinearity='lReLU', use_wscale=True, use_pixel_norm=False,
                 use_instance_norm=True, blur_filter=None, structure='linear', **kwargs):
        super().__init__()

        def nf(stage):                                  # 定义每个阶段的特征图数目
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure                      # 选择是否渐进式训练
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = int(resolution_log2 / 2)           # 深度
        self.num_layers = 2 * self.depth                # 层数
        self.num_styles = self.num_layers if use_styles else 1
        act, gain = {'ReLU': (torch.nn.ReLU(), np.sqrt(2)),
                     'lReLU': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # 生成网络的第 1 个模块
        self.init_block = InputBlock(nf(1), dlatent_size, const_input_layer, gain, use_wscale,
                                     use_noise, use_pixel_norm, use_instance_norm, use_styles, act)
        # 将输出转换为 RGB 图像
        rgb_converters = [EqualizedConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # 生成网络的其他模块
        blocks = []
        for res in range(4, resolution_log2 + 1, 2):
            last_channels = nf(res - 3)
            channels = nf(res - 1)
            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain,
                                          use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                          use_styles, act))
            rgb_converters.append(EqualizedConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)

        # 生成临时上采样函数
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=4)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':                   # 不采用渐进式训练
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':                # 渐进式训练
            x = self.init_block(dlatents_in[:, 0:2])
            if depth > 0:
                for i, block in enumerate(self.blocks[:depth - 1]):
                    x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](self.blocks[depth - 1](
                    x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))
                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)
        return images_out


class Generator(nn.Module):
    """生成器的构建：forward(latents_in, depth, alpha) → 特征图 (B,1,S,S)，值域 [0,1]。"""

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, output_norm="batch", **kwargs):
        super(Generator, self).__init__()
        assert output_norm in ("batch", "sample", "sigmoid"), output_norm
        self.output_norm = output_norm                  # batch=舊（整個 batch 一起 min-max）；sample=逐樣本；sigmoid=機率圖
        self.style_mixing_prob = style_mixing_prob      # 训练时采用风格混合的概率
        self.num_layers = (int(np.log2(resolution)) // 2) * 2

        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, dlatent_size=dlatent_size, **kwargs)

        # 截断技巧乘数
        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size), max_layer=truncation_cutoff,
                                         threshold=truncation_psi, beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in, depth, alpha, labels_in=None):
        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # 使用风格混合技巧
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
                                             ).to(latents_in.device)
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1, cur_layers) \
                    if random.random() < self.style_mixing_prob else cur_layers
                dlatents_in = torch.where(layer_idx < mixing_cutoff, dlatents_in, dlatents2)

        # 应用截断技巧
        if self.truncation is not None:
            dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)
        if self.output_norm == "sigmoid":               # 輸出視為各格機率，並對稱化（ρ 為實對稱）
            p = torch.sigmoid(fake_images)
            return (p + p.transpose(-1, -2)) / 2
        if self.output_norm == "sample":                # 逐樣本 min-max：樣本之間互不影響（FR-008）
            flat = fake_images.flatten(1)
            fi_min, fi_max = flat.min(1).values.view(-1, 1, 1, 1), flat.max(1).values.view(-1, 1, 1, 1)
            return (fake_images - fi_min) / (fi_max - fi_min).clamp_min(1e-8)
        fi_min = fake_images.min()
        fi_max = fake_images.max()
        fake_images = (fake_images - fi_min) / (fi_max - fi_min)  # [0,1]（舊行為：整個 batch 一起）
        return fake_images
