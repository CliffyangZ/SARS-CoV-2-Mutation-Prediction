"""古典漸進式判別器（reference_gan.py 的 Discriminator，FR-012）。

輸入與量子判別器相同：單通道 2ⁿ×2ⁿ 密度矩陣；forward(images_in, depth, alpha) → 分數 (B,1)。
depth 與生成器相同（0 為 4×4）。
"""
import numpy as np
import torch
import torch.nn as nn

from models.StyleGAN.blocks import DiscriminatorBlock, DiscriminatorTop
from models.StyleGAN.custom_layers import EqualizedConv2d


class Discriminator(nn.Module):
    def __init__(self, resolution, num_channels=1, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lReLU', use_wscale=True, mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, structure='linear', **kwargs):
        super(Discriminator, self).__init__()

        # 返回在第 stage 层中的特征图数量, stage<=4, 特征图数量为 512; stage>4, 每多一层特征图数量减半
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_features = mbstd_num_features    # mbstd 通道数
        self.mbstd_group_size = mbstd_group_size        # mbstd 组数
        self.structure = structure
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = int(resolution_log2 / 2)
        act, gain = {'ReLU': (torch.nn.ReLU(), np.sqrt(2)),
                     'lReLU': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # 构建判别器模块
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -2):
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 3), gain=gain, use_wscale=use_wscale,
                                             activation_layer=act, blur_kernel=blur_filter))
            # 转换 RGB 图像输入
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1, gain=gain,
                                            use_wscale=use_wscale))
        self.blocks = nn.ModuleList(blocks)

        # 构建最后一个判别器
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(1), intermediate_channels=nf(1), gain=gain,
                                            use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(1), kernel_size=1, gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # 生成临时下采样函数
        self.temporaryDownsampler = nn.AvgPool2d(4)

    def forward(self, images_in, depth, alpha=1., labels_in=None):
        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':                   # 不采用渐进式训练
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)
        elif self.structure == 'linear':                # 采用渐进式训练
            if depth > 0:
                residual = self.from_rgb[self.depth - depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth - 1](self.from_rgb[self.depth - depth - 1](images_in))
                x = (alpha * straight) + ((1 - alpha) * residual)
                for block in self.blocks[(self.depth - depth):]:
                    x = block(x)
            else:
                x = self.from_rgb[-1](images_in)
            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)
        return scores_out
