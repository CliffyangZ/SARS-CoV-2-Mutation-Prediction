"""StyleGAN 模型套件（結構依 reference_gan.py）。"""
import torch


@torch.no_grad()
def update_average(model_tgt, model_src, beta):
    """EMA：tgt ← beta·tgt + (1-beta)·src（參數）；buffer（如 Truncation.avg_latent）直接複製自 src。"""
    src_params = dict(model_src.named_parameters())
    for name, p in model_tgt.named_parameters():
        p.copy_(beta * p + (1.0 - beta) * src_params[name])
    src_bufs = dict(model_src.named_buffers())
    for name, b in model_tgt.named_buffers():
        b.copy_(src_bufs[name])
