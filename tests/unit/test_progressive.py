"""區塊排程與淡入。"""
import torch
from models.StyleGAN.progressive import downscale_classical, level_alpha
from Research.StyleGAN.utils.config import make_config


def test_schedule_even_and_alpha_monotone():
    sch = make_config(2, steps=3500).block_schedule
    assert [b - a for a, b in sch] == [700] * 5
    for lv in range(1, 5):
        a, b = sch[lv]
        al = [level_alpha(sch, s)[1] for s in range(a, b)]
        assert level_alpha(sch, a) == (lv, 0.0) and al[-1] == 1.0
        assert all(x <= y for x, y in zip(al, al[1:]))
    assert level_alpha(sch, 0) == (0, 1.0)


def test_downscale_is_avg_pool():
    x = torch.arange(16.).reshape(1, 1, 4, 4)
    assert downscale_classical(x, 4).item() == x.mean().item()


def test_fade_frac_default_equals_half_and_boundaries():
    """003：fade_frac 預設 0.5 與原行為逐步相同；0.2 時更早到 1；1.0 時整個區塊才升到 1。"""
    sch = make_config(1, steps=3500).block_schedule
    a, b = sch[2]
    for st in range(a, b, 50):
        assert level_alpha(sch, st) == level_alpha(sch, st, 0.5)
    mid = a + (b - a) // 2
    assert level_alpha(sch, mid, 0.2)[1] == 1.0 and level_alpha(sch, mid, 1.0)[1] < 1.0
    assert level_alpha(sch, b - 1, 1.0)[1] > 0.99 and level_alpha(sch, a, 0.2) == (2, 0.0)
