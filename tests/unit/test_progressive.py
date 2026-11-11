"""區塊排程與淡入。"""
import torch
from Research.models.progressive import downscale_classical, level_alpha
from Research.utils.config import make_config


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
