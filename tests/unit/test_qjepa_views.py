"""view：最少突變數、遮蔽符號、重現性、遮蔽分佈預算可比、不輸出未回報的空 view。"""
import numpy as np

from Research.QuSpike_JEPA import dataset as D
from Research.QuSpike_JEPA import views as V
from tests.unit.test_quspike_jepa import _toy_tokenset


def test_min_mutations_kept_when_available():
    ts = _toy_tokenset()
    vm = V.ViewMaker(ts, variant="VC", mask_mut=0.0, mask_pos=0.0, min_mut=3, trim=(0.5, 1.0), local_trim=(0.3, 0.5), seed=0)
    for i in range(30):
        assert len(vm.one_view(i, local=True)) >= 3 or vm.stats["retry_fail"] > 0


def test_batches_reproducible_per_seed_and_epoch():
    ts = _toy_tokenset()
    a = V.ViewMaker(ts, variant="VA", seed=7).batch(np.arange(10), epoch=2)
    b = V.ViewMaker(ts, variant="VA", seed=7).batch(np.arange(10), epoch=2)
    c = V.ViewMaker(ts, variant="VA", seed=7).batch(np.arange(10), epoch=3)
    assert all((x == y).all() for x, y in zip(a, b))
    assert any(x.shape != y.shape or (x != y).any() for x, y in zip(a, c))


def test_p0_and_p1_have_comparable_mask_budget():
    ts = _toy_tokenset(n=100)
    ids = ts.pos[ts.valid].astype(np.int64) * D.N_SYM + ts.sym[ts.valid]
    p0 = V.ViewMaker(ts, mask_prior="P0", seed=0)._mask_rate(ids).mean()
    p1 = V.ViewMaker(ts, mask_prior="P1", seed=0)._mask_rate(ids).mean()
    assert abs(p0 - p1) / p0 < 0.35


def test_masked_tokens_are_distinguishable_from_reference_positions():
    """被遮蔽處為 MASK_SYM；未列出的位置即「確定等於參考」。"""
    ts = _toy_tokenset()
    vm = V.ViewMaker(ts, variant="VC", mask_mut=1.0, mask_pos=0.0, trim=(1.0, 1.0), seed=1)
    v = vm.one_view(0, local=False)
    assert v and all(s == D.MASK_SYM for _, s in v)


def test_zero_mutation_sequence_counted_as_empty():
    ts = _toy_tokenset(n=6)
    ts.valid[0] = False
    vm = V.ViewMaker(ts, variant="VC", seed=0)
    assert vm.one_view(0) == [] and vm.stats["empty_views"] == 1
