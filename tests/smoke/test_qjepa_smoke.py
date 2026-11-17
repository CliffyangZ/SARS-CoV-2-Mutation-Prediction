"""一個 batch 的前向與反向：古典與量子頭。"""
import copy

import numpy as np
import pytest
import torch

from Research.QuSpike_JEPA import loss as L
from Research.QuSpike_JEPA import pretrain as PT
from Research.QuSpike_JEPA import views as V
from tests.unit.test_quspike_jepa import _toy_tokenset


@pytest.mark.parametrize("head", ["classical", "quantum", "quantum_1layer"])
def test_forward_backward(head):
    cfg = copy.deepcopy(PT.DEFAULTS)
    cfg["head"] = head
    ts = _toy_tokenset(n=32)
    vm = V.ViewMaker(ts, variant="VA", seed=0)
    model = PT.Model(cfg, 0)
    pa, sa, va, pb, sb, vb = [torch.as_tensor(x) for x in vm.batch(np.arange(16))]
    qa, za = model.embed(pa, sa, va)
    qb, _ = model.embed(pb, sb, vb)
    loss, _ = L.vicreg(qa, qb)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
