"""軟路由（Q3，FR-007／FR-008）。"""
import numpy as np
import pytest

from Research.QuSpike_PQK import features as F, router


class G:
    def __init__(self, name, ids):
        self.name, self.class_ids = name, tuple(ids)


GROUPS = [G("delta", [0, 1]), G("b1", [2, 3]), G("others", [4, 5])]


def _logits(rows):
    return np.log(np.asarray(rows, float))


def test_threshold_zero_equals_hard_routing():
    rng = np.random.default_rng(0)
    lg = rng.normal(size=(200, 6))
    w, names = router.route_weights(lg, GROUPS, 0.0)
    routed, _ = router.route(lg, GROUPS)
    assert [names[i] for i in w.argmax(1)] == list(routed)
    assert np.all(w.sum(1) == 1.0)


def test_close_top_two_selects_two_groups_weights_sum_le_one():
    # 群機率 ≈ delta .50, b1 .45, others .05 → 差 .05 < 0.1 → 觸發
    p = [[.25, .25, .225, .225, .025, .025]]
    w, names = router.route_weights(_logits(p), GROUPS, 0.1)
    assert (w > 0).sum() == 2
    assert w.sum() <= 1.0 + 1e-12
    assert w[0, names.index("others")] == 0 or w[0, names.index("others")] < 0.1


def test_confident_row_stays_hard():
    p = [[.9, .05, .01, .01, .015, .015]]
    w, names = router.route_weights(_logits(p), GROUPS, 0.3)
    assert (w > 0).sum() == 1 and w[0, 0] == 1.0


def test_at_most_three_groups_and_others_included_in_denominator():
    p = [[.17, .16, .17, .16, .17, .17]]  # 三群幾乎等機率
    w, names = router.route_weights(_logits(p), GROUPS, 0.5)
    assert (w > 0).sum() <= 3
    # others 計入分母 → 專家（delta、b1）權重和 < 1
    expert_sum = w[0, names.index("delta")] + w[0, names.index("b1")]
    assert expert_sum < 1.0


def test_coverage_hard_equals_router_accuracy():
    rng = np.random.default_rng(1)
    lg = rng.normal(size=(300, 6))
    y = rng.integers(0, 6, 300)
    w, names = router.route_weights(lg, GROUPS, 0.0)
    routed, _ = router.route(lg, GROUPS)
    assert router.coverage(w, names, y, GROUPS) == pytest.approx(router.router_accuracy(routed, y, GROUPS))


def test_soft_coverage_not_below_hard():
    rng = np.random.default_rng(2)
    lg = rng.normal(size=(300, 6))
    y = rng.integers(0, 6, 300)
    wh, names = router.route_weights(lg, GROUPS, 0.0)
    ws, _ = router.route_weights(lg, GROUPS, 0.3)
    assert router.coverage(ws, names, y, GROUPS) >= router.coverage(wh, names, y, GROUPS)
