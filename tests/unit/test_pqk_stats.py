"""多種子彙整、雜訊下限重估、對照組一致性（FR-003／FR-004／FR-010）。"""
import numpy as np
import pytest

from Research.QuSpike_PQK import criteria, stats


def test_noise_floor_is_delta_when_seed_variance_small():
    sc = {"C0": [0.75, 0.7501, 0.7499, 0.75, 0.7502], "MAIN": [0.76, 0.7601, 0.7599, 0.76, 0.7602]}
    f = stats.estimate_noise_floor(sc)
    assert f["floor"] == 0.02 and f["floor_source"] == "delta"


def test_noise_floor_dominated_by_seed_variance():
    sc = {"C0": [0.70, 0.78, 0.74, 0.80, 0.72], "MAIN": [0.71, 0.79, 0.75, 0.81, 0.70]}
    f = stats.estimate_noise_floor(sc)
    assert f["floor"] > 0.02 and f["floor_source"] == "seed_variance"
    assert f["floor"] == pytest.approx(f["sigma_pooled"])


def test_rejects_mixed_seed_lists():
    a = {"config": {"seeds": [0, 1, 2, 3, 4]}}
    b = {"config": {"seeds": [0, 1, 2]}}
    with pytest.raises(ValueError):
        stats.assert_same_seeds([a, b])
    stats.assert_same_seeds([a, a])


def test_assert_same_improvements():
    ok = {"MAIN": dict(routing="soft", supplement="pls"), "C1": dict(routing="soft", supplement="pls")}
    stats.assert_same_improvements(ok)
    bad = {"MAIN": dict(routing="soft"), "C1": dict(routing="hard")}
    with pytest.raises(ValueError):
        stats.assert_same_improvements(bad)


def test_criteria_records_floor_source():
    sc = {"MAIN": [0.8, 0.81], "C0": [0.7, 0.71], "C1": [0.7, 0.7], "C2": [0.7, 0.7],
          "C3": [0.7, 0.7], "C4": [0.7, 0.7], "C5": [0.7, 0.7]}
    recs = criteria.evaluate(sc)
    assert all("floor_source" in r for r in recs)
    assert recs[0]["verdict"] == "holds"
