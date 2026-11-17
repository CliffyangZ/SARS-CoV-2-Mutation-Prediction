"""run.py：消融展開、錯誤輸入、balanced 標示、未給 --final 時不碰 test。"""
import pytest

from Research.QuSpike_JEPA import run as R


def test_unknown_axis_rejected():
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["--ablate", "nope", "--values", "1"])


def test_bad_value_exits():
    a = R.build_parser().parse_args(["--ablate", "n_layers", "--values", "x"])
    with pytest.raises(SystemExit):
        R.plan(a)


def test_ablation_plan_changes_single_setting():
    a = R.build_parser().parse_args(["--head", "quantum", "--ablate", "n_layers", "--values", "1", "2"])
    jobs = R.plan(a)
    assert [R.make_cfg(b)["n_layers"] for b, _ in jobs] == [1, 2]
    assert all(R.make_cfg(b)["n_qubits"] == 6 for b, _ in jobs)
    assert {t for _, t in jobs} == {"__n_layers=1", "__n_layers=2"}


def test_mask_dist_and_level_mapping():
    a = R.build_parser().parse_args(["--mask-dist", "variability", "--mask-level", "mutation", "--mask-ratio", "0.3"])
    c = R.make_cfg(a)
    assert c["mask_prior"] == "P1" and c["mask_mut"] == 0.3 and c["mask_pos"] == 0.0


def test_final_not_given_never_embeds_test(monkeypatch, tmp_path):
    """未給 --final 時，test 不會被送進嵌入或評估。"""
    import numpy as np
    from Research.QuSpike_JEPA import dataset as D
    from tests.unit.test_quspike_jepa import _toy_tokenset

    train, test = _toy_tokenset(n=90), _toy_tokenset(n=30, seed=5)
    seen = []
    orig = R.PR.embed
    monkeypatch.setattr(R.PR, "embed", lambda m, ts, *a, **k: (seen.append(id(ts)), orig(m, ts, *a, **k))[1])
    a = R.build_parser().parse_args(["--epochs", "1", "--seeds", "0", "--out", str(tmp_path), "--device", "cpu"])
    R.run_one(a, R.make_cfg(a), R.config_name(a), 0, (train, test, "t"))
    assert id(test) not in seen
