"""小資料跑通古典與量子各一個；續跑與重現。"""
import json

import torch

from Research.QuSpike_JEPA import run as R
from tests.unit.test_quspike_jepa import _toy_tokenset


def _args(tmp, *extra):
    return R.build_parser().parse_args(["--epochs", "2", "--seeds", "0", "--out", str(tmp), "--device", "cpu", *extra])


def _run(tmp, *extra):
    a = _args(tmp, *extra)
    data = (_toy_tokenset(n=150), _toy_tokenset(n=60, seed=3), "toy")
    return R.run_one(a, R.make_cfg(a), R.config_name(a), 0, data)


def test_classical_and_quantum_complete(tmp_path):
    for head in ("classical", "quantum"):
        r = _run(tmp_path / head, "--head", head, "--final")
        assert r["status"] == "complete" and "metrics_val" in r and "metrics_test" in r and r["effective_rank"] > 0


def test_same_seed_same_metrics(tmp_path):
    a = _run(tmp_path / "a", "--head", "classical")
    b = _run(tmp_path / "b", "--head", "classical")
    assert a["metrics_val"]["f1_macro"] == b["metrics_val"]["f1_macro"]


def test_completed_run_is_skipped(tmp_path):
    _run(tmp_path, "--head", "classical")
    assert _run(tmp_path, "--head", "classical") is None


def test_resume_equals_uninterrupted(tmp_path):
    full = _run(tmp_path / "full", "--head", "classical")
    # 先跑 1 epoch 留下檢查點，再以 2 epoch 續跑
    a1 = R.build_parser().parse_args(["--epochs", "1", "--seeds", "0", "--out", str(tmp_path / "r"), "--device", "cpu"])
    data = (_toy_tokenset(n=150), _toy_tokenset(n=60, seed=3), "toy")
    R.run_one(a1, R.make_cfg(a1), R.config_name(a1), 0, data)
    (tmp_path / "r" / "jepa_VA_classical__mut__seed0.json").unlink()
    res = _run(tmp_path / "r", "--head", "classical")
    assert res["loss_history"] == full["loss_history"]


def test_finetune_quantum_and_scratch(tmp_path):
    r = _run(tmp_path / "ft", "--head", "quantum", "--finetune", "--ft-epochs", "2", "--final")
    assert r["status"] == "complete" and r["config"]["name"].endswith("_ft")
    assert "metrics_val_probe" in r and "metrics_test_probe" in r and len(r["ft_loss_history"]) == 2
    a = _args(tmp_path / "sc", "--head", "classical", "--finetune", "--ft-epochs", "1", "--epochs", "0")
    assert R.config_name(a).endswith("_ft_scratch")


def test_mutset_linear_baseline(tmp_path):
    r = _run(tmp_path / "ml", "--head", "mutset_linear", "--final")
    assert r["status"] == "complete" and 0.0 <= r["metrics_test"]["f1_macro"] <= 1.0
