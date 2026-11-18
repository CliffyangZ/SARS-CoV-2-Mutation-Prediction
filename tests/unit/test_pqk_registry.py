"""v2 登錄檔與防護（US3，FR-011／FR-014）。"""
import json
import os

import numpy as np
import pytest

from Research.QuSpike_PQK import features as F, registry

DRAFT = dict(version="v9.0", hypothesis="h", design={"n_qubits": {"delta": 4}},
             controls=["V2_classical", "V2_nocouple", "BEST_CLASSICAL_US2"],
             criteria=[{"claim": "x", "a": "V2", "b": "V2_classical", "rule": "delta >= max(sigma, 0.02)"}],
             stop_rule="停止", seeds=[0, 1, 2, 3, 4], selection_log=[])


def test_lock_makes_file_readonly_and_cannot_relock(tmp_path):
    path = registry.lock(dict(DRAFT), root=tmp_path)
    assert not os.access(path, os.W_OK) or os.geteuid() == 0
    with pytest.raises(registry.RegistryError):
        registry.lock(dict(DRAFT), root=tmp_path)


def test_hash_mismatch_detected(tmp_path):
    path = registry.lock(dict(DRAFT), root=tmp_path)
    os.chmod(path, 0o644)
    rec = json.loads(path.read_text())
    rec["hypothesis"] = "改過的假設"
    path.write_text(json.dumps(rec))
    with pytest.raises(registry.RegistryError, match="雜湊"):
        registry.load("v9.0", root=tmp_path)


def test_missing_registry_refused(tmp_path):
    with pytest.raises(registry.RegistryError):
        registry.assert_locked("v404", root=tmp_path)


def test_incomplete_draft_and_too_few_seeds_refused(tmp_path):
    with pytest.raises(registry.RegistryError):
        registry.lock({k: v for k, v in DRAFT.items() if k != "stop_rule"}, root=tmp_path)
    with pytest.raises(registry.RegistryError):
        registry.lock(dict(DRAFT, seeds=[0, 1, 2]), root=tmp_path)


def test_new_version_keeps_old(tmp_path):
    registry.lock(dict(DRAFT), root=tmp_path)
    registry.lock(dict(DRAFT, version="v9.1", supersedes="v9.0"), root=tmp_path)
    assert registry.all_versions(tmp_path) == ["v9.0", "v9.1"]
    assert registry.load("v9.0", root=tmp_path)["supersedes"] is None


def test_eval_count_and_timing(tmp_path):
    registry.lock(dict(DRAFT), root=tmp_path)
    assert registry.timing_ok("v9.0", tmp_path) is None
    assert registry.record_test_eval("v9.0", "a", tmp_path) == 1
    assert registry.record_test_eval("v9.0", "b", tmp_path) == 2
    assert registry.timing_ok("v9.0", tmp_path) is True


def test_validation_view_has_no_test_arrays():
    fs = F.load_features(source="mutation")
    from Research.QuSpike_PQK.run import subsample

    view = registry.ValidationOnlyView(subsample(fs, 800))
    with pytest.raises(AttributeError):
        view.x_test
    with pytest.raises(AttributeError):
        view.y_test
    sub = view.as_featureset()
    assert len(sub.y_test) == len(view.y_val) and len(sub.y_train) == len(view.y_fit)
