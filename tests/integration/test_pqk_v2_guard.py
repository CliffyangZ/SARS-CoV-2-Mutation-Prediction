"""未登錄不得評估測試集（US3，SC-007）。"""
import pytest

from Research.QuSpike_PQK import registry
from tests.unit.test_pqk_registry import DRAFT


def test_unlocked_never_loads_test_data(tmp_path):
    """未鎖定：loader（會載入測試陣列）完全不被呼叫，也不寫入評估紀錄。"""
    called = []

    def loader():
        called.append(1)
        return "TEST-DATA"

    with pytest.raises(registry.RegistryError):
        registry.evaluate_on_test("v9.0", loader, lambda fs, rec: None, root=tmp_path)
    assert called == []
    assert not (tmp_path / registry.EVALS).exists()


def test_locked_first_eval_after_lock(tmp_path):
    registry.lock(dict(DRAFT), root=tmp_path)
    idx, out = registry.evaluate_on_test("v9.0", lambda: "T", lambda fs, rec: (fs, rec["version"]),
                                         run_id="r1", root=tmp_path)
    assert idx == 1 and out == ("T", "v9.0")
    assert registry.timing_ok("v9.0", tmp_path) is True
    idx2, _ = registry.evaluate_on_test("v9.0", lambda: "T", lambda fs, rec: None, root=tmp_path)
    assert idx2 == 2  # 第二次評估被計數；正式報告只採第 1 次
