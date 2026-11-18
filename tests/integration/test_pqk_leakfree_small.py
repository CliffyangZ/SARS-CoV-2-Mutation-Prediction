"""無洩漏整合測試（US1，FR-001／FR-002／SC-001）。

小樣本跑完七組設定；對每個設定做「打亂評估標籤 → 預測逐筆相同」。
另斷言：路由到 others 的列與 C0 逐筆相同（SC-004 的 008 改述）。
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from Research.QuSpike_PQK import features as F, pipeline, run
from Research.QuSpike_PQK.run import subsample

CFG = dict(n_qubits={"delta": 4, "b1": 2, "all": 4}, trotter_steps=4, evolution_time=0.8,
           gamma=None, projection="lda", shots=None, noise_p=0.0, device="default.qubit")


@pytest.fixture(scope="module")
def fs():
    return subsample(F.load_features(source="mutation"), 1200)


@pytest.mark.parametrize("control", run.CONTROLS)
def test_shuffled_labels_do_not_change_predictions(fs, control):
    """七組設定各自：打亂評估標籤後預測逐筆不變。"""
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        base = pipeline.evaluate_control(control, fs, F.make_groups(fs), CFG, 0, cache, tune=False)[0]
        sh = run._shuffled(fs, 7)
        got = pipeline.evaluate_control(control, sh, F.make_groups(sh), CFG, 0, cache, tune=False)[0]
    assert np.array_equal(base, got)


def test_leakcheck_cli_exit_code_zero():
    """leakcheck 子命令對小資料回傳 0。"""
    with tempfile.TemporaryDirectory() as td:
        code = run.main(["leakcheck", "--limit", "800", "--controls", "C0,C1", "--seeds", "0",
                         "--out", td, "--no-tune"])
    assert code == 0


def test_soft_routing_and_pls_run_end_to_end(fs):
    """軟路由 + PLS 補維端到端可跑，且回報 coverage 與解耦位元。"""
    cfg = dict(CFG, routing="soft", soft_threshold=0.2, supplement="pls")
    with tempfile.TemporaryDirectory() as td:
        pred, proba, info = pipeline.evaluate_control("MAIN", fs, F.make_groups(fs), cfg, 0,
                                                      Path(td), tune=False)
    assert pred.shape == fs.y_test.shape
    assert info["route_coverage"]["soft"] >= info["route_coverage"]["hard"] - 1e-12
    assert all(v == [] for v in info["decoupled_qubits"].values())
