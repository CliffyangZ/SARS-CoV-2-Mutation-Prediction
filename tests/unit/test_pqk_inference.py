"""無標籤推論（Q1，FR-001／FR-002）。

v1 的洩漏：專家只套用在「被路由到該群**且真實標籤屬於該群**」的列。
這些測試保證 008 的推論路徑不含任何標籤輸入，且被誤路由的列確實收到專家修正。
"""
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest

from Research.QuSpike_PQK import experts, features as F, inference, pipeline, router
from Research.QuSpike_PQK.run import subsample

CFG = dict(n_qubits={"delta": 4, "b1": 2, "all": 4}, trotter_steps=4, evolution_time=0.8,
           gamma=None, projection="lda", shots=None, noise_p=0.0, device="default.qubit")


@pytest.fixture(scope="module")
def small():
    """小樣本 FeatureSet + 分群（真實資料，取子集以加速）。"""
    fs = subsample(F.load_features(source="mutation"), 2500)
    return fs, F.make_groups(fs)


def _label_params(fn):
    return [p for p in inspect.signature(fn).parameters if p.startswith("y")]


def test_inference_functions_take_no_labels():
    """predict／route／route_weights／combine／expert_blocks／assemble 的簽名都沒有 y 開頭的參數。"""
    for fn in (inference.predict, router.route, router.route_weights, experts.combine,
               inference.expert_blocks, inference.assemble, router.group_proba):
        assert _label_params(fn) == [], f"{fn.__qualname__} 含標籤參數"
    assert _label_params(inference.predict_legacy_leaky) == ["y_eval"], "洩漏版必須是唯一讀標籤者"


def test_predictions_unchanged_when_eval_labels_shuffled(small):
    """打亂評估標籤後整條管線的預測逐筆相同（SC-001，以古典 C1 快速驗證）。"""
    fs, _ = small
    preds = []
    for k in range(3):
        y = fs.y_test.copy()
        if k:
            np.random.default_rng(k).shuffle(y)
        fsk = F.FeatureSet(**{**fs.__dict__, "y_test": y})
        with tempfile.TemporaryDirectory() as td:
            preds.append(pipeline.evaluate_control("C1", fsk, F.make_groups(fsk), CFG, 0,
                                                   Path(td), tune=False)[0])
    assert np.array_equal(preds[0], preds[1]) and np.array_equal(preds[0], preds[2])


def test_misrouted_rows_receive_expert(small):
    """被路由到 delta 但真實標籤不在 delta 的列，仍收到 delta 專家（US1 情境 2）。"""
    fs, groups = small
    with tempfile.TemporaryDirectory() as td:
        st, _ = pipeline.fit_state("classical_rbf", None, True, fs, groups, CFG, 0, Path(td),
                                   fs.data_hash)
    ref = router.logits(st.ref_model, fs.x_test, st.device)
    weights, _ = inference._weights(st, ref, "hard", 0.0)
    delta = [g for g in groups if g.name == "delta"][0]
    rows, w = weights["delta"]
    wrong = rows[~np.isin(fs.y_test[rows], list(delta.class_ids))]
    assert len(wrong) > 0, "測試前提：小樣本中應有被誤路由到 delta 的列"
    assert np.all(w == 1.0)
    # 這些列的最終 logits 與參考線不同（專家確實作用）
    blocks = inference.expert_blocks(st, fs.x_test, "ev", {"delta": rows, "b1": np.array([], int)})
    assert np.abs(blocks["delta"][1][np.isin(rows, wrong)]).sum() > 0


def test_scale_zero_equals_reference(small):
    """scale=0 時 predict 等於參考線（結構下限）。"""
    fs, groups = small
    with tempfile.TemporaryDirectory() as td:
        st, _ = pipeline.fit_state("classical_rbf", None, True, fs, groups, CFG, 0, Path(td),
                                   fs.data_hash)
    pred, _, _ = inference.predict(st, fs.x_test, scale=0.0)
    ref = router.logits(st.ref_model, fs.x_test, st.device).argmax(1)
    assert np.array_equal(pred, ref)


def test_legacy_leaky_differs_only_on_misrouted(small):
    """洩漏版只在「被路由到該群但真實不屬該群」的列上與無標籤版不同。"""
    fs, groups = small
    with tempfile.TemporaryDirectory() as td:
        st, _ = pipeline.fit_state("classical_rbf", None, True, fs, groups, CFG, 0, Path(td),
                                   fs.data_hash)
    clean, _, _ = inference.predict(st, fs.x_test, scale=1.0)
    leaky, _, _ = inference.predict_legacy_leaky(st, fs.x_test, fs.y_test, scale=1.0)
    ref = router.logits(st.ref_model, fs.x_test, st.device)
    routed, _ = router.route(ref, groups)
    lut = router.group_of_class(groups)
    truth = np.array([lut[int(c)] for c in fs.y_test])
    correctly_routed = routed == truth
    assert np.array_equal(clean[correctly_routed], leaky[correctly_routed])
