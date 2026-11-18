"""路由與結構下限的單元測試（T014）。

test_others_group_predictions_identical_to_reference 對應 SC-004：
易分群不經量子路徑，其預測必須與 C0 逐筆完全相同。這不是風格問題——
若易分群也被動到，報告的分群分解就無法歸因。
"""
import numpy as np
import pytest

from Research.QuSpike_PQK import features as F, pipeline, router


@pytest.fixture(scope="module")
def small():
    """小樣本 FeatureSet + 分群（真實資料，取子集以加速）。"""
    fs = F.load_features(source="mutation")
    from Research.QuSpike_PQK.run import subsample

    fs = subsample(fs, 2500)
    return fs, F.make_groups(fs)


def test_group_of_class_covers_all(small):
    """每個類別都要有對應的群。"""
    fs, groups = small
    lut = router.group_of_class(groups)
    assert set(lut) == set(range(len(fs.labels)))


def test_route_assigns_every_sample(small):
    """每個樣本都被指派到一個群。"""
    fs, groups = small
    model, dev = router.fit_reference(fs.x_train, fs.y_train, len(fs.labels), seed=0)
    lg = router.logits(model, fs.x_test, dev)
    routed, pred = router.route(lg, groups)
    assert len(routed) == len(fs.y_test)
    assert set(routed) <= {g.name for g in groups}
    assert len(pred) == len(fs.y_test)


def test_router_accuracy_in_range_and_meaningful(small):
    """路由準確率落在 [0,1]，且明顯高於隨機（它是量子專家的改善上限）。"""
    fs, groups = small
    model, dev = router.fit_reference(fs.x_train, fs.y_train, len(fs.labels), seed=0)
    routed, _ = router.route(router.logits(model, fs.x_test, dev), groups)
    acc = router.router_accuracy(routed, fs.y_test, groups)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.5, f"路由準確率僅 {acc:.3f}，量子專家看到的樣本多半是錯的"


def test_reference_beats_majority(small):
    """參考線必須明顯優於全猜多數類，否則 C0 這條下限本身無意義。"""
    fs, groups = small
    model, dev = router.fit_reference(fs.x_train, fs.y_train, len(fs.labels), seed=0)
    pred = router.logits(model, fs.x_test, dev).argmax(1)
    acc = float((pred == fs.y_test).mean())
    maj = float(np.bincount(fs.y_test, minlength=len(fs.labels)).max() / len(fs.y_test))
    assert acc > maj + 0.1


def test_others_group_predictions_identical_to_reference(small):
    """SC-004（008 改述）：**被路由到** others 的列，預測與 C0 逐筆完全相同。

    007 的版本以「真實標籤屬於 others」選列——那只在推論偷看標籤時才成立
    （被誤路由到 delta 的 others 序列永遠收不到專家）。無標籤推論下，
    保證只剩「路由到 others 的列沒有專家」。
    """
    fs, groups = small
    cfg = dict(n_qubits={"delta": 4, "b1": 2, "all": 4}, trotter_steps=4, evolution_time=0.8,
               gamma=None, projection="lda", shots=None, noise_p=0.0, device="default.qubit")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        main_pred = pipeline.evaluate_control("MAIN", fs, groups, cfg, 0, cache, tune=False)[0]
        c0_pred = pipeline.evaluate_control("C0", fs, groups, cfg, 0, cache, tune=False)[0]
        model, dev = router.fit_reference(fs.x_train, fs.y_train, len(fs.labels), seed=0)
        routed, _ = router.route(router.logits(model, fs.x_test, dev), groups)
    mask = routed == "others"
    assert mask.sum() > 0
    diff = int((main_pred[mask] != c0_pred[mask]).sum())
    assert diff == 0, f"路由到 others 的列有 {diff} 筆預測被量子路徑改動"


def test_c0_is_exactly_the_reference_line(small):
    """C0 的預測必須等於參考線本身的 argmax（它是結構下限）。"""
    fs, groups = small
    cfg = dict(n_qubits={"delta": 4, "b1": 2, "all": 4}, trotter_steps=4, evolution_time=0.8,
               gamma=None, projection="lda", shots=None, noise_p=0.0, device="default.qubit")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        c0_pred = pipeline.evaluate_control("C0", fs, groups, cfg, 0, Path(td), tune=False)[0]
    model, dev = router.fit_reference(fs.x_train, fs.y_train, len(fs.labels), seed=0)
    assert np.array_equal(c0_pred, router.logits(model, fs.x_test, dev).argmax(1))


def test_zero_scale_recovers_reference(small):
    """混合權重為 0 時整個模型還原成參考線（FR-009 的結構下限）。"""
    fs, groups = small
    cfg = dict(n_qubits={"delta": 4, "b1": 2, "all": 4}, trotter_steps=4, evolution_time=0.8,
               gamma=None, projection="lda", shots=None, noise_p=0.0, device="default.qubit")
    import tempfile
    from pathlib import Path

    from Research.QuSpike_PQK import experts

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        c0_pred = pipeline.evaluate_control("C0", fs, groups, cfg, 0, cache, tune=False)[0]
        model, dev = router.fit_reference(fs.x_train, fs.y_train, len(fs.labels), seed=0)
        ref_ev = router.logits(model, fs.x_test, dev)
        g = [x for x in groups if x.is_quantum][0]
        pr = F.fit_projection(fs, g, 2, method="lda")
        phi_tr, phi_ev, _ = pipeline.feature_map("pqk", pr.z_train, pr.z_test, g.name, 2, cfg,
                                                 cache, True, 0, fs.data_hash, "lda")
        block, _ = experts.fit_group_expert("kernel", phi_tr, phi_ev, fs.y_train[g.train_idx],
                                            g.class_ids, len(fs.labels), len(g.test_idx),
                                            seed=0, scale=0.0)
    merged = experts.combine(ref_ev, [(g.test_idx, block)])
    assert np.array_equal(merged.argmax(1), c0_pred)
