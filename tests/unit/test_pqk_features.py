"""特徵、分群與降維的單元測試（T005／T007）。

test_lda_fitted_on_train_only 對應 FR-003：降維只能用 train 擬合，
否則 test 標籤會洩漏到特徵裡，整個評估失效。
"""
import numpy as np
import pytest

from Research.QuSpike_PQK import features as F


@pytest.fixture(scope="module")
def fs():
    """真實資料的 FeatureSet（22 類）。"""
    return F.load_features(source="mutation")


# --- FeatureSet（T007）---

def test_feature_set_shape_and_labels(fs):
    """22 類、突變指標維度為詞彙大小、train/test 維度一致。"""
    assert len(fs.labels) == 22
    assert fs.x_train.shape[1] == fs.x_test.shape[1]
    assert fs.x_train.shape[0] == len(fs.y_train) == 22175
    assert fs.x_test.shape[0] == len(fs.y_test) == 9514
    assert fs.source == "mutation_indicator"


def test_mutation_indicator_is_binary(fs):
    """突變指標的值必須只有 0 與 1。"""
    vals = np.unique(fs.x_train[:2000])
    assert set(vals.tolist()) <= {0.0, 1.0}


def test_labels_in_range(fs):
    """標籤值域為 [0, 22)。"""
    assert fs.y_train.min() >= 0 and fs.y_train.max() < 22
    assert fs.y_test.min() >= 0 and fs.y_test.max() < 22


def test_provenance_recorded(fs):
    """資料雜湊、切分版本與選定理由都必須記錄（FR-001／FR-002）。"""
    assert fs.data_hash and len(fs.data_hash) >= 8
    assert "+" in fs.split_version
    assert fs.selection_reason


# --- Group（T007）---

def test_groups_partition_all_classes(fs):
    """三群互斥且聯集為全部 22 類。"""
    groups = F.make_groups(fs)
    seen = []
    for g in groups:
        seen += list(g.class_ids)
    assert sorted(seen) == list(range(22))
    assert len(seen) == len(set(seen)), "群之間不得重疊"


def test_group_sizes_match_design(fs):
    """群規模須符合 data-model.md 記錄的實測值。"""
    by = {g.name: g for g in F.make_groups(fs)}
    assert (by["delta"].n_train, by["delta"].n_test) == (9538, 4091)
    assert (by["b1"].n_train, by["b1"].n_test) == (4200, 1801)
    assert (by["others"].n_train, by["others"].n_test) == (8437, 3622)
    assert by["delta"].is_quantum and by["b1"].is_quantum
    assert not by["others"].is_quantum, "others 不得進量子路徑"


def test_small_group_degrades_to_reference():
    """類別數 < 2 或樣本數過少時，該群降級為不走量子（edge case）。"""
    g = F.Group(name="tiny", class_ids=(0,), n_train=3, n_test=1, is_quantum=True)
    assert F.degrade_if_too_small(g, min_train=10).is_quantum is False


# --- Projection（T005）---

def test_lda_fitted_on_train_only(fs):
    """FR-003：把 test 標籤打亂後 z_test 必須逐元素不變。"""
    groups = {g.name: g for g in F.make_groups(fs)}
    g = groups["b1"]
    p1 = F.fit_projection(fs, g, n_qubits=2, method="lda")

    shuffled = F.FeatureSet(**{**fs.__dict__, "y_test": np.random.default_rng(0).permutation(fs.y_test)})
    p2 = F.fit_projection(shuffled, g, n_qubits=2, method="lda")

    assert np.array_equal(p1.z_test, p2.z_test), "test 標籤影響了降維 → 標籤洩漏"
    assert np.array_equal(p1.z_train, p2.z_train)


def test_projection_range_within_period(fs):
    """縮放後 z ∈ [−π/2, π/2]（避免 R_y 週期首尾重疊）。"""
    g = {x.name: x for x in F.make_groups(fs)}["b1"]
    p = F.fit_projection(fs, g, n_qubits=2)
    for z in (p.z_train, p.z_test):
        assert z.min() >= -np.pi / 2 - 1e-6 and z.max() <= np.pi / 2 + 1e-6


def test_projection_dims_and_padding(fs):
    """n_components + n_padded == 2n，且補零維度恆為 0。"""
    groups = {g.name: g for g in F.make_groups(fs)}
    for name, n_q, want_comp, want_pad in (("delta", 4, 6, 2), ("b1", 2, 3, 1)):
        p = F.fit_projection(fs, groups[name], n_qubits=n_q)
        assert (p.n_components, p.n_padded) == (want_comp, want_pad)
        assert p.n_components + p.n_padded == 2 * n_q
        assert p.z_train.shape[1] == 2 * n_q
        pad = p.z_train[:, p.n_components:]
        assert np.allclose(pad, 0.0), "補零維度必須恆為 0"


def test_projection_clip_bounds_from_train(fs):
    """截斷界限取自 train 的 1%／99% 分位數。"""
    g = {x.name: x for x in F.make_groups(fs)}["b1"]
    p = F.fit_projection(fs, g, n_qubits=2)
    assert p.clip_lo.shape == (p.n_components,) and p.clip_hi.shape == (p.n_components,)
    assert np.all(p.clip_lo < p.clip_hi)
