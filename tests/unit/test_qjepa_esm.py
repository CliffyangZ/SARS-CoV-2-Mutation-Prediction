"""ESM 基準：視窗涵蓋整條序列。"""
from Research.QuSpike_JEPA.esm_baseline import STRIDE, WINDOW, windows


def test_windows_cover_sequence():
    for n in (500, 1000, 1273, 1277):
        ws = windows(n)
        covered = set()
        for s in ws:
            covered.update(range(s, min(s + WINDOW, n)))
        assert covered == set(range(n)) and all(s + 1 <= n for s in ws)


def test_align_pairs_marks_substitutions_and_skips_deletions():
    from Research.QuSpike_JEPA.esm_baseline import align_pairs
    from Research.QuSpike_JEPA.mutations import REFERENCE as R

    j, i, mut = align_pairs(R[:613] + "G" + R[614:])
    assert len(j) == len(R) and mut.sum() == 1 and i[mut][0] == 613
    j, i, mut = align_pairs(R[:100] + R[103:])
    assert len(j) == len(R) - 3 and not mut.any() and 100 not in i and 101 not in i and 102 not in i


def test_site_features_shape_and_zero_site_when_identical():
    import torch

    from Research.QuSpike_JEPA.esm_baseline import site_features
    from Research.QuSpike_JEPA.mutations import REFERENCE as R

    h_ref = torch.randn(len(R), 8)
    f = site_features(h_ref.clone(), h_ref, R)
    assert f.shape == (24,) and f[8:].abs().max() == 0.0
    f2 = site_features(h_ref.clone(), h_ref, R[:10] + "W" + R[11:])
    assert f2.shape == (24,)
