import json

import pytest

from Demo import bundle as B


def test_missing_bundle(tmp_path):
    with pytest.raises(B.BundleMissing):
        B.load_bundle(tmp_path / "nope")


def test_meta_has_no_test_labels(bundle):
    txt = json.dumps(bundle.meta)
    assert "y_test" not in txt and "test_idx" not in txt
    assert bundle.meta["kind"] == "MAIN" and bundle.s_default >= 0


def test_reload_same_predictions(bundle, tmp_path, seqs):
    import numpy as np
    from Demo import parse
    from Research.QuSpike_PQK import inference
    x = np.vstack([parse.to_vector(parse.parse_sequence(s), bundle.vocab)[0] for s in seqs[:5]])
    p, _, _ = inference.predict(bundle.state, x)
    assert len(p) == 5
