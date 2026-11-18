"""22 類同切分基線（Q9，FR-005）。"""
import numpy as np
import pytest

from Research.QuSpike_PQK import baselines22, features as F
from Research.QuSpike_PQK.run import subsample


@pytest.fixture(scope="module")
def fs():
    return subsample(F.load_features(source="mutation"), 1500)


def test_baseline_uses_same_matrices_and_22_classes(fs, tmp_path):
    res = baselines22.fit_eval("lr", fs, 0)
    assert res["config"]["n_classes"] == 22
    assert res["config"]["n_train"] == len(fs.y_train) and res["config"]["n_test"] == len(fs.y_test)
    assert np.isfinite(res["metrics_test"]["f1_macro"])


def test_lr_convergence_is_reported(fs):
    res = baselines22.fit_eval("lr", fs, 0)
    assert res["converged"] in (True, False)  # 必須明確回報，不得為 None


def test_nonconverged_flag_when_max_iter_tiny(fs, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    monkeypatch.setattr(baselines22, "build_model",
                        lambda name, seed: (LogisticRegression(max_iter=1, random_state=seed), {"max_iter": 1}))
    assert baselines22.fit_eval("lr", fs, 0)["converged"] is False
