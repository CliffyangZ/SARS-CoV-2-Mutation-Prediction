"""資料多樣性上限分析（004 US1）：相異數、抽樣、判定規則與可重現。"""
import numpy as np

from Research.StyleGAN.evaluate import ceiling as C


def test_n_distinct_and_effective_n():
    """相異列數與有效樣本數。"""
    X = np.array([[1, 0], [1, 0], [0, 1]])
    assert C.n_distinct(X) == 2
    assert C.effective_n([1, 1, 1, 1]) == 4.0
    assert C.effective_n([4, 0.0001]) < 1.01


def test_resample_known_answer():
    """只有 1 種向量時永遠只有 1 種；兩種等權重抽 1000 條幾乎必得 2 種。"""
    X1 = np.array([[1, 0, 1]] * 5)
    assert set(C.resample_distinct(X1, np.ones(5), 100, 5, 0)) == {1}
    X2 = np.array([[1, 0], [0, 1]])
    assert set(C.resample_distinct(X2, np.ones(2), 1000, 5, 0)) == {2}


def test_resample_weights_matter():
    """權重極度集中時，抽樣相異數遠小於相異向量數。"""
    X = np.eye(50, dtype=int)
    w = np.array([1000.0] + [0.01] * 49)
    assert C.resample_distinct(X, w, 100, 10, 0).mean() < 5


def test_independent_distinct_bounds():
    """頻率 0 的位點永遠不出現；全 0 頻率只有 1 種。"""
    assert set(C.independent_distinct(np.zeros(8), 200, 3, 0)) == {1}
    assert C.independent_distinct(np.full(16, 0.5), 200, 3, 0).min() > 100


def test_same_seed_same_result():
    """同種子重跑結果相同（FR-003）。"""
    rng = np.random.default_rng(1)
    X = (rng.random((60, 10)) < 0.3).astype(int); w = rng.integers(1, 20, 60); sp = np.array(["train"] * 40 + ["test"] * 20)
    a, b = C.analyze(X, w, sp, 200, 5, 3), C.analyze(X, w, sp, 200, 5, 3)
    assert a == b and a["verdict"] in ("reasonable", "too_high", "undetermined")


def test_verdict_rules():
    """≥120 合理；<100 偏高且修訂為 70% 取整；其餘無法判定、維持 100。"""
    assert C.verdict_of(150) == ("reasonable", 100)
    assert C.verdict_of(120) == ("reasonable", 100)
    assert C.verdict_of(60) == ("too_high", 42)
    assert C.verdict_of(99.9) == ("too_high", 69)
    assert C.verdict_of(110) == ("undetermined", 100)


def test_report_files(tmp_path):
    """寫出 ceiling.json 與 ceiling_report.md，且內容含判定。"""
    rng = np.random.default_rng(0)
    X = (rng.random((30, 6)) < 0.4).astype(int); w = np.ones(30); sp = np.array(["train"] * 20 + ["test"] * 10)
    C.write_report(C.analyze(X, w, sp, 50, 3, 0), tmp_path)
    assert (tmp_path / "ceiling.json").exists() and "判定" in (tmp_path / "ceiling_report.md").read_text()
