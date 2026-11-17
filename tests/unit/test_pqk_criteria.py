"""判準自動判定的單元測試（T016）。

對應 SC-002：四項判準全部由程式判定為 holds／fails／within_noise，
不存在需人工解讀的欄位。
"""
import numpy as np

from Research.QuSpike_PQK import criteria as C


def test_verdict_requires_both_noise_floor_and_seed_std():
    """holds 須同時超過雜訊下限與兩組的種子標準差。"""
    d, v = C.verdict((0.80, 0.0), (0.70, 0.0))
    assert v == "holds" and abs(d - 0.10) < 1e-12
    # 差異大於下限，但未超過種子標準差 → 仍在雜訊內
    _, v = C.verdict((0.80, 0.15), (0.70, 0.01))
    assert v == "within_noise"


def test_verdict_within_noise_below_floor():
    """小於雜訊下限即視為雜訊內，即使標準差為 0。"""
    _, v = C.verdict((0.7150, 0.0), (0.7000, 0.0))
    assert v == "within_noise"


def test_verdict_fails_when_clearly_worse():
    """明顯更差判為 fails。"""
    d, v = C.verdict((0.60, 0.0), (0.70, 0.0))
    assert v == "fails" and d < 0


def test_verdict_three_states_only():
    """除資料不足外，只能出現三態。"""
    seen = set()
    for a in (0.5, 0.7, 0.9):
        for sd in (0.0, 0.05):
            seen.add(C.verdict((a, sd), (0.7, 0.0))[1])
    assert seen <= {"holds", "fails", "within_noise"}


def test_insufficient_data_is_explicit():
    """缺資料時必須明確標示，不得假裝有結論。"""
    _, v = C.verdict((float("nan"), 0.0), (0.7, 0.0))
    assert v == "insufficient_data"
    recs = C.evaluate({"MAIN": [0.7, 0.71, 0.72]})  # 沒有任何對照組
    assert all(r["verdict"] == "insufficient_data" for r in recs)


def test_evaluate_returns_four_claims():
    """四項判準都要有紀錄，且欄位完整。"""
    s = {"MAIN": [0.75, 0.76, 0.74], "C0": [0.70, 0.70, 0.70], "C1": [0.71, 0.72, 0.70],
         "C2": [0.69, 0.70, 0.71], "C3": [0.74, 0.75, 0.73], "C4": [0.72, 0.71, 0.73],
         "C5": [0.73, 0.74, 0.72]}
    recs = C.evaluate(s)
    assert len(recs) == 4
    for r in recs:
        assert set(r) == {"claim", "a", "b", "delta", "std_a", "std_b", "noise_floor", "verdict"}
        assert np.isfinite(r["delta"])


def test_quantum_feature_claim_uses_strongest_classical_control():
    """「量子特徵有用」必須對上最強的古典對照（C1／C2／C4 中平均最高者）。"""
    s = {"MAIN": [0.75], "C1": [0.60], "C2": [0.74], "C4": [0.65]}
    rec = [r for r in C.evaluate(s) if r["claim"] == "量子特徵有用"][0]
    assert rec["b"] == "C2", "挑了較弱的對照組會高估量子的增量"
    assert rec["verdict"] == "within_noise"


def test_entanglement_claim_compares_main_to_c3():
    """「量子性有用」比的是 MAIN 對 C3（無糾纏）。"""
    rec = [r for r in C.evaluate({"MAIN": [0.80], "C3": [0.70]})
           if r["claim"].startswith("量子性")][0]
    assert rec["a"] == "MAIN" and rec["b"] == "C3" and rec["verdict"] == "holds"


def test_stop_condition_triggers_when_both_fail():
    """兩項關鍵判準都不成立時，停止條件成立（提案 §4）。"""
    s = {"MAIN": [0.70], "C0": [0.69], "C1": [0.70], "C2": [0.70], "C3": [0.70], "C5": [0.70]}
    assert C.stop_condition_met(C.evaluate(s)) is True


def test_stop_condition_not_met_when_any_holds():
    """任一項成立時不觸發停止條件。"""
    s = {"MAIN": [0.90], "C0": [0.70], "C1": [0.70], "C2": [0.70], "C3": [0.70], "C5": [0.70]}
    assert C.stop_condition_met(C.evaluate(s)) is False


def test_noise_floor_is_preregistered():
    """雜訊下限必須是事前訂定的 0.02。"""
    assert C.NOISE_FLOOR == 0.02
