"""端到端整合測試（T017）。

最重要的一條是 test_exit_code_zero_even_when_claims_fail：負面結果是有效結論，
不是執行錯誤（FR-027／SC-010）。前一輪的經驗是，若管線在判準不成立時中止或報錯，
就會產生「只有正面結果才跑得完」的選擇偏差。
"""
import json

import pytest

from Research.QuSpike_PQK import criteria, report, run

LIMIT = 1200


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """以小樣本跑完七組設定（單一種子，關閉調權以加速）。"""
    out = tmp_path_factory.mktemp("pqk_run")
    code = run.main(["--limit", str(LIMIT), "--seeds", "0", "--out", str(out), "--no-tune"])
    return out, code


def test_exit_code_zero_even_when_claims_fail(ran):
    """FR-027：任一判準不成立，離開碼仍為 0。"""
    out, code = ran
    assert code == 0


def test_all_seven_controls_produce_results(ran):
    """七組設定各產出一個結果檔。"""
    out, _ = ran
    files = sorted(out.glob("*.json"))
    assert len(files) == 7, [f.name for f in files]
    controls = {json.loads(f.read_text())["config"]["control"] for f in files}
    assert controls == set(run.CONTROLS)


def test_results_record_provenance(ran):
    """結果檔必須含設定、切分版本與環境，足以重建執行。"""
    out, _ = ran
    r = json.loads(next(out.glob("*MAIN*.json")).read_text())
    assert r["split_version"] and "+" in r["split_version"]
    assert r["config"]["features"] == "mutation_indicator"
    assert r["config"]["features_selection_reason"]
    assert r["environment"]
    assert r["config"]["backend"] == "simulator"


def test_cache_key_has_no_seed(ran):
    """量子特徵的快取鍵不含種子（研究 D2）。"""
    out, _ = ran
    r = json.loads(next(out.glob("*MAIN*.json")).read_text())
    keys = r["quantum_feature_cache_key"]
    assert keys, "MAIN 應使用量子特徵"
    for _, v in keys.items():
        assert "seed" not in v["cache_key"]


def test_main_and_c3_differ_only_in_entangle(ran):
    """MAIN 與 C3 只差 entangle 一個欄位（FR-011：量子性消融必須乾淨）。"""
    out, _ = ran
    a = json.loads(next(out.glob("*MAIN*.json")).read_text())["config"]
    b = json.loads(next(out.glob("*C3*.json")).read_text())["config"]
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff <= {"entangle", "name", "control", "expert_scale"}, f"多餘差異：{diff}"
    assert a["entangle"] is True and b["entangle"] is False


def test_controls_share_features_and_projection(ran):
    """七組設定共用同一組特徵與降維方式（FR-011）。"""
    out, _ = ran
    cfgs = [json.loads(f.read_text())["config"] for f in out.glob("*.json")]
    assert len({c["features"] for c in cfgs}) == 1
    assert len({c["projection"] for c in cfgs}) == 1
    assert len({c["n_classes"] for c in cfgs}) == 1


def test_group_metrics_present(ran):
    """分群分數必須存在（FR-024）。"""
    out, _ = ran
    r = json.loads(next(out.glob("*MAIN*.json")).read_text())
    assert set(r["group_metrics_test"]) >= {"delta", "b1", "others"}


def test_report_builds_and_has_required_sections(ran):
    """報告可產出，且含判準、分群、錨點三節，並註明上界為樂觀估計。"""
    out, _ = ran
    txt = report.build(out)
    for need in ("判準", "分群", "錨點", "0.747", "0.930", "樂觀估計", "未送出任何硬體作業"):
        assert need in txt, f"報告缺少：{need}"


def test_report_rejects_hardware_claims(ran, tmp_path):
    """FR-028：若結果檔聲稱來自硬體，報告必須拒絕產出。"""
    out, _ = ran
    src = next(out.glob("*MAIN*.json"))
    r = json.loads(src.read_text())
    r["config"]["backend"] = "hardware"
    (tmp_path / "fake__mut__seed0.json").write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(ValueError, match="FR-028"):
        report.build(tmp_path)


def test_criteria_verdicts_are_machine_generated(ran):
    """四項判準全部由程式判定為三態之一（SC-002）。"""
    out, _ = ran
    g = report.collect(out)
    recs = criteria.evaluate({n: d["vals"] for n, d in g.items()})
    assert len(recs) == 4
    assert all(r["verdict"] in {"holds", "fails", "within_noise", "insufficient_data"}
               for r in recs)
