"""008 端到端冒煙測試：小資料跑 leakcheck、ceiling、軟路由＋補維、approx、report。"""
import json

from Research.QuSpike_PQK import run


def test_pipeline_extras_smoke(tmp_path):
    out = tmp_path / "opt"
    assert run.main(["leakcheck", "--limit", "700", "--controls", "C0,C1", "--seeds", "0",
                     "--out", str(out), "--no-tune"]) == 0
    assert run.main(["--limit", "900", "--seeds", "0", "--all-seeds", "0", "1", "2", "3", "4",
                     "--controls", "MAIN,C0,C1", "--routing", "soft", "--soft-threshold", "0.2",
                     "--supplement", "pls", "--out", str(out), "--no-tune"]) == 0
    r = json.loads(next(out.glob("*MAIN*seed0.json")).read_text())
    assert r["config"]["inference_mode"] == "label_free" and r["config"]["routing"] == "soft"
    assert r["config"]["seeds"] == [0, 1, 2, 3, 4]
    assert all(v == [] for v in r["decoupled_qubits"].values())
    assert run.main(["approx", "--limit", "900", "--controls", "C1", "--seeds", "0",
                     "--out", str(tmp_path / "ap"), "--no-tune"]) == 0
    assert (tmp_path / "ap" / "approx_check.json").exists()


def test_report_has_nine_rows(tmp_path):
    from Research.QuSpike_PQK import opt_report

    md = opt_report.build(dict(base=tmp_path / "none"), registry_root=tmp_path / "reg")
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"):
        assert f"| {q} " in md
    assert "論文待更新段落" in md
