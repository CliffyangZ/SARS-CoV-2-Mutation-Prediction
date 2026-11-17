"""CLI smoke 測試（T054）。

重點是確認 hardware 子命令在任何情況下都不連線、不送機。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def _run(args, env=None, timeout=600):
    import os

    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT)
    if env:
        e.update(env)
    return subprocess.run([PY, "-W", "ignore::UserWarning", "-m", "Research.QuSpike_PQK.run", *args],
                          cwd=ROOT, env=e, capture_output=True, text=True, timeout=timeout)


def test_help_works():
    """主命令可啟動。"""
    r = _run(["--help"])
    assert r.returncode == 0 and "QuSpike-PQK" in r.stdout


def test_pipeline_runs_on_tiny_sample(tmp_path):
    """小樣本管線可跑完，離開碼 0，且產出七個結果檔。"""
    r = _run(["--limit", "600", "--seeds", "0", "--out", str(tmp_path), "--no-tune"])
    assert r.returncode == 0, r.stderr[-2000:]
    assert len(list(tmp_path.glob("*.json"))) == 7
    assert "判準" in r.stdout


def test_hardware_without_credential_fails_cleanly(tmp_path):
    """無憑證時明確失敗，且不寫出任何結果檔（FR-019）。"""
    r = _run(["hardware", "--samples", "10", "--out", str(tmp_path)],
             env={"IBM_QUANTUM_TOKEN": ""})
    assert r.returncode == 2
    assert "IBM_QUANTUM_TOKEN" in r.stdout
    assert not list(tmp_path.glob("hardware_estimate.json"))


def test_hardware_help_shows_submit_is_opt_in():
    """hardware 子命令的說明必須標示送機是選擇性啟用。"""
    r = _run(["hardware", "--help"])
    assert r.returncode == 0
    assert "--submit" in r.stdout and "不送機" in r.stdout


def test_hardware_never_submits_without_flag(tmp_path):
    """SC-006：未給 --submit 時，送出的作業數為 0。

    以假憑證 + 注入的候選裝置清單驗證，完全不連線。
    """
    from Research.QuSpike_PQK import hardware as H

    est = H.plan_run(n_samples=250, n_qubits=4, shots=1024, budget_seconds=600,
                     token="fake_token_for_smoke", backends=["fake_backend"])
    assert est["submitted_jobs"] == 0
    blob = json.dumps(est)
    assert "fake_token_for_smoke" not in blob


def test_report_cli_runs(tmp_path):
    """報告 CLI 可在有結果的目錄上執行。"""
    r = _run(["--limit", "600", "--seeds", "0", "--out", str(tmp_path), "--no-tune"])
    assert r.returncode == 0
    import os

    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT)
    rr = subprocess.run([PY, "-m", "Research.QuSpike_PQK.report", "--out", str(tmp_path)],
                        cwd=ROOT, env=e, capture_output=True, text=True, timeout=300)
    assert rr.returncode == 0 and "QuSpike-PQK 結果" in rr.stdout
    assert "未送出任何硬體作業" in rr.stdout
