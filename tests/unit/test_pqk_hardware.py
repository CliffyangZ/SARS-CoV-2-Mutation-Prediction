"""硬體路徑的單元測試（T040–T043）。

本階段**不送機**（使用者決定）。最重要的兩條：
- test_no_submit_without_flag：沒有明確啟用參數時，送出的作業數必須為 0（SC-006）
- test_token_never_serialized：憑證不得出現在任何產出物（SC-007）

比對邏輯以錄製的假回應測試，不需連線（FR-021／SC-012）。
"""
import json
from pathlib import Path

import numpy as np
import pytest

from Research.QuSpike_PQK import hardware as H

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pqk_hardware_response.json"
FAKE_TOKEN = "ibm_fake_token_DO_NOT_USE_0123456789"


@pytest.fixture(autouse=True)
def _no_real_credential(monkeypatch):
    """清掉環境中的真實憑證。

    必要原因：`Research/StyleGAN/utils/tracking.py` 的 load_dotenv() 會把 .env 灌進
    os.environ，使本檔案在完整套件中看到真實 token——2026-11-15 曾因此對 IBM 發出
    未預期的唯讀請求。本 fixture 與 plan_run 的 allow_network=False 是雙重防線。
    """
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)


# --- 送機閘門（SC-006）---

def test_no_submit_without_flag(monkeypatch):
    """預設乾跑：未給 submit 時，送機函式不得被呼叫，submitted_jobs 為 0。"""
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise AssertionError("不應在未啟用時送機")

    monkeypatch.setattr(H, "_submit_jobs", boom)
    est = H.plan_run(n_samples=500, n_qubits=4, shots=1024, budget_seconds=600,
                     token=FAKE_TOKEN, submit=False, backends=["fake_backend"])
    assert est["submitted_jobs"] == 0
    assert calls["n"] == 0


def test_submit_requires_explicit_flag_message():
    """未啟用時，必須說明需要哪個啟用動作（FR-017）。"""
    est = H.plan_run(n_samples=10, n_qubits=2, shots=256, budget_seconds=600,
                     token=FAKE_TOKEN, submit=False, backends=["fake_backend"])
    assert "--submit" in est["submit_hint"]


def test_estimate_only_is_the_default():
    """plan_run 的 submit 預設為 False。"""
    import inspect

    assert inspect.signature(H.plan_run).parameters["submit"].default is False


# --- 憑證（SC-007／FR-019）---

def test_missing_token_fails_clearly():
    """無憑證時必須明確失敗，且不得寫出任何硬體結果。"""
    with pytest.raises(H.CredentialError) as e:
        H.plan_run(n_samples=10, n_qubits=2, shots=256, budget_seconds=600, token=None)
    assert "IBM_QUANTUM_TOKEN" in str(e.value)


def test_token_never_serialized():
    """估算結果序列化後不得含憑證字串。"""
    est = H.plan_run(n_samples=100, n_qubits=4, shots=1024, budget_seconds=600,
                     token=FAKE_TOKEN, backends=["fake_backend"])
    blob = json.dumps(est, ensure_ascii=False)
    assert FAKE_TOKEN not in blob
    assert "token" not in {k.lower() for k in est}


def test_error_message_does_not_echo_token():
    """錯誤訊息不得回顯憑證內容。"""
    with pytest.raises(H.CredentialError) as e:
        H.plan_run(n_samples=10, n_qubits=2, shots=1, budget_seconds=600, token="   ")
    assert "   " not in str(e.value).replace("IBM_QUANTUM_TOKEN", "")


def test_read_token_from_env(monkeypatch):
    """憑證只從環境變數讀取。"""
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", FAKE_TOKEN)
    assert H.read_token() == FAKE_TOKEN
    monkeypatch.delenv("IBM_QUANTUM_TOKEN")
    assert H.read_token() is None


def test_no_network_by_default(monkeypatch):
    """預設不得連線：即使有有效憑證，也不得呼叫裝置列舉的網路路徑。"""
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise AssertionError("預設不應連線")

    monkeypatch.setattr(H, "QiskitRuntimeService_probe", boom, raising=False)
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", FAKE_TOKEN)
    est = H.plan_run(n_samples=10, n_qubits=2, shots=128, budget_seconds=1e9)
    assert est["network_used"] is False
    assert est["candidate_backends"] == []
    assert est["backend_listing_failed"] is True
    assert est["duration_source"] == "documented_constant"


def test_allow_network_is_opt_in():
    """allow_network 的預設值必須是 False。"""
    import inspect

    assert inspect.signature(H.plan_run).parameters["allow_network"].default is False


# --- 用量估算與預算（FR-020／T043）---

def test_circuit_and_job_counts():
    """n_circuits = n_samples × 3n；n_jobs 依每作業電路數上限取天花板。"""
    est = H.plan_run(n_samples=500, n_qubits=4, shots=1024, budget_seconds=1e9,
                     token=FAKE_TOKEN, backends=["fake_backend"], circuits_per_job=300)
    assert est["n_circuits"] == 500 * 3 * 4
    assert est["n_jobs"] == -(-est["n_circuits"] // 300)


def test_budget_exceeded_reports_max_samples():
    """超出預算時必須同時給出預算內可處理的最大樣本數。"""
    est = H.plan_run(n_samples=100000, n_qubits=8, shots=8192, budget_seconds=60,
                     token=FAKE_TOKEN, backends=["fake_backend"])
    assert est["exceeds_budget"] is True
    assert isinstance(est["max_samples_in_budget"], int)
    assert 0 <= est["max_samples_in_budget"] < 100000


def test_budget_ok_is_not_flagged():
    """預算充足時不得誤標超出。"""
    est = H.plan_run(n_samples=10, n_qubits=2, shots=128, budget_seconds=1e9,
                     token=FAKE_TOKEN, backends=["fake_backend"])
    assert est["exceeds_budget"] is False


def test_duration_source_is_recorded():
    """時長來源必須標明；取不到 backend 指令時長時須標為粗估。"""
    est = H.plan_run(n_samples=10, n_qubits=2, shots=128, budget_seconds=1e9,
                     token=FAKE_TOKEN, backends=["fake_backend"])
    assert est["duration_source"] in {"backend_target", "documented_constant"}
    if est["duration_source"] == "documented_constant":
        assert est["estimate_is_rough"] is True


# --- 錄製假回應的解析與比對（FR-021／SC-012）---

def test_fixture_exists_and_parses():
    """錄製的假回應存在且可解析成 φ 矩陣。"""
    assert FIXTURE.exists(), f"缺少 fixture：{FIXTURE}"
    phi = H.parse_hardware_response(json.loads(FIXTURE.read_text()))
    assert phi.ndim == 2 and phi.shape[1] % 3 == 0
    assert np.abs(phi).max() <= 1.0 + 1e-9


def test_compare_to_simulator_reports_per_dim_diff():
    """「硬體 vs 模擬器」必須給出逐維差異統計。"""
    phi_hw = H.parse_hardware_response(json.loads(FIXTURE.read_text()))
    phi_sim = phi_hw + 0.01
    cmp = H.compare_to_simulator(phi_hw, phi_sim, threshold=0.5)
    assert cmp["n_dims"] == phi_hw.shape[1]
    assert len(cmp["per_dim_mean_abs_diff"]) == phi_hw.shape[1]
    assert abs(cmp["max_abs_diff"] - 0.01) < 1e-6
    assert cmp["comparable"] is True


def test_large_discrepancy_marks_incomparable():
    """差異超過門檻時必須標為不可比，而非直接並列（FR-018）。"""
    phi_hw = H.parse_hardware_response(json.loads(FIXTURE.read_text()))
    cmp = H.compare_to_simulator(phi_hw, phi_hw * -1.0, threshold=0.1)
    assert cmp["comparable"] is False


def test_shape_mismatch_rejected():
    """形狀不符必須明確報錯。"""
    phi = H.parse_hardware_response(json.loads(FIXTURE.read_text()))
    with pytest.raises(ValueError):
        H.compare_to_simulator(phi, phi[:, :-3])


# --- 續跑（FR-022）---

def test_resume_skips_completed_samples(tmp_path):
    """已完成的樣本不重算。"""
    done = H.completed_samples(tmp_path, "key1")
    assert done == set()
    H.mark_completed(tmp_path, "key1", [0, 1, 5])
    assert H.completed_samples(tmp_path, "key1") == {0, 1, 5}
    assert H.pending_samples(tmp_path, "key1", n_samples=8) == [2, 3, 4, 6, 7]
