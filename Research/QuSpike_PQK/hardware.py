"""硬體路徑：就緒但不送機（T044–T050）。

**本階段的核心約束**（使用者決定 2026-11-15，見 spec.md Out of Scope）：
預設行為是乾跑；沒有明確的 submit 啟用參數時，任何路徑都不得送出作業（FR-017）。
這不是風格選擇——免費方案的裝置時間是每月數分鐘級，而完整資料集的量子特徵需要
上萬次線路執行，一次意外送機就可能把配額用完。

憑證只從環境變數讀取，且不得出現在任何結果檔、日誌或錯誤訊息中（FR-019）。

⚠️ **未驗證路徑**：憑證驗證與裝置列舉需要網路連線與使用者的 token。研究階段刻意
未執行（把使用者憑證送往外部服務應由使用者自行決定），故 `_list_backends` 在
首次實際執行前屬未驗證。
"""
import json
import math
import os
from pathlib import Path

import numpy as np

ENV_VAR = "IBM_QUANTUM_TOKEN"
# 每次線路執行的粗估時長（秒／shot）。取不到 backend 指令時長表時使用，並標為粗估。
DOCUMENTED_SECONDS_PER_SHOT = 2.5e-4
CIRCUITS_PER_JOB = 300


class CredentialError(RuntimeError):
    """憑證缺失或無效。訊息不得回顯憑證內容。"""


def read_token():
    """只從環境變數讀取憑證。回傳 None 表示未設定。"""
    v = os.environ.get(ENV_VAR)
    return v if v and v.strip() else None


def _require_token(token):
    t = token if token is not None else read_token()
    if t is None or not str(t).strip():
        raise CredentialError(
            f"找不到可用的硬體憑證。請在環境變數 {ENV_VAR} 中設定（.env 已被 .gitignore 排除）。")
    return t


def _list_backends(token, min_qubits, allow_network=False):
    """列舉候選裝置（唯讀，不送出作業）。

    ⚠️ **需要明確啟用**（allow_network=True）。預設不連線——把「不連線」寄託在
    環境變數剛好沒設是不可靠的：`Research/StyleGAN/utils/tracking.py` 的 load_dotenv()
    會把 .env 灌進 os.environ，使任何後續程序都看得到真實憑證。2026-11-15 的完整
    測試套件就因此對 IBM 發出了未預期的唯讀請求。
    """
    if not allow_network:
        return []
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        return [b.name for b in service.backends(simulator=False, operational=True,
                                                 min_num_qubits=int(min_qubits))]
    except Exception:
        return []


def _seconds_per_shot(token, backend_name, allow_network=False):
    """取每 shot 的估算時長。回傳 (秒, 來源)。**預設不連線**（需 allow_network=True）。"""
    if backend_name and token and allow_network:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            target = service.backend(backend_name).target
            durs = [p.duration for ins in target.values() for p in ins.values()
                    if getattr(p, "duration", None)]
            if durs:
                return float(np.mean(durs)) * 2.0, "backend_target"
        except Exception:
            pass
    return DOCUMENTED_SECONDS_PER_SHOT, "documented_constant"


def _submit_jobs(*args, **kwargs):
    """真正送出作業。本階段不呼叫；保留以免未來送機時重寫介面。"""
    raise NotImplementedError(
        "實際送機已列為 Out of Scope（spec.md）。要啟用需傳 submit=True 並自行確認配額。")


def plan_run(n_samples, n_qubits, shots, budget_seconds, token=None, submit=False,
             backends=None, circuits_per_job=CIRCUITS_PER_JOB, backend=None,
             allow_network=False):
    """乾跑：估算用量並檢查預算。

    **兩道預設關閉的閘門**：
    - `submit=False`（FR-017）：不送出任何作業。
    - `allow_network=False`：**完全不連線**，純離線估算。網路存取與送機一樣必須明確啟用。

    第二道閘門是 2026-11-15 的事故後補上的：完整測試套件因 load_dotenv() 把 .env
    灌進環境，導致本函式以真實憑證對 IBM 發出唯讀請求。把「不連線」寄託在環境變數
    剛好沒設是不可靠的，必須寫進程式結構。
    """
    tok = _require_token(token)
    n_circuits = int(n_samples) * 3 * int(n_qubits)
    n_jobs = int(math.ceil(n_circuits / max(int(circuits_per_job), 1)))
    cands = (list(backends) if backends is not None
             else _list_backends(tok, n_qubits, allow_network=allow_network))
    chosen = backend or (cands[0] if cands else None)
    sec_per_shot, src = _seconds_per_shot(None if backends is not None else tok, chosen,
                                          allow_network=allow_network)
    est = n_circuits * int(shots) * sec_per_shot
    per_sample = 3 * int(n_qubits) * int(shots) * sec_per_shot
    max_in_budget = int(max(0, math.floor(float(budget_seconds) / per_sample))) if per_sample > 0 else 0

    out = dict(candidate_backends=cands, chosen_backend=chosen, n_samples=int(n_samples),
               n_qubits=int(n_qubits), shots=int(shots), n_circuits=n_circuits, n_jobs=n_jobs,
               estimated_device_seconds=float(est), duration_source=src,
               estimate_is_rough=(src == "documented_constant"),
               budget_seconds=float(budget_seconds),
               exceeds_budget=bool(est > float(budget_seconds)),
               max_samples_in_budget=max_in_budget,
               submitted_jobs=0,
               submit_hint="本階段預設乾跑；要實際送機須明確傳入 --submit 並自行確認配額",
               backend_listing_failed=(backends is None and not cands),
               network_used=bool(allow_network))
    if submit:
        _submit_jobs(out)  # 預設不會走到；呼叫端必須明確啟用
    return out


# --- 硬體結果解析與比對（FR-018／FR-021；以錄製假回應測試）---

def parse_hardware_response(doc):
    """把硬體回應解析成 φ 矩陣 (n_samples, 3n)。"""
    rows = doc["results"]
    rows = sorted(rows, key=lambda r: int(r["sample_index"]))
    phi = np.asarray([r["expectation_values"] for r in rows], np.float64)
    if phi.ndim != 2 or phi.shape[1] % 3 != 0:
        raise ValueError(f"期望值維度應為 3n，得到 {phi.shape}")
    return np.clip(phi, -1.0, 1.0)


def compare_to_simulator(phi_hw, phi_sim, threshold=0.2):
    """硬體 vs 模擬器的逐維差異統計。差異超過門檻時標為不可比（FR-018）。"""
    a = np.asarray(phi_hw, np.float64)
    b = np.asarray(phi_sim, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"形狀不符：硬體 {a.shape} vs 模擬器 {b.shape}")
    d = np.abs(a - b)
    return dict(n_samples=int(a.shape[0]), n_dims=int(a.shape[1]),
                per_dim_mean_abs_diff=[float(x) for x in d.mean(0)],
                mean_abs_diff=float(d.mean()), max_abs_diff=float(d.max()),
                threshold=float(threshold), comparable=bool(d.max() <= float(threshold)))


# --- 續跑（FR-022）---

def _ledger(state_dir, key):
    return Path(state_dir) / f"hw_done_{key}.json"


def completed_samples(state_dir, key):
    """已完成的樣本索引集合。"""
    p = _ledger(state_dir, key)
    if not p.exists():
        return set()
    try:
        return set(int(x) for x in json.loads(p.read_text())["completed"])
    except Exception:
        return set()


def mark_completed(state_dir, key, indices):
    """登記已完成的樣本（以 cache_key 為單位，不含種子）。"""
    p = _ledger(state_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    done = completed_samples(state_dir, key) | {int(i) for i in indices}
    p.write_text(json.dumps({"completed": sorted(done)}), encoding="utf-8")
    return done


def pending_samples(state_dir, key, n_samples):
    """尚未完成的樣本索引（FR-022：已完成者不重算）。"""
    done = completed_samples(state_dir, key)
    return [i for i in range(int(n_samples)) if i not in done]
