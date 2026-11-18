"""v1 量子超參數凍結檢查（FR-015）。

此測試失敗代表有人調了 v1 的 (n, T, τ, γ 規則)——論文的停止規則禁止為了翻轉結論而這樣做。
v2 另放 pqk_v2.py，不得改動 v1 線路。
"""
import hashlib
import json
from pathlib import Path

from Research.QuSpike_PQK import pqk, run

FROZEN_PQK_SHA = "af075f13defc2006"


def _cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def test_v1_default_hyperparameters_frozen():
    """預設設定與論文 v1 完全一致。"""
    a = run.build_parser().parse_args([])
    cfg = run.make_cfg(a)
    assert cfg["n_qubits"] == {"delta": 4, "b1": 2, "all": 4}
    assert cfg["trotter_steps"] == 4
    assert cfg["evolution_time"] == 0.8
    assert cfg["gamma"] is None  # auto：中位數啟發式
    assert cfg["projection"] == "lda"
    assert pqk.DEFAULTS["trotter_steps"] == 4 and pqk.DEFAULTS["evolution_time"] == 0.8


def test_v1_circuit_source_hash_frozen():
    """pqk.py（v1 線路）的原始碼雜湊——任何修改都必須是刻意的（FR-015）。

    v2 另放 pqk_v2.py。若確實需要改 pqk.py（例如雜訊擴充），改放 noise.py 的包裝函式；
    必須修改時，更新此常數並在 PR 說明為何不違反停止規則。
    """
    src = Path(pqk.__file__).read_bytes()
    assert hashlib.sha256(src).hexdigest()[:16] == FROZEN_PQK_SHA, hashlib.sha256(src).hexdigest()[:16]


def test_v1_cfg_hash_matches_literal():
    """以字面值組出的 v1 設定雜湊等於預設設定的雜湊。"""
    a = run.build_parser().parse_args([])
    cfg = run.v1_quantum_cfg(run.make_cfg(a))
    literal = dict(n_qubits={"delta": 4, "b1": 2, "all": 4}, trotter_steps=4, evolution_time=0.8,
                   gamma=None, projection="lda", shots=None, noise_p=0.0, device="default.qubit")
    assert set(cfg) == set(literal)
    assert _cfg_hash(cfg) == _cfg_hash(literal)
