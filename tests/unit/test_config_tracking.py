"""設定與追蹤：離線可完成、不洩漏金鑰。"""
import os
import pandas as pd
from Research.utils.config import make_config, save_config, load_config
from Research.utils.tracking import Tracker


def test_config_roundtrip(tmp_path):
    c = make_config(2, seed=3)
    assert c.L == 1024 and c.n_qubits == 10 and len(c.block_sizes) == 5 and c.k == 3
    save_config(c, tmp_path / "c.json")
    assert load_config(tmp_path / "c.json").seed == 3
    assert make_config(1).block_schedule[-1][1] == 3500


def test_offline_tracking_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "SECRET-KEY-123")   # 設了也不能出現在輸出
    t = Tracker(tmp_path, {"a": 1}, mode="offline")
    for i in range(3):
        t.log(i * 74, 0, 0.5, 1.0, 2.0)
    t.close()
    df = pd.read_csv(tmp_path / "losses.csv")
    assert list(df.columns) == ["step", "block", "fade_alpha", "loss_g", "loss_d"] and len(df) == 3
    for p in tmp_path.rglob("*"):
        if p.is_file() and p.suffix in {".csv", ".json", ".yaml", ".txt", ".log"}:
            assert "SECRET-KEY-123" not in p.read_text(errors="ignore")
