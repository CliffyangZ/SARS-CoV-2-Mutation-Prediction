"""規模一、20 步、離線 wandb 能完成。"""
import json
import numpy as np
import pandas as pd
from Research.train import train_classical as tc
from Research.utils.config import make_config


def test_smoke(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=20, model="classical")
    X = (np.random.rand(100, 64) < 0.05).astype(np.uint8)
    G, D = tc.build_models(cfg)
    from Research.train.loop import train
    st = train(cfg, G, D, X, tmp_path, "offline")
    assert st["state"] == "finished"
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "finished"
    assert len(pd.read_csv(tmp_path / "losses.csv")) == 1      # 第 0 步（20 步內只有一個記錄點）
    assert (tmp_path / "checkpoint.pt").exists() and (tmp_path / "config.json").exists()
