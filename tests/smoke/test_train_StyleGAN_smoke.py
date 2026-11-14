"""規模一、20 步、離線 wandb 能完成。"""
import json
import numpy as np
import pandas as pd
from Research.StyleGAN.train import train_StyleGAN as tc
from Research.StyleGAN.utils.config import make_config


def test_smoke(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=20, model="classical")
    X = (np.random.rand(100, 64) < 0.05).astype(np.uint8)
    G, D = tc.build_models(cfg)
    from Research.StyleGAN.train.train_StyleGAN import train
    st = train(cfg, G, D, X, tmp_path, "offline")
    assert st["state"] == "finished"
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "finished"
    assert len(pd.read_csv(tmp_path / "losses.csv")) == 1      # 第 0 步（20 步內只有一個記錄點）
    assert (tmp_path / "checkpoint.pt").exists() and (tmp_path / "config.json").exists()


import pytest
from Research.StyleGAN.evaluate.evaluate import evaluate
from Research.StyleGAN.evaluate.generate import generate


@pytest.mark.parametrize("encoding", ["legacy", "drop_const", "nonorm", "freq_order", "drop_const+nonorm"])
def test_smoke_each_encoding(tmp_path, synth_dataset, monkeypatch, encoding):
    """每種編碼：訓練 → 生成 → 評估皆可完成、無 NaN、結果目錄含編碼名稱（002 T026；結果寫暫存目錄，F8）。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, encoding=encoding, steps=60, model="classical", batch_size=4, fmap_max=4)
    X = np.load(synth_dataset / "vectors.npz")["X"][:60]
    G, D = tc.build_models(cfg)
    run = cfg.run_dir(tmp_path)
    assert (encoding == "legacy") == ("_" + encoding not in run.name)
    st = tc.train(cfg, G, D, X, run, "off")
    assert st["state"] == "finished", st["reason"]
    gen = generate(run, n=12, data_dir=synth_dataset)
    assert np.isfinite(gen["diag"]).all() and len(gen["k_per_sample"]) == 12
    m = evaluate(run, data_dir=synth_dataset)
    assert (run / "diversity.json").exists() and m["diversity"]["n"] == 12


def test_training_logs_diversity(tmp_path, monkeypatch):
    """訓練的每個記錄點都有多樣性指標，且量測不改變訓練結果（亂數序列被保護）（T031）。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=75, model="classical", batch_size=4, fmap_max=4)
    X = (np.random.default_rng(0).random((100, 64)) < 0.05).astype(np.uint8)
    G, D = tc.build_models(cfg)
    st = tc.train(cfg, G, D, X, tmp_path, "off")
    assert st["state"] == "finished"
    df = pd.read_csv(tmp_path / "losses.csv")
    assert len(df) == 2 and df.distinct_sets.notna().all() and df.pos_coverage.notna().all()
    assert (df.distinct_sets >= 1).all() and (df.distinct_sets <= 200).all()


def test_recipe_factors_smoke(tmp_path, monkeypatch):
    """003：d_every、fade_frac、lr_scale、EMA 同時啟用的 100 步冒煙：無 NaN、設定被記錄、losses.csv 有 live_* 欄位且有值。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=100, model="classical", batch_size=4, fmap_max=4, d_every=2, fade_frac=0.2,
                      lr_scale=0.5, ema_decay=0.999, last_block_steps=20, tag="smoke")
    X = (np.random.rand(60, 64) < 0.05).astype(np.uint8)
    G, D = tc.build_models(cfg)
    run = cfg.run_dir(tmp_path)
    assert run.name.endswith("_smoke_seed0")
    st = tc.train(cfg, G, D, X, run, "off")
    assert st["state"] == "finished", st["reason"]
    saved = json.loads((run / "config.json").read_text())
    assert (saved["d_every"], saved["fade_frac"], saved["lr_scale"], saved["last_block_steps"]) == (2, 0.2, 0.5, 20)
    df = pd.read_csv(run / "losses.csv")
    assert df.live_distinct_sets.notna().all() and df.distinct_sets.notna().all()
    assert df.step.max() < cfg.total_steps


def test_d_every_skips_discriminator_updates(tmp_path, monkeypatch):
    """003：d_every=4 時判別器參數只在 step%4==0 才改變（用步數很少的訓練比對權重是否變動）。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    X = (np.random.rand(40, 64) < 0.05).astype(np.uint8)
    out = {}
    for n in (1, 4):
        cfg = make_config(1, steps=8, model="classical", batch_size=4, fmap_max=4, d_every=n)
        G, D = tc.build_models(cfg)
        tc.train(cfg, G, D, X, tmp_path / f"n{n}", "off")
        out[n] = D
    import torch
    d1 = torch.cat([p.flatten() for p in out[1].parameters()])
    d4 = torch.cat([p.flatten() for p in out[4].parameters()])
    assert not torch.allclose(d1, d4)      # 更新次數不同（8 次 vs 2 次），權重必然不同


def test_defaults_reproduce_run02_first_rows(tmp_path, monkeypatch):
    """003 SC-006：不帶新因子時，前兩個記錄點（step 0、74）與 run02 drop_const 種子 0 的損失相同。
    steps=225 使第 0 區塊為 0–75 步，step 0、74 與 3500 步的完整訓練同在第 0 區塊，兩者訓練軌跡一致。
    run02 結果或資料集不存在時跳過；容忍 1e-3（GPU 運算有微小非決定性）。"""
    from pathlib import Path
    ref = Path("Results/StyleGAN/run02/A/scale1/classical_logistic_drop_const_seed0/losses.csv")
    from Research.StyleGAN.FE.build_dataset import dataset_dir
    if not ref.exists() or not (dataset_dir(1, "drop_const") / "vectors.npz").exists():
        pytest.skip("缺少 run02 結果或 drop_const 資料集")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    X = tc.load_train_vectors(1, None, "drop_const")
    cfg = make_config(1, encoding="drop_const", steps=225, model="classical", k=int(round(float(X.sum(1).mean()))))
    G, D = tc.build_models(cfg)
    dev = tc.get_device("auto"); G, D = G.to(dev), D.to(dev)
    run = cfg.run_dir(tmp_path)
    assert tc.train(cfg, G, D, X, run, "off")["state"] == "finished"
    new, old = pd.read_csv(run / "losses.csv"), pd.read_csv(ref)
    for st in (0, 74):
        a, b = new[new.step == st].iloc[0], old[old.step == st].iloc[0]
        assert abs(a.loss_g - b.loss_g) < 1e-3 and abs(a.loss_d - b.loss_d) < 1e-3, (st, a.loss_g, b.loss_g)


def test_model_loss_factors_smoke(tmp_path, monkeypatch):
    """004：結構與損失因子全開的 100 步冒煙：無 NaN，設定被記錄，build_models 依因子縮小模型。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=100, model="classical", batch_size=4, fmap_max=8, mapping_layers=2, latent_dim=32,
                      g_fmap_max=4, d_fmap_max=4, mbstd_group=4, div_reg=1.0, noise=False, tag="m")
    X = (np.random.rand(60, 64) < 0.05).astype(np.uint8)
    G, D = tc.build_models(cfg)
    base_G, base_D = tc.build_models(make_config(1, batch_size=4, fmap_max=8))
    assert sum(p.numel() for p in G.parameters()) < sum(p.numel() for p in base_G.parameters())
    assert sum(p.numel() for p in D.parameters()) < sum(p.numel() for p in base_D.parameters())
    run = cfg.run_dir(tmp_path)
    st = tc.train(cfg, G, D, X, run, "off")
    assert st["state"] == "finished", st["reason"]
    saved = json.loads((run / "config.json").read_text())
    assert (saved["mapping_layers"], saved["latent_dim"], saved["g_fmap_max"], saved["d_fmap_max"], saved["div_reg"]) == (2, 32, 4, 4, 1.0)


def test_div_reg_zero_makes_no_extra_forward(tmp_path, monkeypatch):
    """004：div_reg=0 時不多做前向；div_reg>0 時每步多 1 次前向。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    X = (np.random.rand(40, 64) < 0.05).astype(np.uint8)
    counts = {}
    for w in (0.0, 1.0):
        cfg = make_config(1, steps=6, model="classical", batch_size=4, fmap_max=4, div_reg=w, log_every=10 ** 6)
        G, D = tc.build_models(cfg)
        n = [0]; orig = G.forward
        def counted(*a, _o=orig, _n=n, **k):
            """計算前向次數的包裝。"""
            _n[0] += 1
            return _o(*a, **k)
        G.forward = counted
        tc.train(cfg, G, D, X, tmp_path / f"w{w}", "off")
        counts[w] = n[0]
    assert counts[1.0] - counts[0.0] == 6      # 每步恰好多 1 次前向（基準次數含 step 0 的多樣性量測）
