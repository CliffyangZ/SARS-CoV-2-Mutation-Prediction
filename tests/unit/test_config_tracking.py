"""設定與追蹤：離線可完成、不洩漏金鑰。"""
import os
import pandas as pd
from Research.StyleGAN.utils.config import make_config, save_config, load_config
from Research.StyleGAN.utils.tracking import Tracker


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
    assert list(df.columns)[:5] == ["step", "block", "fade_alpha", "loss_g", "loss_d"] and len(df) == 3
    for p in tmp_path.rglob("*"):
        if p.is_file() and p.suffix in {".csv", ".json", ".yaml", ".txt", ".log"}:
            assert "SECRET-KEY-123" not in p.read_text(errors="ignore")


def test_old_config_loads_as_legacy(tmp_path):
    """舊 run（config.json 沒有 encoding 欄位）可載入，且視為 legacy（002 T007）。"""
    import json
    c = make_config(1)
    save_config(c, tmp_path / "c.json")
    d = json.loads((tmp_path / "c.json").read_text()); d.pop("encoding")
    (tmp_path / "old.json").write_text(json.dumps(d))
    assert load_config(tmp_path / "old.json").enc.name == "legacy"


def test_run_dir_encoding_suffix():
    """legacy 的路徑與 run01 同格式；其他編碼加後綴。"""
    assert make_config(1).run_dir("R").as_posix() == "R/A/scale1/classical_logistic_seed0"
    assert make_config(1, encoding="drop_const").run_dir("R").as_posix() == "R/A/scale1/classical_logistic_drop_const_seed0"
    c = make_config(1, encoding="drop_const+nonorm")
    assert c.enc.drop_const_threshold == 0.9 and c.enc.normalize == "none" and c.enc.output_norm == "sigmoid"


def test_tracker_diversity_columns(tmp_path, monkeypatch):
    """Tracker.log 帶多樣性欄位時 losses.csv 欄位完整；舊呼叫（無額外欄位）留空（T029）。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    t = Tracker(tmp_path, {"a": 1}, mode="off")
    t.log(0, 0, 1.0, 1.0, 2.0)
    t.log(74, 0, 1.0, 1.0, 2.0, distinct_sets=5, pos_coverage=7, freq_corr=0.5)
    t.close()
    df = pd.read_csv(tmp_path / "losses.csv")
    assert list(df.columns)[:8] == ["step", "block", "fade_alpha", "loss_g", "loss_d", "distinct_sets", "pos_coverage", "freq_corr"]
    assert list(df.columns)[8:] == ["live_distinct_sets", "live_pos_coverage", "live_freq_corr"]   # 003：即時權重欄位
    assert df.distinct_sets.isna().tolist() == [True, False] and df.pos_coverage.iloc[1] == 7


def test_recipe_defaults_keep_run02_behavior():
    """003：新因子預設值不改變既有行為——區塊排程、總步數、run 目錄都與 002 相同。"""
    c = make_config(1, encoding="drop_const")
    assert (c.lr_scale, c.d_every, c.last_block_steps, c.fade_frac, c.tag) == (1.0, 1, None, 0.5, "")
    assert c.block_schedule == [(0, 1167), (1167, 2333), (2333, 3500)] and c.total_steps == 3500
    assert c.run_dir("R").as_posix() == "R/A/scale1/classical_logistic_drop_const_seed0"


def test_last_block_steps_and_tag():
    """003：last_block_steps 只改最後區塊（其餘區塊不動），總步數隨之改變；tag 加在目錄名稱。"""
    c = make_config(1, encoding="drop_const", last_block_steps=600, tag="last600")
    assert c.block_schedule == [(0, 1167), (1167, 2333), (2333, 2933)] and c.total_steps == 2933
    assert c.run_dir("R").name == "classical_logistic_drop_const_last600_seed0"
    assert make_config(1, last_block_steps=2400).total_steps == 2333 + 2400


def test_old_config_without_recipe_fields_loads(tmp_path):
    """003：舊 run（config.json 沒有新因子欄位）可載入並取預設值。"""
    import json
    c = make_config(1)
    save_config(c, tmp_path / "c.json")
    d = json.loads((tmp_path / "c.json").read_text())
    for k in ("lr_scale", "d_every", "last_block_steps", "fade_frac", "tag"):
        d.pop(k)
    (tmp_path / "old.json").write_text(json.dumps(d))
    o = load_config(tmp_path / "old.json")
    assert o.d_every == 1 and o.last_block_steps is None and o.fade_frac == 0.5


def test_model_factor_defaults_and_old_config(tmp_path):
    """004：模型與損失因子預設值＝003 設定；舊 config（沒有新欄位）可載入；None 表示同 fmap_max。"""
    import json
    c = make_config(1)
    assert (c.g_fmap_max, c.d_fmap_max, c.mbstd_group, c.div_reg) == (None, None, 4, 0.0)
    assert (c.g_fmap_max or c.fmap_max) == c.fmap_max == 512
    save_config(c, tmp_path / "c.json")
    d = json.loads((tmp_path / "c.json").read_text())
    for k in ("g_fmap_max", "d_fmap_max", "mbstd_group", "div_reg"):
        d.pop(k)
    (tmp_path / "old.json").write_text(json.dumps(d))
    o = load_config(tmp_path / "old.json")
    assert o.mbstd_group == 4 and o.div_reg == 0.0 and o.g_fmap_max is None
