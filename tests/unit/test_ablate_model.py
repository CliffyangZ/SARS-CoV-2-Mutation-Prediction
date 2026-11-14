"""模型與損失消融（004）：因子表、模式、判定、失真檢查、雙門檻、組合與最佳設定。"""
import json

import pandas as pd
import pytest

from Research.StyleGAN.evaluate import ablate_model as am


def _rows(factor, value, distinct, mean_cnt=2.0, ham=3.0, cov=20):
    """合成 3 個種子的消融列。"""
    return pd.DataFrame([dict(factor=factor, value=str(value), seed=s, distinct_mutation_sets=d, position_coverage=cov,
                              freq_correlation=0.8, random_freq_correlation=0.1, mean_mutation_count=mean_cnt,
                              train_mean_count=2.0, min_hamming_model=ham, min_hamming_freq_baseline=3.5,
                              seconds=500.0) for s, d in enumerate(distinct)])


BASE = _rows("baseline", "-", [50, 55, 60], cov=18)


def test_factor_table_modes():
    """縮減模式每因子 1 值；完整模式 noise 以外都是 2 值；因子共 7 個。"""
    assert len(am.FACTORS) == 7 and set(am.STRUCTURE + am.LOSS) == set(am.FACTORS)
    assert all(len(am.factor_values(f, "shrunk")) == 1 for f in am.FACTORS)
    assert {f: len(am.factor_values(f, "full")) for f in am.FACTORS}["noise"] == 1
    assert sum(len(am.factor_values(f, "full")) for f in am.FACTORS) == 13
    assert len(am.plan_round1("shrunk", [0, 1, 2])) == 21 and len(am.plan_round1("full", [0, 1, 2])) == 39


def test_args_for():
    """命令列參數對應；未知值報錯。"""
    assert am.args_for("div_reg", "1.0") == ("div1", ["--div-reg", "1.0"])
    assert am.args_for("noise", "off")[1] == ["--no-noise"]
    with pytest.raises(KeyError):
        am.args_for("div_reg", "9")


def test_mode_from_ceiling(tmp_path):
    """too_high → shrunk＋修訂門檻；其他 → full＋100；缺檔時報錯並提示先跑上限分析。"""
    with pytest.raises(SystemExit):
        am.mode_from_ceiling(tmp_path)
    (tmp_path / "ceiling").mkdir()
    (tmp_path / "ceiling" / "ceiling.json").write_text(json.dumps({"verdict": "too_high", "revised_target": 42}))
    assert am.mode_from_ceiling(tmp_path) == ("shrunk", 42)
    (tmp_path / "ceiling" / "ceiling.json").write_text(json.dumps({"verdict": "undetermined", "revised_target": 100}))
    assert am.mode_from_ceiling(tmp_path) == ("full", 100)


def test_refuses_old_results():
    """結果目錄是 run02／run03 時拒絕（舊結果不可覆寫）。"""
    for old in ("run02", "run03"):
        with pytest.raises(SystemExit):
            am.main(["--stage", "round1", "--results", f"Results/StyleGAN/{old}"])


def test_distortion_downgrades_effective():
    """相異集夠多但平均突變數偏離訓練期 >1 → 不算有效。"""
    ok = _rows("div_reg", "1.0", [100, 110, 120], mean_cnt=2.2)
    bad = _rows("div_reg", "0.3", [100, 110, 120], mean_cnt=4.5)
    v = am.judge_model(pd.concat([ok, bad]), BASE)
    assert v[("div_reg", "1.0")] == "有效"
    assert v[("div_reg", "0.3")] == "無效（以失真換多樣性）"


def test_pass_columns_dual_threshold():
    """舊新門檻各自判定；種子下限 30 仍適用。"""
    d = pd.concat([_rows("a", "1", [60, 65, 70]), _rows("b", "1", [150, 20, 160])])
    pc = am.pass_columns(d, 60).set_index("factor")
    assert bool(pc.loc["a", "pass_new"]) and not bool(pc.loc["a", "pass_old"])
    assert not bool(pc.loc["b", "pass_new"])      # 有種子 <30


def test_combo_and_select_best():
    """有效因子 <2 時不組合；選最佳：先篩再取最低 Hamming，無人通過取相異集最高並標未達標。"""
    from Research.StyleGAN.evaluate import ablate_recipe as ar
    a, b = _rows("mapping_layers", "2", [120, 130, 140], ham=3.0), _rows("g_fmap_max", "256", [110, 115, 120], ham=2.9)
    df = pd.concat([a, b])
    v = am.judge_model(df, BASE)
    assert ar.combo_plan(df, v) == {"mapping_layers": "2", "g_fmap_max": "256"}
    assert ar.combo_plan(a, am.judge_model(a, BASE)) is None
    assert ar.select_best(df, 100) == ("g_fmap_max", "256", True)
    low = pd.concat([_rows("x", "1", [40, 45, 50]), _rows("y", "1", [60, 62, 64])])
    assert ar.select_best(low, 100) == ("y", "1", False)
    assert ar.select_best(low, 55)[:2] == ("y", "1")      # 修訂門檻 55 時 y 通過


def test_report_has_dual_threshold_section(tmp_path):
    """報告含失真檢查與新舊門檻表。"""
    df = _rows("div_reg", "1.0", [70, 75, 80])
    am.write_report_model(df, BASE, tmp_path, "round1", 70)
    t = (tmp_path / "ablation_report.md").read_text()
    assert "模型與損失消融報告" in t and "失真檢查與新舊門檻" in t and "div_reg" in t
