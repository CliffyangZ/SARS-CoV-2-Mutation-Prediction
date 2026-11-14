"""訓練設定消融（003）：用合成指標表與合成軌跡驗證判定與選擇規則，不需訓練。"""
import pandas as pd
import pytest
from Research.StyleGAN.evaluate import ablate_recipe as ar


def _df(spec, hams=None):
    """spec: {(factor, value): [相異突變集…每個種子]}；hams 給各鍵的 min_hamming_model（預設 3.0）。"""
    rows = []
    for (f, v), ds in spec.items():
        for i, d in enumerate(ds):
            rows.append(dict(round=1, encoding="drop_const", seed=i, distinct_mutation_sets=d, position_coverage=20 if d >= 50 else 5,
                             freq_correlation=0.3, random_freq_correlation=0.0, mean_mutation_count=1.0, train_mean_count=1.0,
                             min_hamming_model=(hams or {}).get((f, v), 3.0) + 0.01 * i, min_hamming_test_k_diagnostic=7.0,
                             min_hamming_freq_baseline=3.5, loss_g_final=1.0, collapse_warning=d < 10,
                             factor=f, value=str(v), seconds=500.0))
    return pd.DataFrame(rows)


BASE = _df({("baseline", "-"): [50, 55, 60]})


def test_factor_table_tags_unique_and_cover_six_factors():
    tags = [t for vals in ar.FACTORS.values() for t, _, _ in vals]
    assert len(ar.FACTORS) == 6 and len(tags) == 12 and len(set(tags)) == 12
    assert all(len(v) == 2 for v in ar.FACTORS.values())


def test_args_for_roundtrip():
    for f, vals in ar.FACTORS.items():
        for tag, args, v in vals:
            assert ar.args_for(f, str(v)) == (tag, args)
    with pytest.raises(KeyError):
        ar.args_for("gamma1", "999")


def test_run_dir_tag_matches_config_run_dir():
    from Research.StyleGAN.utils.config import make_config
    c = make_config(1, encoding="drop_const", tag="dev2", seed=1)
    assert ar.run_dir_for_recipe("R", "drop_const", "dev2", 1) == c.run_dir("R")


def test_judge_recipe_three_outcomes_and_hamming_guard():
    df = _df({("a", 1): [100, 110, 120],               # 平均 110 ≥ 1.5×55，預測相同 → 有效
              ("b", 1): [60, 55, 58],                   # 沒有 1.5 倍 → 無效
              ("c", 1): [3, 4, 5],                      # 低於基準最小且覆蓋低 → 變差
              ("d", 1): [100, 110, 120]},               # 多樣性夠但預測明顯變差 → 不是有效
             hams={("d", 1): 5.0})
    v = ar.judge_recipe(df, BASE)
    assert v[("a", "1")] == "有效" and v[("b", "1")] == "無效" and v[("c", "1")] == "變差" and v[("d", "1")] != "有效"


def test_select_best_filter_then_hamming():
    df = _df({("a", 1): [120, 130, 125], ("b", 1): [110, 105, 130], ("c", 1): [300, 20, 400]},   # c 有種子 <30 → 不通過
             hams={("a", 1): 3.2, ("b", 1): 2.9, ("c", 1): 1.0})
    assert ar.select_best(df) == ("b", "1", True)           # a、b 通過篩選，取 Hamming 較低的 b；c 的 Hamming 雖低但被篩掉


def test_select_best_fallback_when_nobody_passes():
    df = _df({("a", 1): [40, 45, 42], ("b", 1): [30, 31, 29]})
    assert ar.select_best(df) == ("a", "1", False)          # 無人通過：取相異突變集最高並標示未達標
    assert ar.select_best(df.iloc[0:0]) is None


def test_combo_plan_needs_two_effective_factors():
    df = _df({("a", 1): [100, 100, 100], ("a", 2): [200, 200, 200], ("b", 1): [100, 100, 100], ("c", 1): [50, 50, 50]})
    v = {("a", "1"): "有效", ("a", "2"): "有效", ("b", "1"): "有效", ("c", "1"): "無效"}
    assert ar.combo_plan(df, v) == {"a": "2", "b": "1"}      # 每因子取較佳的有效值
    assert ar.combo_plan(df, {("a", "1"): "有效", ("b", "1"): "無效"}) is None


def _traj(peak, final, n=12):
    """合成最後區塊軌跡：前段（未訓練區間）高，之後在 peak 附近，最後降到 final。"""
    vals = [200, 200] + [peak] * (n - 3) + [final]
    return pd.DataFrame(dict(step=range(n), block=2, distinct_sets=vals, live_distinct_sets=vals))


def test_us1_verdict_real_needs_two_of_three_seeds():
    real = {0: _traj(100, 20), 1: _traj(100, 30), 2: _traj(100, 90)}
    assert ar.us1_verdict(real)["ema"] == "real" and ar.us1_verdict(real)["seeds_real"]["ema"] == 2
    art = {0: _traj(100, 20), 1: _traj(100, 80), 2: _traj(100, 90)}
    assert ar.us1_verdict(art)["ema"] == "artifact"


def test_us1_verdict_excludes_untrained_initial_segment():
    """前 10% 的檢查點（未訓練）值很高，不得被當作『最高值』；峰值取自其後。"""
    d = pd.DataFrame(dict(step=range(20), block=2, distinct_sets=[500, 500] + [40] * 17 + [30], live_distinct_sets=[500, 500] + [40] * 17 + [30]))
    assert ar.us1_verdict({0: d, 1: d, 2: d})["ema"] == "artifact"      # 40→30 只降 25%，若把 500 算進去就會誤判為 real


def test_us1_verdict_live_and_ema_can_differ():
    d_ema = _traj(100, 90)
    d = d_ema.assign(live_distinct_sets=_traj(100, 20).live_distinct_sets)
    v = ar.us1_verdict({0: d, 1: d, 2: d})
    assert v["ema"] == "artifact" and v["live"] == "real"


def test_report_marks_unreached_and_sc007(tmp_path):
    df = _df({("a", 1): [40, 45, 42]})
    best = ar.select_best(df)
    ar.write_report_recipe(df, BASE, tmp_path, "round1", best)
    t = (tmp_path / "ablation_report.md").read_text()
    assert "未達標" in t and "SC-007" in t and (tmp_path / "ablation_summary.csv").exists()


def test_main_refuses_run02(tmp_path):
    with pytest.raises(SystemExit):
        ar.main(["--stage", "us1", "--results", ar.BASE_RESULTS])


def test_confirm_without_best_recipe_gives_clear_error(tmp_path):
    """T028：缺少 US2 的 best_recipe.json 時，confirm 以清楚訊息結束，不是 FileNotFoundError。"""
    with pytest.raises(SystemExit) as e:
        ar.main(["--stage", "confirm", "--results", str(tmp_path)])
    assert "best_recipe.json" in str(e.value)
