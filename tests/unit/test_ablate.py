"""消融彙總與判定（002 T034）：用合成指標表驗證，不需訓練。"""
import pandas as pd
from Research.StyleGAN.evaluate.ablate import BASE, judge, plan_round2, success_table, summarize, write_report


def _df(spec):
    """spec: {編碼: [(相異突變集, 覆蓋), ...每個種子]} → 彙總用的 DataFrame。"""
    rows = []
    for enc, seeds in spec.items():
        for i, (d, c) in enumerate(seeds):
            rows.append(dict(round=1, encoding=enc, seed=i, distinct_mutation_sets=d, position_coverage=c,
                             freq_correlation=0.8, random_freq_correlation=0.0, mean_mutation_count=1.0,
                             train_mean_count=1.0, min_hamming_model=2.0 + 0.1 * i, min_hamming_freq_baseline=3.0, min_hamming_test_k_diagnostic=7.0,
                             loss_g_final=1.0, collapse_warning=d < 10))
    return pd.DataFrame(rows)


def test_summarize_mean_min_max():
    s = summarize(_df({BASE: [(2, 4), (4, 6), (3, 5)]}))
    assert s.loc[BASE, "distinct_mutation_sets_mean"] == 3 and s.loc[BASE, "distinct_mutation_sets_min"] == 2
    assert s.loc[BASE, "position_coverage_max"] == 6 and s.loc[BASE, "n_seeds"] == 3


def test_judge_effective_ineffective_worse():
    df = _df({BASE: [(3, 4), (5, 6), (4, 5)],              # legacy 最大 5 / 覆蓋最大 6
              "good": [(60, 20), (80, 25), (70, 22)],      # 平均 70 ≥ 2×5 且覆蓋平均 > 6 → 有效
              "same": [(4, 5), (6, 6), (5, 5)],            # 差不多 → 無效
              "bad": [(1, 2), (1, 1), (2, 2)]})            # 低於 legacy 最小 → 變差
    v = judge(summarize(df))
    assert v == {"good": "有效", "same": "無效", "bad": "變差"}


def test_effective_needs_mean_not_one_lucky_seed():
    df = _df({BASE: [(3, 4), (5, 6), (4, 5)], "lucky": [(12, 10), (2, 3), (3, 4)]})
    assert judge(summarize(df))["lucky"] != "有效"


def test_plan_round2_only_combines_effective():
    assert plan_round2({"a": "有效", "b": "無效", "c": "有效"}) == "a+c"
    assert plan_round2({"a": "有效", "b": "無效"}) is None       # 只有 1 個有效：沒有可組合的
    assert plan_round2({"a": "無效", "b": "變差"}) is None


def test_report_conclusion_when_nothing_effective(tmp_path):
    """沒有任何有效編碼：報告寫出 SC-007 的結論句（encoding 並非唯一主因）。"""
    df = _df({BASE: [(3, 4), (5, 6), (4, 5)], "x": [(4, 5), (6, 6), (5, 5)]})
    write_report(df, tmp_path, 1)
    rep = (tmp_path / "ablation_report.md").read_text()
    assert "encoding 並非唯一主因" in rep and "無效" in rep
    assert list(pd.read_csv(tmp_path / "ablation_summary.csv").columns)[:3] == ["round", "encoding", "seed"]


def test_success_table_uses_thresholds():
    df = _df({BASE: [(3, 4)] * 3, "g": [(120, 30), (150, 31), (110, 29)], "weak": [(50, 30), (60, 30), (40, 30)]})
    t = success_table(df).set_index("encoding")
    assert t.loc["g", "SC002"] and not t.loc["weak", "SC002"]      # weak 平均 <100
