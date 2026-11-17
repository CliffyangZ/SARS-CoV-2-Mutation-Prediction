"""修訂版報告（§4.2 增量判定、§5 分群與差距）的單元測試。"""
import json

from Research.QuSpike_JEPA import summary as S


def _write(tmp, name, seed, f1, cfg, per_class=None, **extra):
    r = dict(config=dict(cfg, name=name), seed=seed, status="complete",
             metrics_test={"f1_macro": f1}, train_seconds=1.0, ft_seconds=1.0, **extra)
    if per_class:
        r["per_class_test"] = per_class
    (tmp / f"{name}__mut__seed{seed}.json").write_text(json.dumps(r), encoding="utf-8")


def _cfg(**kw):
    return {"channels": "q", "finetune": True, "epochs": 10, **kw}


def test_incomplete_and_collapsed_probe_results_are_excluded(tmp_path):
    """未完成、以及「非 fine-tune 且表徵塌陷」的結果不進平均（後者分數本身失效）。"""
    _write(tmp_path, "ok", 0, 0.70, _cfg())
    _write(tmp_path, "bad_probe", 0, 0.99, {"finetune": False}, collapsed=True)
    (tmp_path / "part__mut__seed0.json").write_text(json.dumps(
        dict(config=dict(name="part"), status="skipped", metrics_test={"f1_macro": 0.9})), encoding="utf-8")
    g = S.collect(tmp_path)
    assert set(g) == {"ok"}


def test_mean_and_std_over_seeds(tmp_path):
    """多種子取平均與樣本標準差。"""
    for s, v in enumerate([0.70, 0.72, 0.74]):
        _write(tmp_path, "cfgA", s, v, _cfg())
    g = S.collect(tmp_path)
    m, sd = S._ms(g["cfgA"]["vals"])
    assert abs(m - 0.72) < 1e-9 and abs(sd - 0.02) < 1e-9
    assert S.fmt((m, sd)) == "0.720±0.020"


def test_verdict_uses_noise_floor():
    """小於雜訊下限的差異判為雜訊內，即使標準差為 0。"""
    d, v = S.verdict((0.71, 0.0), (0.70, 0.0))
    assert abs(d - 0.01) < 1e-9 and "雜訊內" in v
    d, v = S.verdict((0.80, 0.0), (0.70, 0.0))
    assert v == "較佳"
    d, v = S.verdict((0.60, 0.0), (0.70, 0.0))
    assert v == "較差"


def test_verdict_respects_seed_std():
    """差異雖大於雜訊下限，但未超過種子標準差時仍判雜訊內。"""
    _, v = S.verdict((0.75, 0.10), (0.70, 0.02))
    assert "雜訊內" in v


def test_increments_table_reports_both_hypotheses(tmp_path):
    """同時產出 H4′（預訓練貢獻）與 H3′（跳接增量）兩列。"""
    for s in range(3):
        _write(tmp_path, "pre_both", s, 0.80, _cfg(channels="both", epochs=10))
        _write(tmp_path, "scratch_both", s, 0.70, _cfg(channels="both", epochs=0))
        _write(tmp_path, "ref", s, 0.747, dict(channels="skip", finetune=True, epochs=0))
    txt = "\n".join(S.table_increments(S.collect(tmp_path)))
    assert "H4′ 預訓練貢獻（all／both）" in txt and "+0.100" in txt
    assert "H3′ 跳接增量" in txt


def test_main_table_shows_gaps_to_reference_and_ceiling(tmp_path):
    """主表同時給出距參考線與距天花板的差距。"""
    _write(tmp_path, "cfgA", 0, 0.80, _cfg())
    txt = "\n".join(S.table_main(S.collect(tmp_path), reference=0.747, ceiling=0.930))
    assert "+0.053" in txt and "-0.130" in txt


def test_group_table_breaks_down_by_lineage_group(tmp_path):
    """分群表按 delta_parent／delta_sub／b1_family／others 呈現。"""
    pc = [{"lineage": "B.1.617.2", "f1": 0.5, "precision": 0, "recall": 0, "support": 1},
          {"lineage": "AY.44", "f1": 0.4, "precision": 0, "recall": 0, "support": 1},
          {"lineage": "P.1", "f1": 1.0, "precision": 0, "recall": 0, "support": 1}]
    _write(tmp_path, "cfgA", 0, 0.8, _cfg(), per_class=pc)
    txt = "\n".join(S.table_groups(S.collect(tmp_path)))
    assert "delta_parent" in txt and "0.500" in txt and "0.400" in txt


def test_build_runs_with_no_results(tmp_path):
    """空目錄也能產生報告，不拋例外。"""
    txt = S.build(tmp_path, reference=0.747, ceiling=0.930)
    assert "QuSpike-JEPA 修訂版結果" in txt and "缺少可比較的設定" in txt


def test_collapsed_finetuned_results_are_kept(tmp_path):
    """表徵塌陷不應丟棄 fine-tune 的任務分數（fine-tune 會重訓編碼器），但會丟棄凍結 probe 的。"""
    _write(tmp_path, "ft_collapsed", 0, 0.73, _cfg(finetune=True), collapsed=True)
    _write(tmp_path, "probe_collapsed", 0, 0.11, {"finetune": False}, collapsed=True)
    g = S.collect(tmp_path)
    assert set(g) == {"ft_collapsed"} and g["ft_collapsed"]["collapsed"] == [True]


def test_main_table_shows_collapse_count(tmp_path):
    """主表標示該設定有幾個種子的表徵塌陷。"""
    _write(tmp_path, "cfgA", 0, 0.73, _cfg(finetune=True), collapsed=True)
    _write(tmp_path, "cfgA", 1, 0.74, _cfg(finetune=True))
    txt = "\n".join(S.table_main(S.collect(tmp_path)))
    assert "1/2" in txt
