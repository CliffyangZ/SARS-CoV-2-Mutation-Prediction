"""過濾規則與去重。"""
from Research.FE import load_spike as ls


def test_filter_and_dedupe(synth_fasta):
    df = ls.read_fasta(synth_fasta)
    f = ls.quality_filter(df, length=20)
    assert set(f.acc) == {"REF.1", "A1", "A2", "A3"}     # 過短/壞日期/終止碼/過多X 皆被剔除
    u = ls.dedupe(f)
    assert len(u) == 3
    row = u[u["count"] == 2].iloc[0]
    assert row.first_date.strftime("%Y-%m-%d") == "2020-03-01" and row.last_date.strftime("%Y-%m-%d") == "2020-05-10"
    assert row.n_country == 2
