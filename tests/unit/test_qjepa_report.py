"""報告：標準差判斷、塌陷不入平均。"""
from Research.QuSpike_JEPA import report as R


def _r(name, f1, collapsed=False, seed=0):
    return dict(config=dict(name=name), status="complete", collapsed=collapsed, seed=seed,
                metrics_val=dict(f1_macro=f1, accuracy=f1, f1_weighted=f1, precision_weighted=f1, recall_weighted=f1, roc_auc_ovr=f1),
                train_seconds=1.0, effective_rank=5.0)


def test_collapsed_excluded_from_mean():
    g = R.group([_r("a", 0.5), _r("a", 0.7, seed=1), _r("a", 0.0, collapsed=True, seed=2)])
    assert len(g["a"]["ok"]) == 2 and len(g["a"]["collapsed"]) == 1
    assert abs(R.agg(g["a"]["ok"], "val")[0] - 0.6) < 1e-9


def test_differs_uses_std():
    assert not R.differs((0.5, 0.1), (0.55, 0.05))
    assert R.differs((0.5, 0.01), (0.6, 0.01))


def test_report_runs(tmp_path):
    import json
    for s in range(2):
        (tmp_path / f"jepa_VA_classical__mut__seed{s}.json").write_text(json.dumps(_r("jepa_VA_classical", 0.5 + 0.1 * s, seed=s)))
    assert R.main(["--results", str(tmp_path), "--baselines", str(tmp_path / "none")]) == 0
    assert "jepa_VA_classical" in (tmp_path / "report.md").read_text()
