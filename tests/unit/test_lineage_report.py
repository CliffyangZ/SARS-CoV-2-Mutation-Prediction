"""彙整報告的單元測試：平均±標準差、決定性標示、未完成分流、缺論文值（FR-009、SC-004）。"""
import numpy as np

from Research.lineage_baselines import report as RP

METRICS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "roc_auc_ovr"]


def _res(model, rep, seed, acc, status="complete", det=False, worst=("L2", 0.1)):
    """人工構造的結果。"""
    m = {k: acc for k in METRICS}
    return {"model": model, "representation": rep, "seed": seed, "status": status, "deterministic": det,
            "metrics": m, "train_seconds": 10.0, "epochs_done": 100 if status == "complete" else 40,
            "hyperparameters": {"pretrained_imagenet": False},
            "per_class": [{"lineage": "L1", "f1": 0.9, "support": 10}, {"lineage": worst[0], "f1": worst[1], "support": 5}]}


PAPER = {"rows": [
    {"model": "rf", "representation": "ae_embedding", "label": "AE+NTK / RF", **{k: 0.5 for k in METRICS}, "train_time_printed": "0.1 Sec.", "label_inferred": False},
    {"model": "dt", "representation": "ae_embedding", "label": "AE+NTK / DT", **{k: 0.4 for k in METRICS}, "train_time_printed": "0.1 Sec.", "label_inferred": False},
    {"model": "cnn3", "representation": "rp", "label": "RP / 3 Layer CNN", **{k: 0.78 for k in METRICS}, "train_time_printed": "1 Hours", "label_inferred": False},
]}


def test_aggregate_mean_std():
    """多種子的平均與樣本標準差（ddof=1）計算正確。"""
    rs = [_res("rf", "ae_embedding", s, a) for s, a in enumerate([0.5, 0.6, 0.7])]
    agg = RP.aggregate(rs)[("rf", "ae_embedding")]
    assert agg["n_seeds"] == 3 and np.isclose(agg["mean"]["accuracy"], 0.6) and np.isclose(agg["std"]["accuracy"], 0.1)


def test_deterministic_has_no_std():
    """決定性模型只有一筆時標為決定性，不報標準差。"""
    agg = RP.aggregate([_res("knn", "onehot", 0, 0.5, det=True)])[("knn", "onehot")]
    assert agg["deterministic"] is True and agg["std"] is None


def test_incomplete_separated():
    """未完成的結果不混入平均，並列在獨立區段。"""
    rs = [_res("cnn3", "rp", 0, 0.8), _res("cnn3", "rp", 1, 0.0, status="incomplete")]
    agg = RP.aggregate(rs)
    assert agg[("cnn3", "rp")]["n_seeds"] == 1
    text = RP.build_report(rs, PAPER)
    assert "未完成" in text and "cnn3" in text


def test_missing_paper_value_marked():
    """論文沒有對應列的模型，顯示「論文未提供」。"""
    text = RP.build_report([_res("mtf_model", "mtf", 0, 0.5)], PAPER)
    assert "論文未提供" in text


def test_worst_lineage_reported():
    """報告指出各模型最差的譜系。"""
    text = RP.build_report([_res("rf", "ae_embedding", 0, 0.5, worst=("AY.9", 0.05))], PAPER)
    assert "AY.9" in text


def test_ranking_comparison_present():
    """共同的模型對照出排名一致性。"""
    rs = [_res("rf", "ae_embedding", 0, 0.5), _res("dt", "ae_embedding", 0, 0.3), _res("cnn3", "rp", 0, 0.8)]
    text = RP.build_report(rs, PAPER)
    assert "排名" in text and "Spearman" in text
