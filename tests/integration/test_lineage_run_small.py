"""傳統 baseline 的小規模整合測試：結果檔完整、無洩漏、決定性（US1）。"""
import json

import numpy as np
import pandas as pd
import pytest

from Research.lineage_baselines import autoencoder as AE
from Research.lineage_baselines import data as D
from Research.lineage_baselines import run as RUN

AA = "ACDEFGHIKLMNPQRSTVWY"


def _synthetic_tsv(path, n_per=40, length=60, seed=0):
    """三個譜系，各自帶有不同的固定 motif，並有少量 X；寫成 dataset.tsv 格式。"""
    rng = np.random.default_rng(seed)
    rows = []
    for k, lin in enumerate(["L1", "L2", "L3"]):
        for i in range(n_per):
            s = list(rng.choice(list(AA), size=length))
            s[5:10] = list("AAAAA") if k == 0 else list("CCCCC") if k == 1 else list("DDDDD")
            if i % 7 == 0:
                s[30] = "X"
            rows.append(dict(lineage=lin, accession=f"{lin}_{i}", date="2021", count=1,
                             split="train" if i < int(n_per * 0.7) else "test", sequence="".join(s)))
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture
def tsv(tmp_path):
    """合成資料檔。"""
    p = tmp_path / "dataset.tsv"
    _synthetic_tsv(p)
    return p


def _run(tsv, out, *extra):
    """以小參數執行 run.main。"""
    argv = ["--data", str(tsv), "--out", str(out), "--ae-epochs", "2", *extra]
    return RUN.main(argv)


def test_all_seven_classical_produce_complete_results(tsv, tmp_path):
    """七個分類器各產生一個結果檔，指標與必要欄位齊全。"""
    out = tmp_path / "res"
    assert _run(tsv, out, "--group", "classical", "--representation", "onehot", "--seeds", "0") == 0
    files = sorted(out.glob("*.json"))
    assert len(files) == 7
    for f in files:
        r = json.loads(f.read_text())
        assert r["status"] == "complete"
        for k in ("accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr", "precision_weighted", "recall_weighted"):
            assert np.isfinite(r["metrics"][k])
        assert r["train_seconds"] >= 0 and r["provenance"] and r["preprocessing"]["x_symbol"] is True
        assert len(r["per_class"]) == 3


def test_ae_embedding_path_runs(tsv, tmp_path):
    """自編碼器嵌入路徑可跑通，並記錄自編碼器超參數與來源。"""
    out = tmp_path / "res"
    assert _run(tsv, out, "--model", "rf", "--representation", "ae_embedding", "--seeds", "0") == 0
    r = json.loads(next(out.glob("rf__ae_embedding__seed0.json")).read_text())
    assert r["hyperparameters"]["autoencoder"]["encoder_dims"] == [1024, 512, 256, 128]
    assert r["provenance"]["autoencoder"]["epochs"] == "paper"


def test_deterministic_models_run_once(tsv, tmp_path):
    """KNN 與 Naive Bayes 是決定性模型，多個種子只產生一個結果檔並標示。"""
    out = tmp_path / "res"
    _run(tsv, out, "--model", "knn", "--representation", "onehot", "--seeds", "0", "1", "2")
    files = list(out.glob("knn__*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["deterministic"] is True


def test_complete_results_are_skipped(tsv, tmp_path, capsys):
    """已完成的結果檔不會重算。"""
    out = tmp_path / "res"
    args = ("--model", "dt", "--representation", "onehot", "--seeds", "0")
    _run(tsv, out, *args)
    capsys.readouterr()
    _run(tsv, out, *args)
    assert "跳過" in capsys.readouterr().out


def test_same_seed_same_result(tsv, tmp_path):
    """相同種子重跑，隨機模型的指標完全一致。"""
    a, b = tmp_path / "a", tmp_path / "b"
    for o in (a, b):
        _run(tsv, o, "--model", "rf", "--representation", "onehot", "--seeds", "3")
    ra = json.loads(next(a.glob("*.json")).read_text())["metrics"]
    rb = json.loads(next(b.glob("*.json")).read_text())["metrics"]
    assert ra == rb


def test_no_leakage_from_test_into_train_artifacts(tsv, tmp_path):
    """只改動 test 序列：共同長度 L、train 編碼與 train 嵌入完全不變。"""
    df = pd.read_csv(tsv, sep="\t")
    df2 = df.copy()
    te = df2["split"] == "test"
    df2.loc[te, "sequence"] = df2.loc[te, "sequence"].str.replace("A", "W")
    p1, p2 = D.prepare_from_frame(df), D.prepare_from_frame(df2)
    assert p1.length == p2.length and np.array_equal(p1.x_train, p2.x_train)
    e1, _, _ = AE.fit_embed(p1.x_train, p1.x_test, seed=0, epochs=2, device="cpu")
    e2, _, _ = AE.fit_embed(p2.x_train, p2.x_test, seed=0, epochs=2, device="cpu")
    assert np.allclose(e1, e2, atol=1e-6)


def test_missing_data_file_nonzero_exit(tmp_path):
    """資料檔不存在時以非零碼結束。"""
    assert RUN.main(["--data", str(tmp_path / "none.tsv"), "--model", "dt", "--out", str(tmp_path)]) != 0


def test_determinism_depends_on_representation(tsv, tmp_path):
    """ae_embedding 的嵌入隨種子變，因此 KNN 在 ae_embedding 下要跑多個種子；SVM 在 onehot 下只跑一次。"""
    assert RUN.is_deterministic("svm", "onehot") and RUN.is_deterministic("knn", "onehot")
    assert not RUN.is_deterministic("knn", "ae_embedding") and not RUN.is_deterministic("rf", "onehot")
    out = tmp_path / "res"
    _run(tsv, out, "--model", "knn", "--representation", "ae_embedding", "--seeds", "0", "1")
    assert len(list(out.glob("knn__ae_embedding__*.json"))) == 2
