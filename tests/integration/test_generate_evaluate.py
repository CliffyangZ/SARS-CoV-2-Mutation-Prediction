"""以小型合成資料與小模型跑 generate + evaluate，檢查輸出欄位符合 contracts/artifacts.md。"""
import json
import numpy as np
import pandas as pd
from Research.train import train_classical as tc
from Research.train.evaluate import evaluate
from Research.train.generate import generate
from Research.train.loop import train
from Research.utils.config import make_config


def test_pipeline(tmp_path, synth_dataset, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=30, model="classical", batch_size=4, channels=[4, 4, 4])
    G, D = tc.build_models(cfg)
    X = np.load(synth_dataset / "vectors.npz")["X"][:60]
    run = tmp_path / "run"
    assert train(cfg, G, D, X, run, "off")["state"] == "finished"
    generate(run, n=20, data_dir=synth_dataset)
    m = evaluate(run, data_dir=synth_dataset)
    g = np.load(run / "generated.npz")
    assert {"diag", "mut_index", "mut_spike_pos", "mapped_seq", "k", "seed"} <= set(g.files)
    assert g["diag"].shape == (20, 64) and g["mut_index"].shape == (20, cfg.k) and len(g["mapped_seq"][0]) == 1273
    ex = pd.read_csv(run / "explanations.csv")
    assert list(ex.columns) == ["gen_id", "index", "spike_pos", "region", "train_freq", "trend_slope", "is_novel", "count_matches_k"]
    assert len(ex) == 20 * cfg.k
    j = json.loads((run / "metrics.json").read_text())
    assert {"fidelity", "prediction", "diversity"} == set(j)
    assert {"reference_set", "formula", "source", "interpretation", "min", "mean", "max", "paper_threshold", "passed"} <= set(j["fidelity"][0])
    assert {r["source"] for r in j["prediction"]} == {"model", "reference", "random", "freq", "trend"}
    assert {"distinct_sequences", "position_coverage"} <= set(j["diversity"])
    assert {r["reference_set"] for r in j["fidelity"]} == {"train", "test", "test_delta_like"}
