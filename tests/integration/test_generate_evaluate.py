"""以小型合成資料與小模型跑 generate + evaluate，檢查輸出欄位符合 contracts/artifacts.md。"""
import json
import numpy as np
import pandas as pd
from Research.StyleGAN.train import train_StyleGAN as tc
from Research.StyleGAN.evaluate.evaluate import evaluate
from Research.StyleGAN.evaluate.generate import generate
from Research.StyleGAN.train.train_StyleGAN import train
from Research.StyleGAN.utils.config import make_config


def test_pipeline(tmp_path, synth_dataset, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=30, model="classical", batch_size=4, fmap_max=4)
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
    assert {"fidelity", "prediction", "diversity", "k_source_fallback"} == set(j)
    assert j["k_source_fallback"] is False and {r["k_source"] for r in j["prediction"]} >= {"model", "test_mean_diagnostic"}
    assert (run / "diversity.json").exists() and "reference" in j["diversity"]
    assert {"reference_set", "formula", "source", "interpretation", "min", "mean", "max", "paper_threshold", "passed"} <= set(j["fidelity"][0])
    assert {r["source"] for r in j["prediction"]} == {"model", "reference", "random", "freq", "trend"}
    assert {"distinct_sequences", "position_coverage"} <= set(j["diversity"])
    assert {r["reference_set"] for r in j["fidelity"]} == {"train", "test", "test_delta_like"}


def _make_run(tmp_path, synth_dataset, monkeypatch):
    """訓練極小模型並生成，回傳 run 目錄（供塌縮／多樣性測試改寫 generated.npz）。"""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = make_config(1, steps=10, model="classical", batch_size=4, fmap_max=4)
    G, D = tc.build_models(cfg)
    run = tmp_path / "run"
    train(cfg, G, D, np.load(synth_dataset / "vectors.npz")["X"][:60], run, "off")
    generate(run, n=20, data_dir=synth_dataset)
    return run


def _rewrite(run, mut_index):
    """把 generated.npz 的突變集改成指定值（其餘欄位保留），模擬塌縮或多樣的生成結果。"""
    g = dict(np.load(run / "generated.npz"))
    g["mut_index"] = mut_index
    g["mapped_seq"] = np.array([str(i % 7) * 1273 for i in range(len(mut_index))]) if len(set(map(tuple, mut_index))) > 1 else np.array(["A" * 1273] * len(mut_index))
    np.savez(run / "generated.npz", **g)


def test_collapse_warning_written(tmp_path, synth_dataset, monkeypatch):
    """全部相同的突變集 → diversity.json 有塌縮警示（T010）。"""
    run = _make_run(tmp_path, synth_dataset, monkeypatch)
    _rewrite(run, np.tile([[3, 5]], (20, 1)))
    evaluate(run, data_dir=synth_dataset)
    d = json.loads((run / "diversity.json").read_text())
    assert d["distinct_mutation_sets"] == 1 and d["position_coverage"] == 2 and d["collapse_warning"] is True


def test_diverse_no_warning_and_outside_frac(tmp_path, synth_dataset, monkeypatch):
    """多樣的結果不警示；並驗證 test_outside_encoding_frac 與手算一致（T010、FR-001b）。"""
    run = _make_run(tmp_path, synth_dataset, monkeypatch)
    rng = np.random.default_rng(0)
    _rewrite(run, np.stack([rng.choice(64, 2, replace=False) for _ in range(20)]))
    # 讓一條測試期序列在編碼位點之外多一個突變
    v = dict(np.load(synth_dataset / "vectors.npz", allow_pickle=True))
    import pandas as pd
    spike = set(pd.read_csv(synth_dataset / "position_map.csv").spike_pos - 1)
    out = next(p for p in range(1273) if p not in spike and v["X_int"][v["split"] == "test"][:, p].tolist().count(v["ref"][p]) == 50)
    row = int(np.where(v["split"] == "test")[0][0])
    v["X_int"][row, out] = (v["ref"][out] + 1) % 20
    np.savez(synth_dataset / "vectors.npz", **v)
    evaluate(run, data_dir=synth_dataset)
    d = json.loads((run / "diversity.json").read_text())
    assert d["collapse_warning"] is False
    te = v["split"] == "test"
    mut = (v["X_int"][te] != v["ref"]) & (v["X_int"][te] != 20)
    w = v["count"][te]
    exp = float(w[0] / (mut * w[:, None]).sum())
    assert abs(d["test_outside_encoding_frac"]["weighted"] - exp) < 1e-9
    assert abs(d["test_outside_encoding_frac"]["distinct_positions"] - 1 / mut.any(0).sum()) < 1e-9


def test_old_run_without_k_per_sample_falls_back(tmp_path, synth_dataset, monkeypatch):
    """舊檔（如 run01）沒有 k_per_sample：視為固定 k，並標記 k_source_fallback（T014）。"""
    run = _make_run(tmp_path, synth_dataset, monkeypatch)
    g = dict(np.load(run / "generated.npz")); g.pop("k_per_sample")
    np.savez(run / "generated.npz", **g)
    assert evaluate(run, data_dir=synth_dataset)["k_source_fallback"] is True
