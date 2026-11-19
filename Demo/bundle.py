"""模型包：以既有 Research/QuSpike_PQK 管線訓練 MAIN 並凍結，供 Demo 推論。

研究程式碼沒有存模型的機制，所以 Demo 自己存；不修改研究碼。
模型包不含任何測試集標籤，訓練只用 train（s 只在 validation 上選）。
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np

from Research.QuSpike_JEPA import dataset as D
from Research.QuSpike_JEPA.mutations import UNKNOWN
from Research.QuSpike_PQK import features as F, inference, pipeline, pqk, run

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE = DEMO_DIR / "artifacts" / "bundle_main_seed0"
DEFAULT_CACHE = D.ROOT / "Results" / "quspike_pqk" / "cache"


class BundleMissing(RuntimeError):
    """模型包不存在，需先執行 `python -m Demo.cli build`。"""


class ModelBundle:
    def __init__(self, state, vocab, labels, s_default, cfg, meta):
        self.state, self.vocab, self.labels = state, vocab, labels
        self.s_default, self.cfg, self.meta = s_default, cfg, meta


def _vocab(data):
    train, _, _ = D.load_splits(data)
    keep = train.valid & (train.sym != UNKNOWN)
    return np.unique(train.pos[keep].astype(np.int64) * D.N_SYM + train.sym[keep].astype(np.int64))


def make_featurize(cfg):
    """無快取的 φ：每次對傳入的 z 直接算（單筆查詢不可共用快取檔，否則會拿到舊輸入的特徵）。"""
    def featurize(name, z, tag):
        phi = pqk.features(z, n_qubits=cfg["n_qubits"].get(name, 4), trotter_steps=cfg["trotter_steps"],
                           evolution_time=cfg["evolution_time"], entangle=True, shots=cfg["shots"],
                           noise_p=cfg["noise_p"], device=cfg["device"])
        return np.asarray(phi.cpu().numpy(), np.float64)
    return featurize


def build_bundle(seed=0, out=DEFAULT_BUNDLE, data=F.DEFAULT_DATA, cache=DEFAULT_CACHE, limit=None, log=print):
    """訓練 MAIN（v1 量子超參數、硬路由）並寫出模型包。回傳 ModelBundle。"""
    t0 = time.time()
    cfg = run.make_cfg(run.build_parser().parse_args([]))
    fs = F.load_features("mutation", data)
    if limit:
        fs = run.subsample(fs, limit)
    groups = F.make_groups(fs)
    cache = Path(cache)
    log("在 validation 上選專家權重 s …")
    scale, _, search = pipeline.tune_hyperparams("pqk", True, True, fs, cfg, seed, cache, "zero", "hard", 0.0)
    log(f"s = {scale}；訓練最終模型 …")
    st, _ = pipeline.fit_state("pqk", True, True, fs, groups, cfg, seed, cache, fs.data_hash,
                               supplement="zero", fit_experts=True)
    st.scale, st.routing, st.soft_threshold = scale, "hard", 0.0
    st.ref_model = st.ref_model.cpu()
    st.device = "cpu"
    st.featurize = None  # 閉包不可 pickle；載入時重建
    vocab = _vocab(data)
    if len(vocab) != fs.x_train.shape[1]:
        if not limit:
            raise RuntimeError("詞彙長度與特徵維度不符")
        vocab = _vocab_from_features(fs, data)
    meta = dict(kind="MAIN", seed=seed, s_default=float(scale), labels=list(fs.labels),
                data_hash=fs.data_hash, split_version=fs.split_version, limit=limit,
                n_train=int(len(fs.y_train)), scale_search=search,
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=round(time.time() - t0, 1))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "state.pkl", "wb") as f:
        pickle.dump(dict(state=st, vocab=vocab, cfg=cfg), f, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已寫入 {out}（{meta['seconds']} 秒）")
    return load_bundle(out)


def _vocab_from_features(fs, data):
    """--limit 子集時詞彙仍取自完整 train（x 的欄位定義不變，只有列被抽樣）。"""
    return _vocab(data)


def load_bundle(path=DEFAULT_BUNDLE):
    path = Path(path)
    if not (path / "state.pkl").exists():
        raise BundleMissing(f"找不到模型包 {path}；請先執行 python -m Demo.cli build")
    with open(path / "state.pkl", "rb") as f:
        blob = pickle.load(f)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    st, cfg = blob["state"], blob["cfg"]
    st.featurize = make_featurize(cfg)
    st.device = "cpu"
    st.scale = meta["s_default"]
    return ModelBundle(st, blob["vocab"], meta["labels"], meta["s_default"], cfg, meta)
