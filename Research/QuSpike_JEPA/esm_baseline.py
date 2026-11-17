"""預訓練蛋白質語言模型基準：凍結 ESM-2 的嵌入（整條序列平均池化）＋與 JEPA 相同的 probe 與切分。

ESM-2 上限約 1022 個 token，Spike 有 1273 位，故以長 1000、步幅 600 的視窗分段，視窗重疊處取平均。
權重只從 safetensors 載入。嵌入快取在 Results/quspike_jepa/esm_<模型>_<資料雜湊>.npy。
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from Research.lineage_baselines.metrics import compute_metrics
from Research.lineage_baselines.results_io import environment_info, is_complete, result_path, write_result

from . import dataset as D
from . import probe as PR

MODEL = "facebook/esm2_t30_150M_UR50D"
WINDOW, STRIDE = 1000, 600


def windows(n):
    """涵蓋 [0, n) 的視窗起點。"""
    starts = list(range(0, max(n - WINDOW, 0) + 1, STRIDE))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    return starts


def align_pairs(seq):
    """回傳 (seq_idx, ref_idx, mutated)：序列與參考對位的殘基（0 起算），mutated 標示殘基不同者。

    插入與內部刪除沒有成對殘基，不在其中；首尾比參考短的部分也不在其中。
    """
    from difflib import SequenceMatcher

    from .mutations import N_POS, REFERENCE

    if len(seq) == N_POS:
        j = np.arange(N_POS)
        return j, j, np.array([a != b for a, b in zip(seq, REFERENCE)])
    js, is_ = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, REFERENCE, seq, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            k = min(i2 - i1, j2 - j1)
            is_ += list(range(i1, i1 + k))
            js += list(range(j1, j1 + k))
    js, is_ = np.array(js, int), np.array(is_, int)
    return js, is_, np.array([seq[j] != REFERENCE[i] for j, i in zip(js, is_)], bool)


def site_features(h, h_ref, seq):
    """由逐殘基嵌入 h（len, d）與參考的 h_ref（N_POS, d）組出三段特徵：

    平均池化、突變位點的嵌入差平均（無突變則為 0）、全部對位殘基嵌入差的逐維最大絕對值。
    """
    j, i, mut = align_pairs(seq)
    d = h[torch.as_tensor(j, device=h.device)] - h_ref[torch.as_tensor(i, device=h.device)]
    m = torch.as_tensor(mut, device=h.device)
    site = d[m].mean(0) if bool(m.any()) else torch.zeros(h.shape[1], device=h.device)
    return torch.cat([h.mean(0), site, d.abs().amax(0)])


def _residue_embeddings(model, tok, s, device):
    """整條序列的逐殘基嵌入（視窗重疊處取平均），形狀 (len, hidden)。"""
    hid = model.config.hidden_size
    sums = torch.zeros(len(s), hid, device=device)
    cnt = torch.zeros(len(s), 1, device=device)
    for st in windows(len(s)):
        seg = s[st:st + WINDOW]
        enc = tok(seg, return_tensors="pt").to(device)
        sums[st:st + len(seg)] += model(**enc).last_hidden_state[0, 1:-1].float()
        cnt[st:st + len(seg)] += 1
    return sums / cnt


@torch.no_grad()
def embed_site_features(seqs, model_name=MODEL, device="cuda", log=print):
    """回傳 (n, 3·hidden)：[平均池化, 突變位點差平均, 逐維最大絕對差]。"""
    from transformers import AutoModel, AutoTokenizer

    from .mutations import REFERENCE

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    if device == "cuda":
        model = model.half()
    h_ref = _residue_embeddings(model, tok, REFERENCE, device)
    out = np.zeros((len(seqs), 3 * model.config.hidden_size), np.float32)
    t0 = time.time()
    for n, s in enumerate(seqs):
        out[n] = site_features(_residue_embeddings(model, tok, s, device), h_ref, s).cpu().numpy()
        if n % 2000 == 0:
            log(f"  ESM 位點特徵 {n}/{len(seqs)}  {time.time() - t0:.0f}s")
    return out


@torch.no_grad()
def embed_sequences(seqs, model_name=MODEL, device="cuda", batch=8, log=print):
    """回傳 (n, hidden) 的平均池化嵌入。"""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    if device == "cuda":
        model = model.half()
    out = np.zeros((len(seqs), model.config.hidden_size), np.float32)
    t0 = time.time()
    for i in range(0, len(seqs), batch):
        part = seqs[i:i + batch]
        for r, s in enumerate(part):
            acc, cnt = None, np.zeros(len(s))
            sums = torch.zeros(len(s), model.config.hidden_size, device=device)
            for st in windows(len(s)):
                seg = s[st:st + WINDOW]
                enc = tok(seg, return_tensors="pt").to(device)
                h = model(**enc).last_hidden_state[0, 1:-1].float()
                sums[st:st + len(seg)] += h
                cnt[st:st + len(seg)] += 1
            out[i + r] = (sums / torch.as_tensor(cnt, device=device).unsqueeze(1).float()).mean(0).cpu().numpy()
        if (i // batch) % 100 == 0:
            log(f"  ESM 嵌入 {i}/{len(seqs)}  {time.time() - t0:.0f}s")
    return out


def main(argv=None):
    """命令列入口：取 ESM-2 嵌入並以與 JEPA 相同的切分與 probe 評估。"""
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(D.DEFAULT_DATA))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=str(D.CACHE_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--features", choices=["meanpool", "sites"], default="meanpool")
    a = p.parse_args(argv)
    df = pd.read_csv(a.data, sep="\t", dtype={"date": str})
    labels = sorted(df.lineage.unique())
    idx = {l: i for i, l in enumerate(labels)}
    tr_df = df[df.split == "train"].reset_index(drop=True)
    te_df = df[df.split == "test"].reset_index(drop=True)
    if a.limit:
        tr_df, te_df = tr_df.head(a.limit), te_df.head(a.limit)
    fh = D.file_hash(a.data)
    short = a.model.split("/")[-1]
    tag = "" if a.features == "meanpool" else "_sites"
    cache = D.Path(a.out) / f"esm_{short}_{fh}_{a.limit}{tag}.npz"
    embed = embed_sequences if a.features == "meanpool" else embed_site_features
    if cache.exists():
        z = np.load(cache)
        f_tr, f_te = z["train"], z["test"]
    else:
        t0 = time.time()
        f_tr = embed(tr_df.sequence.tolist(), a.model, a.device)
        f_te = embed(te_df.sequence.tolist(), a.model, a.device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, train=f_tr, test=f_te)
        print(f"嵌入完成 {time.time() - t0:.0f}s")
    y_tr, y_te = tr_df.lineage.map(idx).values, te_df.lineage.map(idx).values
    # 與 JEPA 相同的 validation 規則：train 內各譜系日期最晚 15%
    ts = D.TokenSet(np.zeros((len(y_tr), 1), np.int16), np.zeros((len(y_tr), 1), np.int8), np.ones((len(y_tr), 1), bool), y_tr,
                    np.array([D.parse_month(d) for d in tr_df.date], np.int32), labels)
    fit, val = D.split_validation(ts)
    name = f"esm2_{short}_{'meanpool' if a.features == 'meanpool' else 'sites'}"
    for seed in a.seeds:
        path = result_path(a.out, name, "mut", seed)
        if is_complete(path):
            print(f"[跳過] {path.name}")
            continue
        (rv,), _ = PR.probe(f_tr[fit], y_tr[fit], [f_tr[val]], y_tr[val], len(labels), labels, seed=seed)
        (rt,), _ = PR.probe(f_tr, y_tr, [f_te], y_te, len(labels), labels, seed=seed)
        res = dict(config=dict(name=name, model=a.model, window=WINDOW, stride=STRIDE, head="esm_frozen", features=a.features), seed=seed,
                   split_version=f"{fh}+{D.RULE_VERSION}", config_origin={"model": "chosen", "window": "chosen"}, uses_label=False,
                   status="complete", collapsed=False, effective_rank=None, train_seconds=0.0, environment=environment_info(),
                   metrics_val=rv["linear"], metrics_val_knn=rv["knn"], per_class_val=rv["per_class"],
                   metrics_test=rt["linear"], metrics_test_knn=rt["knn"], per_class_test=rt["per_class"])
        write_result(path, res)
        print(f"[{name} seed{seed}] val macroF1={rv['linear']['f1_macro']:.4f} test macroF1={rt['linear']['f1_macro']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
