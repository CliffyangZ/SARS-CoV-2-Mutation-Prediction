"""JEPA 預訓練：孿生編碼器＋預測頭，二階白化損失，梯度裁剪，每 epoch 檢查點（可續跑，結果與不中斷等價）。"""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import loss as L
from .encoder import Encoder
from .dataset import N_SYM, mutation_vocab
from .qhead import ClassicalHead, MultiQuantumHead, QuantumHead, WideClassicalHead, max_batch
from .views import ViewMaker

DEFAULTS = dict(
    variant="VA", encoder="deepsets", head="classical", n_qubits=6, n_layers=3, entangle=True, trainable_obs=True, trainable_freq=False,
    mask_mut=0.15, mask_pos=0.15, mask_prior="P0", span_prob=0.0, min_mut=3, trim=(0.5, 1.0), local_trim=(0.15, 0.5),
    whiten="vicreg", anchor_sampling="natural", epochs=20, batch_size=256, lr=2e-3, enc_dim=64, clip=1.0,
    sat_reg=1e-3, collapse_rank=2.0, neighbor_dist=4, neighbor_months=1, mem_budget_gb=4.0,
    n_heads=1, recon=0.0, vd_trim=0.02,
)
MULTI_HEADS = ("quantum_multi", "classical_wide")


def _heads(cfg):
    return cfg.get("n_heads", 1) if cfg["head"] in MULTI_HEADS else 1


def angle_dim(cfg):
    """編碼器輸出的角度數 H·n（§3.1 修訂：與編碼器寬度 d_z 解開）。"""
    return cfg["n_qubits"] * _heads(cfg)


def readout_dim(cfg):
    """讀出維度 m = H·2n。"""
    return 2 * cfg["n_qubits"] * _heads(cfg)


class Predictor(nn.Module):
    """預測頭核心（量子或古典）＋共用線性映射 C（可固定為恆等）。"""

    def __init__(self, cfg, seed):
        super().__init__()
        n = cfg["n_qubits"]
        m = readout_dim(cfg)
        if cfg["head"] == "quantum_multi":
            self.core = MultiQuantumHead(n, cfg["n_layers"], cfg["n_heads"], seed=seed, entangle=cfg["entangle"],
                                         trainable_freq=cfg["trainable_freq"])
        elif cfg["head"] == "classical_wide":
            self.core = WideClassicalHead(angle_dim(cfg), m, seed=seed)
        elif cfg["head"] in ("quantum", "quantum_1layer"):
            layers = 1 if cfg["head"] == "quantum_1layer" else cfg["n_layers"]
            self.core = QuantumHead(n, layers, seed=seed, entangle=cfg["entangle"], trainable_freq=cfg["trainable_freq"])
        elif cfg["head"] in ("classical", "classical_matched"):
            ref = QuantumHead(n, cfg["n_layers"], entangle=cfg["entangle"], trainable_freq=cfg["trainable_freq"])
            self.core = ClassicalHead(n, 2 * n, ref.n_quantum_params(), seed=seed)
        else:
            raise ValueError(f"未知的預測頭：{cfg['head']}")
        self.C = nn.Linear(m, m, bias=False)
        if cfg["trainable_obs"]:
            nn.init.orthogonal_(self.C.weight)
        else:
            with torch.no_grad():
                self.C.weight.copy_(torch.eye(m))
            self.C.weight.requires_grad_(False)

    def forward(self, z, **kw):
        """預測頭前向：核心輸出後接線性映射 C。"""
        return self.C(self.core(z, **kw) if kw else self.core(z))


class Model(nn.Module):
    """編碼器加預測頭的孿生模型。"""
    def __init__(self, cfg, seed):
        super().__init__()
        if cfg.get("encoder", "deepsets") == "cnn1d":
            from .encoder_cnn import CNN1DEncoder

            self.enc = CNN1DEncoder(angle_dim(cfg), cfg["enc_dim"], seed=seed)
        else:
            self.enc = Encoder(angle_dim(cfg), cfg["enc_dim"], seed=seed)
        self.pred = Predictor(cfg, seed)

    def embed(self, pos, sym, valid, return_u=False, **kw):
        """回傳 (嵌入 q, 未縮放潛向量 z)；return_u=True 時再加寬潛向量 u。"""
        if return_u:
            zt, z, u = self.enc.encode(pos, sym, valid)
            return self.pred(zt, **kw), z, u
        zt, z = self.enc(pos, sym, valid)
        return self.pred(zt, **kw), z


def n_trainable(model):
    """可訓練參數總數。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def effective_batch(cfg):
    """量子頭依記憶體預算下修批次；放不下回 0。"""
    if cfg["head"] not in ("quantum", "quantum_1layer", "quantum_multi"):
        return cfg["batch_size"]
    layers = 1 if cfg["head"] == "quantum_1layer" else cfg["n_layers"]
    budget = int(cfg["mem_budget_gb"] * 2 ** 30) // _heads(cfg)
    return max_batch(cfg["n_qubits"], layers, budget, cap=cfg["batch_size"])


def _anchors(ts, cfg, seed, epoch):
    rng = np.random.default_rng([seed, epoch, 999])
    n = len(ts)
    if cfg["anchor_sampling"] == "balanced":
        cnt = np.bincount(ts.y)
        w = 1.0 / cnt[ts.y]
        return rng.choice(n, n, p=w / w.sum())
    return rng.permutation(n)


def make_viewmaker(ts, cfg, seed):
    """依設定建立 ViewMaker。"""
    return ViewMaker(ts, variant=cfg["variant"], mask_mut=cfg["mask_mut"], mask_pos=cfg["mask_pos"], trim=tuple(cfg["trim"]),
                     local_trim=tuple(cfg["local_trim"]), min_mut=cfg["min_mut"], mask_prior=cfg["mask_prior"],
                     span_prob=cfg["span_prob"], neighbor_dist=cfg["neighbor_dist"], neighbor_months=cfg["neighbor_months"], seed=seed,
                     vd_trim=cfg.get("vd_trim", 0.02))


def recon_targets(ts, idx, lut, n_vocab, device):
    """錨點序列的完整突變指標（排除 X），(B, V) 的 0/1 張量。"""
    tok = ts.pos[idx].astype(np.int64) * N_SYM + ts.sym[idx].astype(np.int64)
    ids = torch.as_tensor(np.where(ts.valid[idx], lut[tok], -1), device=device)
    out = torch.zeros(len(idx), n_vocab, device=device)
    out.scatter_add_(1, ids.clamp(min=0), (ids >= 0).float())
    return out.clamp_(max=1.0)


def pretrain(ts, cfg, seed, ckpt_path=None, device="cpu", log=print, resume=True):
    """預訓練；回傳 (model, info)。info 含訓練耗時、view／配對統計與每 epoch 損失。"""
    bs = effective_batch(cfg)
    if bs < 1:
        raise MemoryError("量子模擬記憶體不足")
    model = Model(cfg, seed).to(device)
    mu = float(cfg.get("recon", 0.0))
    g_rec, lut, n_vocab = None, None, 0
    if mu > 0:
        lut, n_vocab = mutation_vocab(ts)
        g_rec = nn.Linear(cfg["enc_dim"], n_vocab).to(device)  # §3.3 重建頭，只在預訓練時存在
        model.add_module("recon_head", g_rec)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)
    vm = make_viewmaker(ts, cfg, seed)
    start, elapsed, hist = 0, 0.0, []
    if ckpt_path and resume and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, elapsed, hist = ck["epoch"], ck["elapsed"], ck["hist"]
        vm.stats = ck["stats"]
        log(f"自檢查點續跑：epoch {start}")
    for ep in range(start, cfg["epochs"]):
        t0 = time.time()
        model.train()
        order = _anchors(ts, cfg, seed, ep)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            if len(idx) < 8:
                continue
            pa, sa, va, pb, sb, vb = [torch.as_tensor(x, device=device) for x in vm.batch(idx, epoch=ep)]
            qa, za, ua = model.embed(pa, sa, va, return_u=True)
            qb, zb = model.embed(pb, sb, vb)
            loss, _ = L.vicreg(qa, qb, whiten=cfg["whiten"])
            loss = loss + cfg["sat_reg"] * (za.pow(2).mean() + zb.pow(2).mean())
            if g_rec is not None:
                tgt = recon_targets(ts, idx, lut, n_vocab, device)
                loss = loss + mu * nn.functional.binary_cross_entropy_with_logits(g_rec(ua), tgt)
            opt.zero_grad()
            loss.backward()
            L.clip_gradients(params, cfg["clip"])
            opt.step()
            tot += float(loss.detach())
            nb += 1
        elapsed += time.time() - t0
        hist.append(tot / max(nb, 1))
        log(f"epoch {ep + 1}/{cfg['epochs']} loss={hist[-1]:.4f} t={elapsed:.0f}s")
        if ckpt_path:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep + 1, elapsed=elapsed, hist=hist, stats=vm.stats), ckpt_path)
    return model, dict(train_seconds=elapsed, view_stats=dict(vm.stats), loss_history=hist, n_params=n_trainable(model), batch_size=bs)
