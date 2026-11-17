"""文件修訂版（§3.1–3.3、§3.4 V-D、§5 塌陷檢查）的單元測試。"""
import numpy as np
import torch

from Research.QuSpike_JEPA import dataset as DS
from Research.QuSpike_JEPA import finetune as FT
from Research.QuSpike_JEPA import loss as L
from Research.QuSpike_JEPA import pretrain as PT
from Research.QuSpike_JEPA import views as V
from Research.QuSpike_JEPA.mutations import UNKNOWN
from Research.QuSpike_JEPA.qhead import MultiQuantumHead, QuantumHead, WideClassicalHead
from tests.unit.test_quspike_jepa import _toy_tokenset


# --- §3.2 並聯多頭 ---

def test_multihead_matches_single_head_when_H1():
    """H=1 時，MultiQuantumHead 與 QuantumHead 在相同參數下輸出一致。"""
    m = MultiQuantumHead(4, 2, n_heads=1, seed=5, entangle=True)
    s = QuantumHead(4, 2, seed=5, entangle=True)
    with torch.no_grad():
        s.w.copy_(m.w[0])
        s.b.copy_(m.b[0])
    z = torch.zeros(6, 4).uniform_(-1.5, 1.5)
    assert torch.allclose(m.expectations(z), s.expectations(z), atol=1e-5)


def test_multihead_shape_and_independent_params():
    """H 個頭輸出 H·2n 維，且各頭參數獨立（擾動一個頭只影響對應區塊）。"""
    h = MultiQuantumHead(4, 2, n_heads=3, seed=1)
    z = torch.zeros(5, 12).uniform_(-1.5, 1.5)
    out = h.expectations(z)
    assert out.shape == (5, 3 * 8) and out.abs().max() <= 1.0 + 1e-5
    before = out.clone()
    with torch.no_grad():
        h.w[1] += 0.5
    after = h.expectations(z)
    assert torch.allclose(before[:, :8], after[:, :8], atol=1e-6)
    assert not torch.allclose(before[:, 8:16], after[:, 8:16], atol=1e-4)


def test_multihead_gradients_flow():
    """梯度可回傳到輸入與各頭參數。"""
    h = MultiQuantumHead(3, 2, n_heads=2, seed=2)
    z = torch.zeros(4, 6).uniform_(-1.0, 1.0).requires_grad_(True)
    h.expectations(z).sum().backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(h.w.grad).all()
    assert (h.w.grad.abs().sum((1, 2, 3)) > 0).all(), "每個頭都該收到梯度"


def test_wide_classical_head_width_is_m():
    """寬度匹配古典頭的隱藏層等於讀出維度 m，不是參數量匹配的 3 單元。"""
    w = WideClassicalHead(16, 128, seed=0)
    assert w.net[0].out_features == 128
    assert w(torch.randn(4, 16)).shape == (4, 128) and w(torch.randn(4, 16)).abs().max() <= 1.0


# --- §3.1 d_z 與 n 解開 ---

def test_angle_and_readout_dims_decoupled_from_enc_dim():
    """角度數為 H·n、讀出為 H·2n，皆與 enc_dim 無關。"""
    cfg = dict(PT.DEFAULTS, head="quantum_multi", n_qubits=8, n_heads=4, enc_dim=256)
    assert PT.angle_dim(cfg) == 32 and PT.readout_dim(cfg) == 64
    m = PT.Model(cfg, 0)
    assert m.enc.rho[-1].out_features == 32 and m.pred.C.out_features == 64


def test_encoder_latent_width_follows_enc_dim():
    """encode() 回傳的寬潛向量 u 維度為 enc_dim，供重建頭使用。"""
    cfg = dict(PT.DEFAULTS, head="quantum_multi", n_qubits=4, n_heads=2, enc_dim=128)
    m = PT.Model(cfg, 0)
    ts = _toy_tokenset(n=6)
    q, z, u = m.embed(torch.as_tensor(ts.pos.astype(np.int64)), torch.as_tensor(ts.sym.astype(np.int64)),
                      torch.as_tensor(ts.valid), return_u=True)
    assert u.shape == (6, 128) and q.shape == (6, 16) and z.shape == (6, 8)


# --- §3.3 重建項 ---

def test_recon_targets_are_mutation_indicators():
    """重建目標為 0/1 突變指標，排除 X，且與 valid 一致。"""
    ts = _toy_tokenset(n=8)
    lut, V_ = DS.mutation_vocab(ts)
    tgt = PT.recon_targets(ts, np.arange(8), lut, V_, "cpu")
    assert tgt.shape == (8, V_) and set(tgt.unique().tolist()) <= {0.0, 1.0}
    n_mut = [(ts.valid[i] & (ts.sym[i] != UNKNOWN) & (ts.sym[i] != DS.MASK_SYM)).sum() for i in range(8)]
    assert (tgt.sum(1).numpy() <= np.array(n_mut)).all()


def test_recon_term_changes_training():
    """μ>0 時損失軌跡與 μ=0 不同，且重建頭有被建立。"""
    ts = _toy_tokenset(n=120)
    base = dict(PT.DEFAULTS, head="classical", epochs=2, batch_size=32)
    m0, i0 = PT.pretrain(ts, dict(base, recon=0.0), 0, log=lambda s: None)
    m1, i1 = PT.pretrain(ts, dict(base, recon=1.0), 0, log=lambda s: None)
    assert i0["loss_history"] != i1["loss_history"]
    assert hasattr(m1, "recon_head") and not hasattr(m0, "recon_head")


# --- §3.4 V-D ---

def test_vd_preserves_every_mutation_identity():
    """V-D 不得遮蔽或丟棄突變身分：未落在缺失段／截斷區的突變必須原樣保留。"""
    ts = _toy_tokenset(n=40)
    vm = V.ViewMaker(ts, variant="VD", seed=0)
    for i in range(10):
        rng = np.random.default_rng([0, 0, i])
        out = dict(vm.nuisance_view(i, rng))
        v = ts.valid[i]
        orig = dict(zip(ts.pos[i][v].astype(int).tolist(), ts.sym[i][v].astype(int).tolist()))
        for p, s in orig.items():
            if p in out:
                assert out[p] == s or out[p] == UNKNOWN, "突變只能被 X 覆蓋或落在截斷區，不得改成其他符號"
        assert DS.MASK_SYM not in out.values(), "V-D 不使用 MASK"


def test_vd_partner_is_self_and_pairs_differ():
    """V-D 的正樣本是同一條序列，但兩個 view 因 nuisance 不同而不相等。"""
    ts = _toy_tokenset(n=60)
    vm = V.ViewMaker(ts, variant="VD", seed=0)
    assert vm.partner(3) == 3
    pa, sa, va, pb, sb, vb = vm.batch(np.arange(20), epoch=0)
    assert vm.stats["cross_lineage"] == 0
    assert not (pa.shape == pb.shape and np.array_equal(pa, pb) and np.array_equal(sa, sb))


def test_x_run_stats_matches_contiguous_runs():
    """缺失分佈由實際的 X 連續段統計而來。"""
    ts = _toy_tokenset(n=30)
    cnt, lens = V.x_run_stats(ts)
    assert len(cnt) == 30 and (cnt >= 0).all()
    if len(lens):
        assert (lens >= 1).all() and lens.sum() == sum(
            int((ts.valid[i] & (ts.sym[i] == UNKNOWN)).sum()) for i in range(30))


# --- §3.2 跳接 ---

def test_skip_channel_alone_is_the_linear_reference():
    """channels='skip' 時 q 路徑不存在，等同監督線性參考線。"""
    ts = _toy_tokenset(n=60)
    cfg = dict(PT.DEFAULTS, head="classical", epochs=0)
    model = PT.Model(cfg, 0)
    lut, V_ = DS.mutation_vocab(ts)
    clf = FT.Classifier(model, len(ts.labels), lut=lut, n_vocab=V_, channels="skip")
    assert clf.fc is None and clf.skip is not None
    logits = clf(torch.as_tensor(ts.pos.astype(np.int64)), torch.as_tensor(ts.sym.astype(np.int64)), torch.as_tensor(ts.valid))
    assert logits.shape == (60, len(ts.labels))


def test_both_channels_sum_contributions():
    """channels='both' 的 logits 等於兩條通道各自的輸出相加。"""
    ts = _toy_tokenset(n=20)
    cfg = dict(PT.DEFAULTS, head="classical", epochs=0)
    torch.manual_seed(0)
    clf = FT.Classifier(PT.Model(cfg, 0), len(ts.labels), *DS.mutation_vocab(ts), channels="both")
    torch.nn.init.normal_(clf.skip.weight, std=0.1)
    args = (torch.as_tensor(ts.pos.astype(np.int64)), torch.as_tensor(ts.sym.astype(np.int64)), torch.as_tensor(ts.valid))
    total = clf(*args)
    q, _ = clf.model.embed(*args)
    assert torch.allclose(total, clf.fc(clf.norm(q)) + clf.skip(clf.mut_indicator(*args)), atol=1e-5)


def test_mut_indicator_is_binary_and_excludes_unknown():
    """跳接特徵為 0/1，且不含 X。"""
    ts = _toy_tokenset(n=25)
    clf = FT.Classifier(PT.Model(dict(PT.DEFAULTS, head="classical"), 0), len(ts.labels),
                        *DS.mutation_vocab(ts), channels="skip")
    x = clf.mut_indicator(torch.as_tensor(ts.pos.astype(np.int64)), torch.as_tensor(ts.sym.astype(np.int64)),
                          torch.as_tensor(ts.valid))
    assert set(x.unique().tolist()) <= {0.0, 1.0} and x.shape[1] == clf.n_vocab


# --- §5 塌陷檢查 ---

def test_variance_floor_catches_scale_collapse():
    """尺度塌陷（變異趨近 0 但有殘餘雜訊）會被方差判準抓到，有效秩抓不到。"""
    x = torch.randn(256, 8) * 1e-6
    assert L.min_dim_std(x) < 1e-3 and L.dead_dim_frac(x) == 1.0
    assert L.effective_rank(x) > 2.0, "有效秩與尺度無關，故抓不到這種塌陷"


def test_dead_dim_frac_tolerates_single_dead_dimension():
    """寬讀出中單一維度失效不算塌陷（實測 m=128 時 1 個死維度仍有正常表現）。"""
    x = torch.randn(256, 128)
    x[:, 7] = 0.5  # 一個常數維度
    assert L.min_dim_std(x) < 1e-3, "最小值判準會誤報"
    assert L.dead_dim_frac(x) < 0.5, "比例判準不應誤報"


def test_dead_dim_frac_flags_majority_dead():
    """多數維度失去變異時判為塌陷。"""
    x = torch.randn(256, 16) * 1e-9
    x[:, :4] = torch.randn(256, 4)
    assert L.dead_dim_frac(x) > 0.5


def test_majority_baseline_matches_closed_form():
    """多數類基準與解析值一致。"""
    y = np.r_[np.zeros(80), np.ones(20)].astype(int)
    assert abs(L.majority_macro_f1(y, 2) - (2 * 0.8 / 1.8 / 2)) < 1e-9
