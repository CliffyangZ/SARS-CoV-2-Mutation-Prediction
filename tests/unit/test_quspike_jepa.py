"""QuSpike-JEPA 單元測試（文件 §6）：傅立葉度數、單頻退化、塌陷偵測、view 標籤一致性。"""
import numpy as np
import pytest
import torch

from Research.QuSpike_JEPA import dataset as DS
from Research.QuSpike_JEPA import loss as L
from Research.QuSpike_JEPA import mutations as MUT
from Research.QuSpike_JEPA import views as V
from Research.QuSpike_JEPA.qhead import ClassicalHead, QuantumHead


def _spectrum(head, dim=0, n_grid=64, other=0.3):
    """固定其他維度，掃描第 dim 維 z ∈ [−π, π)，回傳 p 的 rFFT 幅值 (freq, m)。"""
    z = torch.full((n_grid, head.n), other)
    z[:, dim] = torch.linspace(-np.pi, np.pi, n_grid + 1)[:-1]
    with torch.no_grad():
        p = head.expectations(z)
    return torch.fft.rfft(p, dim=0).abs() / n_grid


# --- §3.2 頻譜性質 ---

def test_fourier_degree_at_most_L():
    """固定 a=1 時，期望值對 z̃ 的三角多項式度數 ≤ L（Schuld 式 25）。"""
    for layers in (1, 2, 3):
        head = QuantumHead(n_qubits=4, layers=layers, seed=1)
        amp = _spectrum(head)
        assert amp[: layers + 1].max() > 1e-3, "低頻應有能量"
        assert amp[layers + 1:].max() < 1e-5, f"L={layers} 時出現度數 > L 的頻率"


def test_L1_is_single_frequency():
    """L=1 退化為單頻：只有 0 與 1 次項（H2 的 sanity ablation）。"""
    amp = _spectrum(QuantumHead(n_qubits=4, layers=1, seed=2))
    assert amp[2:].max() < 1e-5


def test_trainable_freq_breaks_integer_spectrum():
    """a 可訓練時頻率非整數，不再是標準傅立葉級數（度數斷言不適用）。"""
    head = QuantumHead(n_qubits=4, layers=2, trainable_freq=True, seed=3)
    with torch.no_grad():
        head.a.mul_(0.0).add_(1.7)
    amp = _spectrum(head)
    assert amp[3:].max() > 1e-5, "非整數頻率應在整數網格上洩漏到高頻"


def test_expectations_bounded_and_differentiable():
    """期望值落在 [−1,1]，且梯度可回傳到輸入 z̃。"""
    head = QuantumHead(n_qubits=5, layers=2, seed=4)
    z = torch.zeros(8, 5).uniform_(-np.pi / 2, np.pi / 2).requires_grad_(True)
    p = head.expectations(z)
    assert p.shape == (8, 10) and p.abs().max() <= 1.0 + 1e-5
    p.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert torch.isfinite(head.w.grad).all()


def test_parameter_count_matches_doc():
    """固定 a 時量子線路參數量為 3nL + 2n（文件 §3.2）。"""
    n, l = 6, 3
    assert QuantumHead(n_qubits=n, layers=l).n_quantum_params() == 3 * n * l + 2 * n
    assert QuantumHead(n_qubits=n, layers=l, trainable_freq=True).n_quantum_params() == 4 * n * l + 2 * n


def test_classical_head_matches_parameter_budget():
    """古典對照頭的參數量與量子頭相差在 10% 內（H3 的公平比較）。"""
    q = QuantumHead(n_qubits=6, layers=3)
    target = sum(p.numel() for p in q.parameters() if p.requires_grad)
    c = ClassicalHead(6, 12, target_params=target)
    got = sum(p.numel() for p in c.parameters())
    assert abs(got - target) / target < 0.1


def test_entanglement_changes_output():
    """CZ 環有無會改變輸出（糾纏消融確實生效）。"""
    z = torch.zeros(4, 4).uniform_(-1, 1)
    a = QuantumHead(n_qubits=4, layers=2, seed=5, entangle=True)
    b = QuantumHead(n_qubits=4, layers=2, seed=5, entangle=False)
    with torch.no_grad():
        assert not torch.allclose(a.expectations(z), b.expectations(z), atol=1e-4)


# --- §3.1 突變表示 ---

def test_mutation_extraction_substitution_and_deletion():
    """取代與刪除都能正確定位；未變動的位點不產生 token。"""
    ref = MUT.REFERENCE
    seq = ref[:613] + "G" + ref[614:]  # D614G（1 起算）
    m = MUT.mutations_of(seq)
    assert m == [(614, MUT.SYM_INDEX["G"])]
    seq2 = ref[:100] + ref[103:]  # 刪除 101–103
    pos = [p for p, s in MUT.mutations_of(seq2) if MUT.SYMBOLS[s] == "-"]
    assert pos == [101, 102, 103]


def test_unknown_residue_kept_as_token():
    """X 保留為未知 token，不被當成與參考相同。"""
    seq = MUT.REFERENCE[:200] + "X" + MUT.REFERENCE[201:]
    assert (201, MUT.UNKNOWN) in MUT.mutations_of(seq)


# --- §3.4 views ---

def _toy_tokenset(n=60, n_class=3, seed=0):
    rng = np.random.default_rng(seed)
    T = 12
    pos = rng.integers(1, MUT.N_POS, size=(n, T)).astype(np.int16)
    sym = rng.integers(0, 20, size=(n, T)).astype(np.int8)
    valid = np.ones((n, T), dtype=bool)
    y = np.arange(n) % n_class
    month = np.full(n, 24000, dtype=np.int32)
    return DS.TokenSet(pos, sym, valid, y, month, [f"L{i}" for i in range(n_class)])


def test_VA_partner_same_lineage_VC_is_self():
    """V-A 的配對同譜系；V-C 的配對為自己（bad view 對照）。"""
    ts = _toy_tokenset()
    va = V.ViewMaker(ts, variant="VA", seed=0)
    assert all(ts.y[va.partner(i)] == ts.y[i] for i in range(len(ts)))
    vc = V.ViewMaker(ts, variant="VC", seed=0)
    assert all(vc.partner(i) == i for i in range(len(ts)))


def test_masked_positions_become_mask_tokens_not_dropped():
    """位置層級遮蔽要產生 [MASK] token，而非直接丟掉（否則未知會被誤認為等於參考）。"""
    ts = _toy_tokenset()
    vm = V.ViewMaker(ts, variant="VC", mask_mut=0.0, mask_pos=0.5, trim=(1.0, 1.0), seed=1)
    syms = [s for _ in range(20) for _, s in vm.one_view(0, local=False)]
    assert DS.MASK_SYM in syms


def test_trim_keeps_contiguous_interval():
    """trim 只保留一段連續位置區間內的突變。"""
    ts = _toy_tokenset()
    vm = V.ViewMaker(ts, variant="VC", mask_mut=0.0, mask_pos=0.0, local_trim=(0.15, 0.2), seed=2)
    for _ in range(10):
        p = [q for q, _ in vm.one_view(0, local=True)]
        full = sorted(ts.pos[0][ts.valid[0]].tolist())
        assert all(q in full for q in p)
        if len(p) >= 2:
            assert max(p) - min(p) <= 0.2 * MUT.N_POS + 1


def test_view_batch_shapes():
    """批次 view 的形狀與 padding 一致。"""
    ts = _toy_tokenset()
    vm = V.ViewMaker(ts, variant="VA", seed=3)
    out = vm.batch(np.arange(8))
    assert len(out) == 6 and all(o.shape[0] == 8 for o in out)


def test_P1_prior_masks_common_tokens_more():
    """P1 先驗下，高頻 token 的遮蔽率高於稀有 token（§3.6）。"""
    ts = _toy_tokenset(n=100)
    ts.pos[:, 0], ts.sym[:, 0] = 614, MUT.SYM_INDEX["G"]  # 所有序列共有
    vm = V.ViewMaker(ts, variant="VC", mask_prior="P1", seed=4)
    common = vm._mask_rate(np.array([614 * DS.N_SYM + MUT.SYM_INDEX["G"]]))
    rare = vm._mask_rate(np.array([ts.pos[0, 1] * DS.N_SYM + ts.sym[0, 1]]))
    assert common[0] > rare[0]


# --- §7-R2 塌陷偵測 ---

def test_effective_rank_detects_collapse():
    """完全塌陷時有效秩接近 1，獨立維度時接近維度數。"""
    x = torch.randn(256, 1) * torch.ones(1, 8)
    assert L.effective_rank(x) < 1.1
    assert L.effective_rank(torch.randn(2048, 8)) > 7.0


def test_vicreg_penalises_collapse_more_than_spread():
    """塌陷的嵌入 VICReg 損失高於白化良好的嵌入。"""
    g = torch.Generator().manual_seed(0)
    spread = torch.randn(256, 8, generator=g)
    collapsed = torch.zeros(256, 8)
    a, _ = L.vicreg(spread, spread + 0.01 * torch.randn(256, 8, generator=g))
    b, _ = L.vicreg(collapsed, collapsed)
    assert float(b) > float(a)


@pytest.mark.slow
def test_real_sequence_mutations_match_known_lineage():
    """真實資料：B.1.351 應含 E484K、N501Y、D614G。"""
    import pandas as pd

    path = DS.ROOT / "Data" / "spike-lineage" / "dataset_v2.tsv"
    if not path.exists():
        pytest.skip("缺少資料檔")
    df = pd.read_csv(path, sep="\t", usecols=["lineage", "sequence"])
    seq = df[df.lineage == "B.1.351"].sequence.iloc[0]
    got = {f"{MUT.REFERENCE[p - 1]}{p}{MUT.SYMBOLS[s]}" for p, s in MUT.mutations_of(seq)}
    assert {"E484K", "N501Y", "D614G"} <= got
