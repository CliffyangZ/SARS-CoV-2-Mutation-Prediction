"""深度模型訓練器的單元測試：續跑等價、累計耗時、不用 test 挑檢查點（FR-012）。"""
import numpy as np
import torch
from torch import nn

from Research.lineage_baselines import trainer as TR


def _toy(n=64, length=12, k=3, seed=0):
    """合成 (ids, y)：類別由第一個符號決定，模型很容易學會。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    x = rng.integers(0, 21, size=(n, length)).astype(np.int8)
    x[:, 0] = y
    return x, y


class _Tiny(nn.Module):
    """極小模型：展平後兩層線性（無 dropout，使結果決定性）。"""

    def __init__(self, length=12, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(length, 16), nn.ReLU(), nn.Linear(16, k))

    def forward(self, x):
        return self.net(x)


def _input(ids):
    """ids → float 張量。"""
    return torch.from_numpy(ids.astype(np.float32))


def _fit(tmp, epochs, resume=True, stop_after=None, name="a"):
    x, y = _toy()
    xt, yt = _toy(seed=1)
    torch.manual_seed(0)
    return TR.fit(_Tiny(), x, y, xt, input_fn=_input, epochs=epochs, batch_size=16, lr=1e-2, seed=0,
                  ckpt_path=tmp / f"{name}.pt", resume=resume, device="cpu", amp=False, stop_after=stop_after)


def test_resume_equals_uninterrupted(tmp_path):
    """訓練 3 輪後中斷、再續跑到 6 輪，與一次跑完 6 輪的預測機率在容差內相同。"""
    full = _fit(tmp_path, 6, name="full")
    part = _fit(tmp_path, 6, name="part", stop_after=3)
    assert part["status"] == "incomplete" and part["epochs_done"] == 3
    resumed = _fit(tmp_path, 6, name="part")
    assert resumed["status"] == "complete" and resumed["epochs_done"] == 6
    assert np.allclose(full["proba"], resumed["proba"], atol=1e-5)


def test_elapsed_time_accumulates_across_interruption(tmp_path):
    """續跑後的累計耗時不小於中斷前的耗時。"""
    part = _fit(tmp_path, 4, name="t", stop_after=2)
    resumed = _fit(tmp_path, 4, name="t")
    assert resumed["train_seconds"] >= part["train_seconds"] > 0


def test_completed_checkpoint_not_retrained(tmp_path):
    """已訓練完成的檢查點再次執行時不重訓（epochs_done 不變、權重不變）。"""
    a = _fit(tmp_path, 3, name="c")
    b = _fit(tmp_path, 3, name="c")
    assert b["epochs_done"] == 3 and np.allclose(a["proba"], b["proba"], atol=1e-6)


def test_no_resume_restarts(tmp_path):
    """--no-resume 會忽略既有檢查點從頭訓練。"""
    _fit(tmp_path, 2, name="r", stop_after=1)
    fresh = _fit(tmp_path, 2, name="r", resume=False)
    assert fresh["epochs_done"] == 2 and fresh["status"] == "complete"


def test_test_data_does_not_affect_training(tmp_path):
    """換掉 test 內容，訓練損失曲線完全相同（test 只用於最後評估，不參與訓練與選檢查點）。"""
    x, y = _toy()
    xt1, _ = _toy(seed=1)
    xt2, _ = _toy(seed=2)
    losses = []
    for k, xt in enumerate((xt1, xt2)):
        torch.manual_seed(0)
        out = TR.fit(_Tiny(), x, y, xt, input_fn=_input, epochs=3, batch_size=16, lr=1e-2, seed=0,
                     ckpt_path=tmp_path / f"l{k}.pt", resume=False, device="cpu", amp=False)
        losses.append(out["train_losses"])
    assert losses[0] == losses[1]


class _BNNet(nn.Module):
    """含 BatchNorm 的小模型，用來驗證評估前的統計校準。"""

    def __init__(self, length=12, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(length, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, k))

    def forward(self, x):
        return self.net(x)


def test_bn_recalibration_fixes_stale_running_stats(tmp_path):
    """BatchNorm 的移動平均被破壞後，評估前的重新校準能還原與 train 模式相近的預測。"""
    x, y = _toy(n=256)
    torch.manual_seed(0)
    net = _BNNet()
    TR.fit(net, x, y, x, input_fn=_input, epochs=8, batch_size=32, lr=1e-2, seed=0,
           ckpt_path=tmp_path / "bn.pt", resume=False, device="cpu", amp=False, bn_recal_samples=0)
    bn = net.net[2]
    bn.running_mean.add_(5.0)
    bn.running_var.mul_(20.0)  # 模擬落後或失準的移動平均
    stale = TR.predict_proba(net, x, _input, "cpu", 64, False).argmax(1)
    n = TR.recalibrate_bn(net, x, _input, "cpu", 64, 256, seed=0, amp=False)
    fixed = TR.predict_proba(net, x, _input, "cpu", 64, False).argmax(1)
    assert n == 256
    assert (fixed == y).mean() > (stale == y).mean() + 0.2  # 統計失準時只剩接近亂猜，校準後應明顯回升


def test_recalibration_noop_without_bn_and_uses_train_only():
    """沒有 BatchNorm 的模型不受影響；校準只吃傳入的 train 資料。"""
    x, _ = _toy()
    assert TR.recalibrate_bn(_Tiny(), x, _input, "cpu", 16, 64, seed=0, amp=False) == 0


def test_nonfinite_loss_raises_and_keeps_last_checkpoint(tmp_path):
    """損失變成非有限值時中止，且保留上一輪的檢查點（不寫入損壞的權重）。"""
    import pytest

    x, y = _toy()
    ck = tmp_path / "nan.pt"
    TR.fit(_Tiny(), x, y, x, input_fn=_input, epochs=4, batch_size=16, lr=1e-2, seed=0, ckpt_path=ck,
           resume=False, device="cpu", amp=False, stop_after=2)
    good = torch.load(ck, weights_only=False)["epoch"]
    bad_input = lambda ids: torch.full((len(ids), 12), float("nan"))
    with pytest.raises(FloatingPointError):  # 持續非有限：步數多、跳過超過門檻 → 判定發散
        TR.fit(_Tiny(), x, y, x, input_fn=bad_input, epochs=4, batch_size=2, lr=1e-2, seed=0, ckpt_path=ck,
               resume=True, device="cpu", amp=False)
    assert torch.load(ck, weights_only=False)["epoch"] == good == 2


def test_occasional_nonfinite_step_is_skipped(tmp_path):
    """偶發的非有限批次被跳過，權重不被污染，訓練仍完成。"""
    x, y = _toy()
    calls = {"n": 0}

    def flaky(ids):
        calls["n"] += 1
        t = _input(ids)
        return torch.full_like(t, float("nan")) if calls["n"] == 2 else t

    out = TR.fit(_Tiny(), x, y, x, input_fn=flaky, epochs=2, batch_size=16, lr=1e-2, seed=0,
                 ckpt_path=tmp_path / "f.pt", resume=False, device="cpu", amp=False)
    assert out["status"] == "complete" and out["skipped_steps"] == 1 and np.isfinite(out["proba"]).all()


def test_grad_clip_bounds_update(tmp_path):
    """啟用梯度裁剪時訓練仍可進行且結果有限。"""
    x, y = _toy()
    out = TR.fit(_Tiny(), x, y, x, input_fn=_input, epochs=2, batch_size=16, lr=1e-2, seed=0,
                 ckpt_path=tmp_path / "c.pt", resume=False, device="cpu", amp=False, clip_grad=0.5)
    assert np.isfinite(out["proba"]).all() and out["status"] == "complete"
