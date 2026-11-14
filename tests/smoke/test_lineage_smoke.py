"""深度 baseline 的冒煙測試：每個模型一個小 batch 的前向與反向，無 NaN、輸出形狀正確。"""
import numpy as np
import pytest
import torch

from Research.lineage_baselines import deep_models as DM
from Research.lineage_baselines import transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 5


def _step(model, x):
    """做一次前向、反向與優化器更新，回傳 logits。"""
    model = model.to(DEV).train()
    out = model(x.to(DEV))
    loss = torch.nn.functional.cross_entropy(out, torch.zeros(len(x), dtype=torch.long, device=DEV))
    loss.backward()
    torch.optim.Adam(model.parameters()).step()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return out


@pytest.mark.parametrize("name", ["cnn3", "cnn4", "resnet34"])
def test_image_models(name):
    """影像模型吃 (B,3,360,360)，輸出 (B,K)。"""
    sig = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (2, 100)).astype(np.float32))
    out = _step(DM.build(name, K), T.to_image("rp", sig))
    assert out.shape == (2, K)


@pytest.mark.parametrize("name", ["fcn", "lstm3bi", "mwdn"])
def test_signal_models(name):
    """訊號模型吃 (B,L)，輸出 (B,K)。"""
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (3, 128)).astype(np.float32))
    assert _step(DM.build(name, K), x).shape == (3, K)


def test_resnet34_pretrained_has_imagenet_norm():
    """預訓練版的 ResNet34 外層加 ImageNet 正規化；非預訓練版沒有。"""
    assert hasattr(DM.build("resnet34", K, pretrained=True), "mean")
    assert not hasattr(DM.build("resnet34", K, pretrained=False), "mean")


def test_resnet34_loads_local_pretrained_file():
    """pretrained/ 內有權重檔時，從本地載入（不需網路），且與 torchvision 快取的權重一致。"""
    if not DM.RESNET34_FILE.exists():
        pytest.skip("pretrained/ 沒有 ResNet34 權重")
    sd = torch.load(DM.RESNET34_FILE, map_location="cpu", weights_only=True)
    net = DM.build("resnet34", K, pretrained=True).inner
    assert torch.equal(net.conv1.weight, sd["conv1.weight"]) and torch.equal(net.layer4[2].conv2.weight, sd["layer4.2.conv2.weight"])
    assert net.fc.out_features == K
