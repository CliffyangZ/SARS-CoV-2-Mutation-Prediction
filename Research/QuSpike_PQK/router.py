"""古典路由與參考線（T018）。

路由器就是監督線性參考線本身（突變指標 → 22 類），這有兩個作用：
1. 它是對照組 C0，也是架構上的下限——量子路徑貢獻為零時整個模型還原成它（FR-009）。
2. 它決定每個樣本進哪一群；易分群直接用它的預測輸出（FR-010），
   所以 `others` 群的預測與 C0 必須逐筆完全相同（SC-004）。
"""
import numpy as np
import torch
from torch import nn


def fit_reference(x_train, y_train, n_classes, seed=0, epochs=60, lr=3e-3, wd=1e-4,
                  batch=512, device=None):
    """訓練監督線性參考線（與 QuSpike_JEPA.baselines.mutset_linear 同一設定）。"""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    x = torch.as_tensor(np.asarray(x_train, np.float32), device=dev)
    y = torch.as_tensor(np.asarray(y_train, np.int64), device=dev)
    model = nn.Linear(x.shape[1], int(n_classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(int(epochs)):
        perm = torch.randperm(len(y), generator=g).to(dev)
        for s in range(0, len(y), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return model, dev


def logits(model, x, device, batch=4096):
    """參考線的 logits（float64 numpy）。"""
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.as_tensor(np.asarray(x[s:s + batch], np.float32), device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def group_of_class(groups):
    """類別索引 → 群名。"""
    return {c: g.name for g in groups for c in g.class_ids}


def route(ref_logits, groups):
    """以參考線的預測決定每個樣本的群。回傳 (群名陣列, 參考線預測)。"""
    pred = ref_logits.argmax(1)
    lut = group_of_class(groups)
    return np.array([lut[int(p)] for p in pred]), pred


def router_accuracy(routed, y_true, groups):
    """路由準確率：樣本被分到其**真實類別**所屬的群的比例（FR-010）。

    這是量子專家能改善的上限——被路錯群的樣本，專家根本看不到。
    """
    lut = group_of_class(groups)
    truth = np.array([lut[int(c)] for c in y_true])
    return float((routed == truth).mean())
