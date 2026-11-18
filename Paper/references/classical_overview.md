# lineage_baselines_v2 — Classical 基線統計總覽

資料來源：[classical/](classical/) 內 58 個 `*.json`（`status=complete`），逐 seed 整理於 [classical_seeds.csv](classical_seeds.csv)。
Log：`../lineage_baselines_v2.classical.log`（結尾 `ALLDONE`，無失敗 run）。

## 設定

- 任務：22 類 lineage 分類；train 22,154 / test 9,503；common_length = 1266
- 表徵：`onehot`（one-hot 序列）、`ae_embedding`（AE 128 維，encoder 1024-512-256-128，100 epochs）
- 模型：svm / nb / mlp / knn / rf / lr / dt
- 指標：mean ± std（跨 seed，樣本標準差）；排序依 macro-F1
- 硬體：RTX 5060 Laptop；sklearn 1.9.0

## Seed 數

| 模型 | onehot | ae_embedding | 備註 |
|---|---|---|---|
| mlp / rf / dt / lr | 5 (0–4) | 5 | |
| svm / nb / knn | **1 (seed0)** | 5 | onehot 版為確定性演算法（`deterministic=True`），只跑一次 |

> LR onehot 雖有 5 seeds，但 5 次結果完全相同（std = 0），因為 lbfgs 是確定性的；且 log 顯示每次都觸及 `max_iter=1000` 未收斂。

## onehot（依 macro-F1 排序）

| 模型 | n | Accuracy | macro-F1 | weighted-F1 | macro-Prec | macro-Rec | ROC-AUC (ovr) | 訓練秒數 |
|---|---|---|---|---|---|---|---|---|
| **lr** | 5 | **0.6783 ± 0.0000** | **0.7339 ± 0.0000** | 0.6707 | 0.7379 | 0.7443 | 0.9671 | 370 |
| rf | 5 | 0.6593 ± 0.0013 | 0.6974 ± 0.0019 | 0.6450 | 0.7217 | 0.7006 | 0.9559 | 35 |
| mlp | 5 | 0.6523 ± 0.0094 | 0.6978 ± 0.0190 | 0.6438 | 0.7239 | 0.7068 | 0.9626 | 1187 |
| dt | 5 | 0.6424 ± 0.0020 | 0.6927 ± 0.0065 | 0.6351 | 0.6873 | 0.7179 | 0.8534 | 24 |
| knn | 1 | 0.6021 | 0.6409 | 0.5916 | 0.6754 | 0.6282 | 0.8830 | <1 |
| svm | 1 | 0.5542 | 0.4410 | 0.4904 | 0.4768 | 0.4642 | 0.9382 | 379 |
| nb | 1 | 0.3626 | 0.2622 | 0.3090 | 0.3274 | 0.3287 | 0.8596 | <1 |

（rf 與 mlp 的 macro-F1 實質持平：0.6974 vs 0.6978，差距遠小於 mlp 的 std 0.019。）

## ae_embedding（依 macro-F1 排序）

| 模型 | n | Accuracy | macro-F1 | weighted-F1 | macro-Prec | macro-Rec | ROC-AUC (ovr) |
|---|---|---|---|---|---|---|---|
| rf | 5 | 0.5769 ± 0.0071 | 0.5696 ± 0.0125 | 0.5639 | 0.6092 | 0.5784 | 0.9233 |
| knn | 5 | 0.5451 ± 0.0074 | 0.5516 ± 0.0131 | 0.5352 | 0.5656 | 0.5620 | 0.8562 |
| mlp | 5 | 0.5477 ± 0.0118 | 0.5185 ± 0.0175 | 0.5213 | 0.5235 | 0.5518 | 0.9246 |
| dt | 5 | 0.5166 ± 0.0038 | 0.5130 ± 0.0090 | 0.5124 | 0.5028 | 0.5318 | 0.7541 |
| lr | 5 | 0.5353 ± 0.0150 | 0.4842 ± 0.0223 | 0.4842 | 0.5021 | 0.5117 | 0.9217 |
| svm | 5 | 0.4933 ± 0.0099 | 0.3792 ± 0.0228 | 0.4166 | 0.4220 | 0.4110 | 0.8979 |
| nb | 5 | 0.2134 ± 0.0222 | 0.2418 ± 0.0191 | 0.2242 | 0.3314 | 0.3092 | 0.7949 |

AE 表徵的訓練秒數（約 540–890 s）主要包含 AE 本身的訓練時間，不代表分類器本身耗時，故未列入比較。

## 主要觀察

1. **onehot 明顯優於 ae_embedding**：同一模型下 macro-F1 差 0.13–0.22（rf 0.697→0.570、lr 0.734→0.484、mlp 0.698→0.519）。128 維 AE 壓縮損失了區分相近 lineage 的資訊。
2. **最佳 baseline 是 LR + onehot**（acc 0.678 / macro-F1 0.734），但未收斂（max_iter=1000 用盡），且無 seed 變異可言。作為論文比較基準時建議註明此點，或加大 max_iter 重跑確認。
3. **rf 最穩定且性價比高**：onehot 下 std 僅 0.002，35 秒完成；mlp 要 ~20 分鐘卻不比 rf 好，且 seed 變異大 10 倍。
4. **SVM / NB 表現差**：SVM 的 ROC-AUC 高（0.938）但 macro-F1 僅 0.44，代表排序能力尚可、決策邊界/類別偏置有問題（default C=1、rbf、OVR）；BernoulliNB 的獨立性假設不適用。兩者皆屬未調參預設值。
5. **ROC-AUC 普遍很高（0.95+）但 F1 偏低**，差距來自少數難分類別（見下）。

## 逐類別難易（lr / rf，onehot，seed0）

- **難**（F1 < 0.5）：B.1.1（LR 0.27 / RF 0.29）、B.1（0.37 / 0.29）、B.1.617.2（0.47 / 0.40）、AY.44、AY.103 — 皆為祖先/泛型 lineage 或 Delta 子系，序列差異小、彼此混淆。
- **易**（F1 > 0.95）：B.1.617.1、C.37、P.1、B.1.1.519、B.1.1.7（lr）— 具特徵性 spike 突變。
- 小類別（AY.2 n=14、B.1.617.1 n=14、C.37 n=35）F1 高但支持數很少，估計不穩。

## 注意事項

- log 中大量 `ConvergenceWarning`：mlp（max_iter=200 未收斂，ae_embedding 與 onehot 皆有）、lr onehot（max_iter=1000）。
- 超參數多數標記為 `chosen`（非論文給定），詳見各 JSON 的 `provenance`。
- 本檔只涵蓋 classical；deep（fcn/lstm3bi/mwdn/cnn3）結果見 `../lineage_baselines_v2.deep.log`，其中 lstm3bi seed0/1 與 mwdn seed2 因 loss 非有限值失敗。
