# lineage_baselines_cumsum_v2 — CNN / ResNet34 × 影像表徵總覽（22 類）

資料來源：本目錄 9 個 `*.json`（皆 `status=complete`、`epochs_done=100`、`skipped_steps=0`，僅 **seed0**）。
Log：[train.log](train.log)。開頭 rp/gaf/mtf 三輪因環境缺 `pandas` 立即失敗（2026-11-16 13:12），之後重跑 rp → gaf → mtf 全部完成，結尾 `ALLDONE 2026-11-17T05:36`。9 組 run 無失敗。
Checkpoint：[checkpoints/](checkpoints/)（9 個 `.pt`）。

## 設定

- 任務：**22 類** lineage 分類（`Data/spike-lineage/dataset_v2.tsv`）；`../lineage_baselines_cumsum` 是舊的 11 類版本，兩者不可直接比。
- 樣本：train 22,154 / test 9,503（排除過短序列後；`common_length = 1266`）
- 訊號：`signal_mode=cumsum`（去均值後沿序列累積、再縮放到 [-1, 1]；論文未說明，provenance=`chosen`）
- 表徵：Spike2Signal → **RP** / **GAF** / **MTF** → 影像
- 模型：
  - `cnn3`：32-64-128
  - `cnn4`：32-64-128-256
  - `resnet34`：torchvision，**ImageNet 預訓練**，ImageNet 均值／標準差正規化，fc 替換；batch=16
  - cnn3/cnn4：Adam、lr=1e-3、batch=32、100 epochs、NLL、bf16 AMP
- 評估前以 train 子集重算 BatchNorm 統計（4096 筆）。
- 指標：單次 seed0，無跨 seed 標準差；排序依 macro-F1。
- 硬體：RTX 5060 Laptop；torch 2.11.0+cu128；峰值顯存 cnn 約 1.6 GB、resnet34 約 1.1 GB。

## 結果總表（依 macro-F1 排序）

| 模型 | 表徵 | Accuracy | macro-F1 | weighted-F1 | macro-Prec | macro-Rec | ROC-AUC (ovr) | 訓練秒數 |
|---|---|---|---|---|---|---|---|---|
| **cnn4** | **rp** | 0.6333 | **0.6787** | 0.6185 | 0.6920 | 0.6868 | 0.9640 | 5462 |
| cnn4 | gaf | 0.6256 | 0.6755 | 0.6096 | 0.6886 | 0.6887 | 0.9616 | 5625 |
| resnet34 | rp | 0.6163 | 0.6633 | 0.6093 | 0.6610 | 0.6848 | 0.9410 | 9152 |
| resnet34 | gaf | 0.6069 | 0.6532 | 0.5993 | 0.6453 | 0.6749 | 0.9437 | 9332 |
| cnn3 | rp | 0.6149 | 0.6328 | 0.5700 | 0.6890 | 0.6539 | 0.9637 | 4988 |
| cnn3 | gaf | 0.6092 | 0.6197 | 0.5614 | 0.6754 | 0.6432 | 0.9624 | 5151 |
| resnet34 | mtf | 0.5544 | 0.5693 | 0.5467 | 0.5732 | 0.5819 | 0.9271 | 9022 |
| cnn4 | mtf | 0.5316 | 0.5464 | 0.5155 | 0.5901 | 0.5391 | 0.9343 | 5306 |
| cnn3 | mtf | 0.5179 | 0.4649 | 0.4640 | 0.5410 | 0.4781 | 0.9350 | 4832 |

## 主要觀察

1. **最佳是 cnn4 + RP**（acc 0.633 / macro-F1 0.679），但與 cnn4 + GAF 只差 0.003，單 seed 下視為持平。
2. **RP ≈ GAF > MTF**：MTF 的 macro-F1 比 RP/GAF 低約 0.1–0.17，三種模型皆如此。
3. **加深有幫助，換成 ResNet34 沒有**：同表徵下 cnn4 比 cnn3 的 macro-F1 高約 0.04–0.08；ResNet34（預訓練）在 RP/GAF 上反而低於 cnn4 約 0.015–0.02，只有 MTF 上較高（0.569 vs 0.546）。ResNet34 的最終訓練 loss 約 0.07–0.11，遠低於 cnn4 的 0.32–0.42，與 test 表現並不同步，**有過擬合跡象**。
4. **macro-F1 > accuracy**：小類（AY.2、B.1.617.1、C.37 等）拉高 macro-F1；大類的 Delta 子系與泛型 lineage 互相混淆，壓低 accuracy 與 weighted-F1。
5. **ROC-AUC 0.93–0.96**：排序能力遠優於硬分類。ResNet34 的 AUC（0.94）低於 cnn3/cnn4（0.96）。
6. 僅 seed0，無法判斷 0.01 量級差異是否穩定；`../lineage_baselines_v2` 的 cnn4 + rp（raw）三個 seed 之間 macro-F1 就在 0.671–0.689 變動。

## 逐類別難易（F1，9 組 run）

- **難**（所有 run 的 F1 皆 < 0.5 或接近）：AY.103（0.01–0.39）、B.1（0.06–0.31）、B.1.1（0.00–0.23）、B.1.617.2（0.26–0.47）、AY.3（0.22–0.44）、AY.44（0.33–0.47）。皆為祖先／泛型 lineage 或 Delta 子系，序列差異小。
- **易**（各 run F1 皆 ≥ 0.85，MTF 略低）：B.1.1.519、B.1.1.7、B.1.526、P.1、B.1.351、C.37。
- **小類不穩**：AY.2（n=14）、B.1.617.1（n=14）、B.1.177（n=48）的 F1 在不同 run 間波動大（例如 AY.2：0.0–0.96），support 太少，不宜解讀。
- AY.103（n=898）在 cnn3 上近乎全錯（F1 0.01–0.05），ResNet34 才到 0.34–0.39。

## 與其他結果的對照

| 對照 | 說明 |
|---|---|
| `../lineage_baselines_v2`（raw，22 類） | 同一資料集。cnn4 + rp 三 seed macro-F1 = 0.689 / 0.671 / 0.671，acc 0.636–0.641；本目錄 cumsum 的 cnn4 + rp（seed0）為 0.679 / 0.633，**落在 raw 的 seed 變動範圍內，看不出 cumsum 有明確增益**。同目錄中 fcn（signal）macro-F1 約 0.71，高於所有影像 CNN。 |
| `../lineage_baselines_cumsum`（11 類） | 任務不同（11 vs 22 類），不可比；該版 cnn4 + gaf macro-F1 0.806 不應與此處的 0.676 比較。 |
| 論文 Table 2 | Spike7k、GISAID，類別組成不同；MTF 在論文中無對應列。 |

## 注意事項

- 超參數多數為 `chosen`（epochs=100、lr、batch、signal_mode、BN recalibration 等），詳見各 JSON 的 `provenance`。
- `cnn3 + mtf` 的 AY.2 F1 = 0、AY.103 F1 = 0.01，是 macro-F1 最低的主因。
- `train.log` 中 NumPy→tensor non-writable 警告無害；`pandas` 缺失的 traceback 只出現在最早的失敗啟動，已由後續重跑取代。
- 尚未做：多 seed（尤其 cnn4 + rp/gaf 與 resnet34）、`signal_mode=raw` 的 ResNet34 對照、GAF/RP 之外的調參。
