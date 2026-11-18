# Spike-PQK

用於 **SARS-CoV-2 Spike 胺基酸譜系分類**的投影量子核（projected quantum kernel, PQK）專家模型。

- 作者：楊欣澤　｜　指導老師：查怡
- 論文：[`Paper/QuSpike-PQK-cn.tex`](Paper/QuSpike-PQK-cn.tex)（中文）、[`Paper/QuSpike-PQK.tex`](Paper/QuSpike-PQK.tex)（英文）

## 摘要

本文以 SARS-CoV-2 spike 胺基酸序列的 **22 類 Pango 譜系分類**檢驗：固定的量子特徵映射，能否提供介面相同之古典替代方案以外的判別資訊？

我們提出 **Spike-PQK**：

1. 序列相對 Wuhan-Hu-1 編碼為二元突變指示向量；
2. 線性參考分類器評分並路由至譜系群組；
3. 對親緣相近的 **Delta（D）** 與 **B.1 家族（B）**，以無可訓練參數、Trotter 化 Heisenberg 演化後的單量子位元約化密度矩陣構成 PQK；
4. 核 SVM 專家對參考 logits 做殘差修正。

每個量子元件皆配有古典對應物，決策準則在評估前固定。

## 主要結果（種子 0）

資料：`Data/spike-lineage/dataset_v2.tsv`（31,689 條；時間切分 train 22,175／test 9,514）。

| 模型 | Macro-F1 | 準確率 | 備註 |
|------|----------|--------|------|
| **Spike-PQK（MAIN）** | **0.7547** | **0.6893** | 訓練約 3.09 分鐘 |
| 線性參考分類器（C0） | 0.7481 | 0.6847 | 突變指示 + 線性 |
| RFF 古典替代（C2） | 0.7552 | 0.6898 | 與 MAIN 持平 |
| 無糾纏電路（C3） | 0.7544 | 0.6891 | ΔF1 ≈ 0.0003 |
| one-hot 邏輯迴歸 | 0.7339 | — | 古典基線 |
| 最佳影像 CNN | 0.6787 | — | RP/GAF/MTF |

結論：**在此任務上 PQK 可被古典方法取代**，不主張量子優勢。相對基線的領先主要來自突變指示表徵與線性參考分類器；量子元件增益（+0.0066）落在預設雜訊下限 δ=0.02 內。

## 方法概要

```
突變向量 x → 線性參考分類器 ℓ_ref → 路由至 D / B / O
                    ↓（僅 D、B）
              LDA 投影 → Heisenberg PQK → 核 SVM 專家 f_g
                    ↓
              ℓ = ℓ_ref + s · f_g(x)   （s=0 即退回參考分類器）
```

- **D 組**：4 qubit；**B 組**：2 qubit；O 組不進量子路徑  
- 電路無可訓練參數；PQK 只需 **O(N)** 電路（非保真度核的 O(N²)）  
- 對照：C0 參考、C1 高斯核、C2 RFF、C3 無耦合、C4 MLP、C5 不分群  

## 儲存庫結構

| 路徑 | 說明 |
|------|------|
| `Paper/` | 中英文稿、圖、編譯設定 |
| `Data/spike-lineage/` | 22 類譜系資料（建議 `dataset_v2.tsv`） |
| `Data/spike-protein/` | Spike AA 靜態庫（ViralEntropR / Zenodo） |
| `Research/QuSpike_PQK/` | Spike-PQK 實驗管線 |
| `Research/lineage_baselines/` | 古典／影像／訊號基線 |
| `Results/quspike_pqk*` | PQK 實驗輸出 |
| `Results/lineage_baselines*` | 基線實驗輸出 |
| `specs/` | 功能規格與設計紀錄 |

## 快速開始

```bash
# 環境（範例）
conda activate qml_env
pip install -r requirements-extra.txt   # 依專案需求

# Spike-PQK 主實驗
python -m Research.QuSpike_PQK.run \
  --data Data/spike-lineage/dataset_v2.tsv \
  --out Results/quspike_pqk \
  --seeds 0

# 古典／影像基線（22 類）
python -m Research.lineage_baselines.run \
  --data Data/spike-lineage/dataset_v2.tsv \
  --out Results/lineage_baselines_v2 \
  --group classical --seeds 0 1 2
```

編譯論文（於 `Paper/`）：

```bash
latexmk -xelatex QuSpike-PQK-cn.tex   # 中文
latexmk -pdf QuSpike-PQK.tex          # 英文
```

詳見 [`Paper/README.md`](Paper/README.md)、[`Data/spike-lineage/README.md`](Data/spike-lineage/README.md)。

## 補充說明

- 所有 Spike-PQK 主結果目前為**單一種子**，應視為指示性結果。  
- 程式與補充資料：見論文補充資料一節（GitHub 連結以稿件為準）。  
- 本專案另含 StyleGAN／JEPA 等探索路徑，與 Spike-PQK 主線分開存放於 `Research/`、`Results/`。
