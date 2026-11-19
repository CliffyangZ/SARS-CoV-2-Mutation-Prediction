# 即時譜系預測器 Demo

貼上 SARS-CoV-2 spike 胺基酸序列（或 FASTA、突變清單）→ 顯示：
1. **Top-k Pango 譜系與機率**
2. **路由群組 D（Delta）／B（B.1 家族）／O（其他）**，以及專家是否套用
3. **參考分類器 vs 加 s·專家後的分數差**（殘差貢獻 = 融合 logit − 參考 logit）

模型是 `Research/QuSpike_PQK` 的 MAIN（種子 0；s 在 validation 上選，目前 0.25）。Demo 不重寫分類邏輯，只呼叫 `router`／`inference`。

## 用法（`conda activate qml_env`，於 repo 根目錄）

```bash
python -m Demo.cli build                         # 一次性，約 2–3 分鐘；產生 Demo/artifacts/（不進版控）
python -m Demo.cli predict --seq-file spike.fa --k 5
python -m Demo.cli predict --mut "L452R, T478K, del69-70" --s 0   # s=0 → 等同參考分類器
python -m Demo.cli serve --port 8000             # 開 http://127.0.0.1:8000
pytest tests/demo -q
```

`predict` 的 exit code：0 成功、2 輸入無效、3 模型包缺失。`--json` 輸出見 `specs/009-lineage-predictor-demo/contracts/web-api.md`。

## 限制與聲明

- 專家（量子路徑）的增益落在雜訊下限 δ=0.02 內，**不代表量子優勢**。
- 機率未經校準；只涵蓋資料集中的 22 個譜系，新譜系只會得到低信心結果。
- 不在訓練詞彙中的突變會被忽略（介面會列出數量）。
- 只綁 localhost，不記錄輸入內容。
