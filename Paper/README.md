# Paper — QuSpike-PQK

SARS-CoV-2 spike 譜系分類的投影量子核（PQK）論文稿。

## 目錄

| 路徑 | 說明 |
|---|---|
| `QuSpike-PQK.tex` | 英文稿（IEEEtran，`pdflatex` / `latexmk -pdf`） |
| `QuSpike-PQK-cn.tex` | 中文稿（article + xeCJK，`xelatex` / `latexmk -xelatex`） |
| `QuSpike-PQK.pdf` / `QuSpike-PQK-cn.pdf` | 編譯產出（根目錄） |
| `figures/` | 論文用圖（PDF／PNG／SVG） |
| `figures/scripts/` | 產圖腳本（`make_*.py`、`qsim.py`、`draw_sars_cov_2.py`） |
| `figures/_archive/` | 未嵌入正文的舊圖／草稿素材 |
| `references/` | 撰寫時對照的資料集／基線摘要 |
| `notes/` | 架構草稿、圖表建議、修改備註 |
| `build/` | latexmk 輔助檔（aux／log／synctex…） |

## 編譯

在 `Paper/` 下：

```bash
# 中文
latexmk -xelatex QuSpike-PQK-cn.tex

# 英文
latexmk -pdf QuSpike-PQK.tex
```

`.latexmkrc` 已將輔助檔導向 `build/`，PDF 留在本目錄。VS Code / LaTeX Workshop 請用專案內的 recipe（見 `.vscode/settings.json`）。

## 重產圖

```bash
cd figures/scripts
python make_figs.py
python make_compare_fig.py
python make_results_figs.py
python draw_sars_cov_2.py
# 圖檔直接寫入 figures/
```

## 備註

- 與中文稿完全相同的 `QuSpike-PQK-cn-opt.tex` 已刪除；請以 `QuSpike-PQK-cn.tex` 為準。
- 倉庫根目錄不再整夾忽略 `Paper/`；`build/` 等由本目錄 `.gitignore` 排除。
